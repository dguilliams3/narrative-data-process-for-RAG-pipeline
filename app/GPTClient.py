import time
import requests
from openai import OpenAI
import tiktoken
import logging
import os
import elasticsearch_and_faiss_query_line
from gpt_config import *
from chunk_manager import ChunkManager
from langchain.prompts import PromptTemplate
import asyncio
import aiohttp
from typing import List, Tuple, Dict, Any
from metrics_base import BaseMetricsProducer
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize logging
ensure_log_dir(LOG_FILE_PATH)
configure_logging(LOG_FILE_PATH)
logger = logging.getLogger(__name__)

class GPTServiceError(Exception):
    """Custom exception for GPT service errors"""
    def __init__(self, message, status_code=500):
        super().__init__(message)
        self.status_code = status_code

class GPTClient:
    """
    Main GPT client for handling conversations with context management.
    Uses ChunkManager for context and the common send_prompt function for all GPT calls.
    """
    def __init__(
        self,
        api_key=None,
        model=GPT_MODEL,
        max_tokens=GPT_MAX_TOKENS,
        temperature=GPT_TEMPERATURE,
        role=ROLE_ANSWER,
        max_context_tokens=GPT_MAX_CONTEXT_TOKENS,
        gpt_service_url: str = None,
        prompt_template: PromptTemplate = MAIN_PROMPT_TEMPLATE
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        self.gpt_service_url = gpt_service_url or os.getenv("GPT_SERVICE_URL")  
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.role = role
        self.prompt_template = prompt_template

        # Initialize chunk manager
        self.max_context_tokens = max_context_tokens
        self.chunker = ChunkManager(
            tokenizer_model=self.model,
            max_total_tokens=self.max_context_tokens,
            summarizer_client=self  # We can still be the summarizer since we're using the same send_prompt
        )

        # Initialize metrics producer
        self.metrics_producer = BaseMetricsProducer(
            service_name="gpt-client",
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            topic="metrics"
        )

    def count_tokens(self, text: str) -> int:
        enc = tiktoken.encoding_for_model(self.model)
        return len(enc.encode(text))

    async def send_prompt(self, user_prompt: str):
        """
        Main user-facing call. Appends prompt to chunker,
        constructs full prompt with context, calls API, adds assistant reply
        back into chunker.

        Args:
            user_prompt (str): The user's input prompt

        Returns:
            str: The assistant's reply text

        Note:
            This method also stores metadata about the response in self.last_response_metadata,
            including:
            - model: The model used for generation
            - usage: Token usage statistics (prompt_tokens, completion_tokens, total_tokens)
            - finish_reason: Why the model stopped generating
            - duration: How long the request took
        """
        role = self.role
        context = self.chunker.get_context()

        # Step 1: assemble full prompt
        full_prompt = self.prompt_template.format(
            role=role,
            context=context,
            prompt=user_prompt
        )
        
        # Log the updated full prompt
        logger.info("Full prompt: %s", full_prompt)
        
        # Step 2: add user prompt to chunker
        self.chunker.add_message("User", user_prompt)

        # Step 3: send prompt to service
        payload = {
            "prompt": full_prompt,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "api_key": self.api_key
        }

        try:
            # Send request event
            await self.metrics_producer.send_event(
                event_type="request_started",
                event_data={
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                }
            )

            start_time = time.time()
            resp = requests.post(f"{self.gpt_service_url}/generate", json=payload, timeout=30)
            resp.raise_for_status()
            response_json = resp.json()
            duration = time.time() - start_time

            if "content" not in response_json:
                logger.error("Unexpected response format from GPT service: %s", response_json)
                raise GPTServiceError("Unexpected response format from GPT service")
            
            # Store metadata for potential use
            self.last_response_metadata = {
                "model": response_json.get("model"),
                "usage": response_json.get("usage"),
                "finish_reason": response_json.get("finish_reason"),
                "duration": response_json.get("duration")
            }
            
            assistant_reply = response_json["content"]

            # Send metrics
            await self.metrics_producer.send_metric(
                metric_name="request_duration",
                metric_value=duration,
                metadata={
                    "model": self.model,
                    "prompt_tokens": response_json.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": response_json.get("usage", {}).get("completion_tokens", 0)
                }
            )

            await self.metrics_producer.send_metric(
                metric_name="total_tokens",
                metric_value=response_json.get("usage", {}).get("total_tokens", 0),
                metadata={
                    "model": self.model,
                    "prompt_tokens": response_json.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": response_json.get("usage", {}).get("completion_tokens", 0)
                }
            )

            # Send completion event
            await self.metrics_producer.send_event(
                event_type="request_completed",
                event_data={
                    "model": self.model,
                    "duration": duration,
                    "tokens": response_json.get("usage", {}).get("total_tokens", 0),
                    "finish_reason": response_json.get("finish_reason")
                }
            )

        except Exception as e:
            logger.exception("Failed to call GPT service")
            # Send error event
            await self.metrics_producer.send_event(
                event_type="request_failed",
                event_data={
                    "model": self.model,
                    "error": str(e)
                }
            )
            raise GPTServiceError(f"GPT service call failed: {str(e)}")
        
        logging.info("User prompt: " + user_prompt)
        logging.info("Assistant reply: %s", assistant_reply)
        if self.last_response_metadata:
            logging.info("Response metadata: %s", self.last_response_metadata)

        # Step 4: add assistant reply to chunker
        self.chunker.add_message("GPT", assistant_reply)

        return assistant_reply

    async def summarize_text(
        self,
        text: str,
        max_tokens: int = None,
        temperature: float = None,
        model: str = None,
        role: str = GPT_SUMMARIZER_ROLE
    ) -> str:
        """
        Summarize text using the GPT service.
        Uses SUMMARIZER_PROMPT_TEMPLATE and allows override of model parameters.

        Args:
            text (str): The text to summarize
            max_tokens (int, optional): Override default max tokens
            temperature (float, optional): Override default temperature
            model (str, optional): Override default model
            role (str, optional): Override default role

        Returns:
            str: The summarized text

        Note:
            This method uses the same GPT service as send_prompt and will store
            response metadata in self.last_response_metadata
        """
        # Format the summarization prompt
        prompt = SUMMARIZER_PROMPT_TEMPLATE.format(
            role=role,
            text=text
        )
        
        # Use provided values or instance defaults
        payload = {
            "prompt": prompt,
            "model": model or self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "api_key": self.api_key
        }

        try:
            # Send summarization event
            await self.metrics_producer.send_event(
                event_type="summarization_started",
                event_data={
                    "model": model or self.model,
                    "max_tokens": max_tokens or self.max_tokens,
                    "temperature": temperature or self.temperature
                }
            )

            start_time = time.time()
            resp = requests.post(f"{self.gpt_service_url}/generate", json=payload, timeout=30)
            resp.raise_for_status()
            response_json = resp.json()
            duration = time.time() - start_time

            if "content" not in response_json:
                logger.error("Unexpected response format from GPT service: %s", response_json)
                raise GPTServiceError("Unexpected response format from GPT service")

            # Send metrics
            await self.metrics_producer.send_metric(
                metric_name="summarization_duration",
                metric_value=duration,
                metadata={
                    "model": model or self.model,
                    "prompt_tokens": response_json.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": response_json.get("usage", {}).get("completion_tokens", 0)
                }
            )

            # Send completion event
            await self.metrics_producer.send_event(
                event_type="summarization_completed",
                event_data={
                    "model": model or self.model,
                    "duration": duration,
                    "tokens": response_json.get("usage", {}).get("total_tokens", 0),
                    "finish_reason": response_json.get("finish_reason")
                }
            )

            return response_json["content"]
        except Exception as e:
            logger.exception("Failed to summarize text")
            # Send error event
            await self.metrics_producer.send_event(
                event_type="summarization_failed",
                event_data={
                    "model": model or self.model,
                    "error": str(e)
                }
            )
            raise GPTServiceError(f"Summarization failed: {str(e)}")

    def extract_text(self, response) -> str:
        try:
            return response.choices[0].message.content
        except Exception:
            logger.exception("Failed to extract response text")
            return ""

    def process_chunks_parallel_sync(self, docs: List[Dict[str, Any]], query: str, convo_history: str) -> List[str]:
        """
        Synchronously summarize retrieved documents in parallel threads.
        Uses the configured summarizer model and token limits.
        """
        def summarize_doc(doc):
            prompt = INITIAL_DOC_PROCESSING_TEMPLATE.format(
                role=GPT_INITIAL_PROCESSOR_ROLE,
                convo_history=convo_history,
                document=doc["summary"],
                query=query
            )
            payload = {
                "prompt": prompt,
                "api_key": self.api_key,
                "model": GPT_SUMMARIZER_MODEL,
                "max_tokens": GPT_SUMMARIZER_MAX_TOKENS,
                "temperature": GPT_SUMMARIZER_TEMPERATURE
            }
            resp = requests.post(
                f"{self.gpt_service_url}/generate",
                json=payload,
                timeout=30
            )
            resp.raise_for_status()
            return resp.json().get("content", "")

        # Use as many workers as there are docs (config-driven count)
        max_workers = min(len(docs), len(docs))
        summaries = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(summarize_doc, d) for d in docs]
            for future in as_completed(futures):
                try:
                    summaries.append(future.result())
                except Exception as e:
                    logger.error("Document summarization failed: %s", e)
        return summaries

    def update_context_with_rag(self, query: str) -> Tuple[str, List[str]]:
        """
        1) FAISS -> Elastic retrieval
        2) Process individual docs with GPT-4o-mini (in parallel)
        3) Combine and deduplicate with fine-tuned model
        4) Inject into chunker

        Args:
            query (str): The search query to use for RAG

        Returns:
            tuple: (final_RAG_summary, file_names)
                - final_RAG_summary (str): The final summarized RAG content
                - file_names (list): List of filenames used in the RAG process
        """
        logger.info("Updating context with RAG")
        try:
            # 1) FAISS
            paths = elasticsearch_and_faiss_query_line.query_faiss(query)
            # 2) ES
            docs = elasticsearch_and_faiss_query_line.retrieve_documents_from_elasticsearch(paths, query)
            
            if not docs:
                logger.warning("No documents retrieved from Elasticsearch")
                return "", []

            # Get last N chunks of conversation history for context
            convo_history = self.chunker.get_last_n_chunks(FINE_TUNED_PREVIOUS_CHUNKS)
            
            # Summarize retrieved documents in parallel synchronously
            total_summaries = self.process_chunks_parallel_sync(docs, query, convo_history)
            
            if not total_summaries:
                logger.warning("No summaries generated from GPT-4o-mini")
                return "", []
            
            # Combine summaries
            combined_summaries = "\n\n".join(total_summaries)
            
            # Final deduplication step with fine-tuned model
            fine_tuned_prompt = FINE_TUNED_RETRIEVED_DOCUMENTS_TEMPLATE.format(
                role=FINE_TUNED_ROLE,
                convo_history=convo_history,
                processed_docs=combined_summaries,
                query=query
            )
            
            payload = {
                "prompt": fine_tuned_prompt,
                "api_key": self.api_key,
                "model": GPT_FINE_TUNED_MODEL,
                "max_tokens": GPT_FINE_TUNED_MAX_TOKENS,
                "temperature": GPT_FINE_TUNED_TEMPERATURE
            }
            
            try:
                resp = requests.post(f"{self.gpt_service_url}/generate", json=payload, timeout=30)
                resp.raise_for_status()
                response_json = resp.json()
                if "content" not in response_json:
                    raise GPTServiceError("Unexpected response format from GPT service")
                final_RAG_summary = response_json["content"]
            except Exception as e:
                logger.exception("Failed to process with fine-tuned model")
                raise GPTServiceError(f"Fine-tuned processing failed: {str(e)}")
            
            # Add to chunker with formatting
            header = "\n\n" + "-"*20 + "[RAG LORE]" + "-"*20 + "\n"
            footer = "\n" + "-"*20 + "[END RAG LORE]" + "-"*20 + "\n\n"
            self.chunker.add_message("RAG_LORE", header + final_RAG_summary + footer)

            return final_RAG_summary, [doc['filename'] for doc in docs]
            
        except Exception as e:
            logger.exception("Failed to update context with RAG")
            raise GPTServiceError(f"RAG update failed: {str(e)}")

    def select_model_by_token_length(self, prompt: str, max_safe_tokens_mini: int = 11000) -> str:
        """
        Selects the appropriate model based on the token count of the prompt.
        If the token count exceeds max_safe_tokens_mini, the function returns "gpt-4o-mini".
        Otherwise, it returns "gpt-4o".
        """
        logging.info("Selecting model based on prompt length")
        token_count = self.count_tokens(prompt)
        logging.info(f"Prompt token count: {token_count}")
        return "gpt-4o-mini" if token_count <= max_safe_tokens_mini else "gpt-4o"

    async def close(self):
        """Clean up resources"""
        await self.metrics_producer.close()

    def __del__(self):
        """Ensure cleanup on object destruction"""
        try:
            # Run cleanup in event loop if it exists
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.close())
            else:
                loop.run_until_complete(self.close())
        except:
            pass
