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

    def count_tokens(self, text: str) -> int:
        enc = tiktoken.encoding_for_model(self.model)
        return len(enc.encode(text))

    def send_prompt(self, user_prompt: str):
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
            resp = requests.post(f"{self.gpt_service_url}/generate", json=payload, timeout=30)
            resp.raise_for_status()
            response_json = resp.json()
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
        except Exception as e:
            logger.exception("Failed to call GPT service")
            raise GPTServiceError(f"GPT service call failed: {str(e)}")
        
        logging.info("User prompt: " + user_prompt)
        logging.info("Assistant reply: %s", assistant_reply)
        if self.last_response_metadata:
            logging.info("Response metadata: %s", self.last_response_metadata)

        # Step 4: add assistant reply to chunker
        self.chunker.add_message("GPT", assistant_reply)

        return assistant_reply

    def summarize_text(
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
            resp = requests.post(f"{self.gpt_service_url}/generate", json=payload, timeout=30)
            resp.raise_for_status()
            response_json = resp.json()
            if "content" not in response_json:
                logger.error("Unexpected response format from GPT service: %s", response_json)
                raise GPTServiceError("Unexpected response format from GPT service")
            return response_json["content"]
        except Exception as e:
            logger.exception("Failed to summarize text")
            raise GPTServiceError(f"Summarization failed: {str(e)}")

    def extract_text(self, response) -> str:
        try:
            return response.choices[0].message.content
        except Exception:
            logger.exception("Failed to extract response text")
            return ""

    def update_context_with_rag(self, query: str) -> str:
        """
        1) FAISS -> Elastic retrieval
        2) Combine raw docs
        3) Summarize via fine‑tuned GPT
        4) Inject into chunker

        Args:
            query (str): The search query to use for RAG

        Returns:
            tuple: (final_RAG_summary, file_names)
                - final_RAG_summary (str): The final summarized RAG content
                - file_names (list): List of filenames used in the RAG process

        Note:
            This method makes multiple calls to the GPT service and will store
            metadata from the final summarization call in self.last_response_metadata
        """
        logger.info("Updating context with RAG")
        # 1) FAISS
        paths = elasticsearch_and_faiss_query_line.query_faiss(query)
        # 2) ES
        docs = elasticsearch_and_faiss_query_line.retrieve_documents_from_elasticsearch(paths, query)
        total_summaries = []
        file_names = []
        
        for doc in docs:
            filename = doc['filename']
            text = doc['summary']

            # Get last N chunks of conversation history for context
            convo_history = self.chunker.get_last_n_chunks(FINE_TUNED_PREVIOUS_CHUNKS)

            # 2) Summarize this one document using fine-tuned config
            prompt = FINE_TUNED_PROMPT_TEMPLATE.format(
                role=FINE_TUNED_ROLE,
                convo_history=convo_history,
                num_chunks=FINE_TUNED_PREVIOUS_CHUNKS,
                raw=text,
                query=query
            )

            logger.debug("Prompt: %s", prompt)

            payload = {
                "prompt": prompt,
                "api_key": self.api_key,
                "model": GPT_FINE_TUNED_MODEL,
                "max_tokens": GPT_FINE_TUNED_MAX_TOKENS,
                "temperature": GPT_FINE_TUNED_TEMPERATURE
            }

            logger.info("Sending payload to fine-tuned GPT...")
            try:
                resp = requests.post(f"{self.gpt_service_url}/generate", json=payload, timeout=30)
                resp.raise_for_status()
                response_json = resp.json()
                if "content" not in response_json:
                    logger.error("Unexpected response format from GPT service: %s", response_json)
                    raise GPTServiceError("Unexpected response format from GPT service")
                doc_summary = response_json["content"]
                logger.info("Summary for %s:\n%s", filename, doc_summary)
                total_summaries.append(doc_summary)
                file_names.append(filename)
            except Exception as e:
                logger.exception("Failed to summarize document %s", filename)
                continue

        # 3) Combine all those mini‑summaries into one big blob
        combined_summaries = "\n\n".join(total_summaries)

        # 4) Wrap with our RAG header/footer
        header = "\n\n" + "-"*20 + "[RAG LORE]" + "-"*20 + "\n"
        footer = "\n" + "-"*20 + "[END RAG LORE]" + "-"*20 + "\n\n"

        # 5) Summarize the combined summaries to remove duplicates    
        logger.info("Summarizing combined summaries...")
        prompt = SUMMARIZER_PROMPT_TEMPLATE.format(
            role=GPT_FINE_TUNED_SUMMARIZER_ROLE,
            text=combined_summaries
        )
        logger.debug("Prompt: %s", prompt)
        
        payload = {
            "prompt": prompt,
            "api_key": self.api_key,
            "model": GPT_SUMMARIZER_MODEL,
            "max_tokens": GPT_SUMMARIZER_MAX_TOKENS,
            "temperature": GPT_SUMMARIZER_TEMPERATURE
        }
        
        logger.info("Sending payload to GPT service...")
        try:
            resp = requests.post(f"{self.gpt_service_url}/generate", json=payload, timeout=30)
            resp.raise_for_status()
            response_json = resp.json()
            if "content" not in response_json:
                logger.error("Unexpected response format from GPT service: %s", response_json)
                raise GPTServiceError("Unexpected response format from GPT service")
            final_RAG_summary = response_json["content"]
            logger.info("Final RAG summary: %s", final_RAG_summary)
        except Exception as e:
            logger.exception("Failed to summarize combined summaries")
            raise GPTServiceError(f"RAG summarization failed: {str(e)}")

        # 5) Add as a single chunk into the chunker
        # (This is a single chunk, so we don't need to worry about trimming)    
        self.chunker.add_message("RAG_LORE", header + final_RAG_summary + footer)

        # 6) Return the human‑readable version & file names used if we want to track it later
        return final_RAG_summary, file_names

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
