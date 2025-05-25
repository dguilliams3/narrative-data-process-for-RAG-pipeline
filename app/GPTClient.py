import time
import requests
from openai import OpenAI
import tiktoken
import logging
import os
import elasticsearch_and_faiss_query_line
from gpt_config import *
from chunk_manager import ChunkManager

# Initialize logging
ensure_log_dir(LOG_FILE_PATH)
configure_logging(LOG_FILE_PATH)
logger = logging.getLogger(__name__)

class GPTClient:
    def __init__(
        self,
        api_key=None,
        model=GPT_MODEL,
        max_tokens=GPT_MAX_TOKENS,
        temperature=GPT_TEMPERATURE,
        role=ROLE_ANSWER,
        max_context_tokens=GPT_MAX_CONTEXT_TOKENS,
          gpt_service_url: str = None
    ):
        # API + model settings
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=self.api_key)
        
          # If you set GPT_SERVICE_URL, we’ll proxy all calls there instead of direct OpenAI.
        self.gpt_service_url = gpt_service_url or os.getenv("GPT_SERVICE_URL")  

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.role = role

        # RAG & chunk manager
        self.max_context_tokens = max_context_tokens
        # Use a separate GPTClient instance for summarization, to avoid recursion
        self.summarizer = OpenAI(api_key=self.api_key)
        self.chunker = ChunkManager(
            tokenizer_model=self.model,
            max_total_tokens=self.max_context_tokens,
            summarizer_client=self
        )

    def count_tokens(self, text: str) -> int:
        enc = tiktoken.encoding_for_model(self.model)
        return len(enc.encode(text))

    def summarize_text(self, text: str, max_tokens: int=GPT_SUMMARIZER_MAX_TOKENS, role=GPT_SUMMARIZER_ROLE, model=None ,temperature=GPT_SUMMARIZER_TEMPERATURE) -> str:
        """
        Utility to summarize arbitrary text via GPT.
        """
        prompt = (
f"""
[ROLE]
{role}
[END ROLE]
[TEXT-TO-SUMMARIZE]
{text}
[END TEXT-TO-SUMMARIZE]
"""
        )
        
        # Allow the possible input of a specific model, else use the client's assigned model
        model = model if model else self.model
        
        resp = self.send_OpenAI_request(prompt=prompt, client=self.summarizer, model=model, max_tokens=max_tokens, temperature=temperature)
        return resp

    def send_prompt(self, user_prompt: str):
        """
        Main user-facing call. Appends prompt to chunker,
        constructs full prompt, calls API, adds assistant reply
        back into chunker.
        """
        role=self.role
        context=self.chunker.get_context()

        # Step 1: assemble full prompt
        full_prompt = GPT_DEFAULT_PROMPT.substitute(
			role=role,
			context=context,
			user_prompt=user_prompt
        )
        
        # Step 2: add user prompt to chunker
        self.chunker.add_message("User", user_prompt)

        assistant_reply = self.send_OpenAI_request(prompt=full_prompt, client=self.client, model=self.model, max_tokens=self.max_tokens, temperature=self.temperature)
        logging.info("User prompt" + user_prompt)
        logging.info("Assistant reply: %s", assistant_reply)

        # Step 4: add assistant reply to chunker
        self.chunker.add_message("GPT", assistant_reply)

        return assistant_reply

    def extract_text(self, response) -> str:
        try:
            return response.choices[0].message.content
        except Exception:
            logger.exception("Failed to extract response text")
            return ""

    def summarize_with_fine_tuned_gpt(
            self,
            raw: str,
            query:str = "",
            max_tokens: int =GPT_FINE_TUNED_MAX_TOKENS,
            prev_chunks: int = FINE_TUNED_PREVIOUS_CHUNKS,
            temperature: float = GPT_FINE_TUNED_TEMPERATURE
        ) -> str:
        """
        Summarize the retrieved documents with the fine‑tuned model,
        including up to `prev_chunks` of the most recent conversation history.
        """
        # 1) grab the last N chunks of user+assistant messages
        # EXPLANATION: slicing a Python list handles "too many" gracefully
        all_chunks   = self.chunker.chunks
        selected     = all_chunks[-prev_chunks:]
        convo_history = "\n".join(selected)

        # 2) build the inline‑role prompt
        prompt = f"""
[ROLE]
{FINE_TUNED_ROLE}
[END ROLE]

[CONVERSATION HISTORY (last {len(selected)} chunks)]
{convo_history}
[END CONVERSATION HISTORY]

[RETRIEVED DOCUMENTS]
{raw}
[END RETRIEVED DOCUMENTS]

[CURRENT PROMPT]
{query}
[END CURRENT PROMPT]
    """.strip()
        logger.info("\n\n[FINE-TUNED GPT PROMPT]\n%s", prompt + "\n[END FINE‑TUNED GPT PROMPT]\n")
        # Note that we send the client as self.summarizer rather than self.client for this call
        summary = self.send_OpenAI_request(prompt=prompt, client=self.summarizer, model=GPT_FINE_TUNED_MODEL, max_tokens=max_tokens, temperature=temperature)        

        # Extract and log the summary
        logger.info("[FINE-TUNED GPT RESPONSE]:\n%s", summary + "\n[END FINE‑TUNED GPT RESPONSE]\n")

        return summary

    def update_context_with_rag(self, query: str) -> str:
        """
        1) FAISS -> Elastic retrieval
        2) Combine raw docs
        3) Summarize via fine‑tuned GPT
        4) Inject into chunker
        """
        # 1) FAISS
        paths = elasticsearch_and_faiss_query_line.query_faiss(query)
        # 2) ES
        docs = elasticsearch_and_faiss_query_line.retrieve_documents_from_elasticsearch(paths, query)
        total_summaries = []
        file_names = []
        
        for doc in docs:
            filename = doc['filename']
            text     = doc['summary']

            # 2) Summarize this one document
            doc_summary = self.summarize_with_fine_tuned_gpt(text, query)
            logger.info("Summary for %s:\n%s", filename, doc_summary)
            total_summaries.append(doc_summary)
            file_names.append(filename)

        # 3) Combine all those mini‑summaries into one big blob
        combined_summaries = "\n\n".join(total_summaries)

        # 4) Wrap with our RAG header/footer
        header = "\n\n" + "-"*20 + "[RAG LORE]" + "-"*20 + "\n"
        footer = "\n" + "-"*20 + "[END RAG LORE]" + "-"*20 + "\n\n"

        # 5) Summarize the combined summaries to remove duplicates    
        final_RAG_summary = self.summarize_text(combined_summaries, role=GPT_FINE_TUNED_SUMMARIZER_ROLE, model=GPT_SUMMARIZER_MODEL)

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
      
    def send_OpenAI_request(
        self,
        prompt: str,
        client=None,
        model=None,
        max_tokens=None,
        temperature=None
    ):
        """
        Sends an OpenAI request with optional overrides. Defaults to self.* if not provided.
        Logs duration, token usage, and handles token-limit retry fallback.
        """
        client = client or self.client
        model = model or self.model
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature

        # If we have a remote GPT service, call that instead
        if self.gpt_service_url:
            payload = {
                "prompt": prompt,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "api_key": self.api_key
            }
            try:
                for attempt in range(3):  # Retry up to 3 times
                    try:
                        resp = requests.post(f"{self.gpt_service_url}/generate", json=payload, timeout=30)
                        resp.raise_for_status()
                        return resp.json()["content"]
                    except requests.exceptions.HTTPError as e:
                        if resp.status_code == 502:  # Handle 502 Bad Gateway
                            logger.warning(f"Attempt {attempt + 1}: Received 502 Bad Gateway. Retrying...")
                            time.sleep(2 ** attempt)  # Exponential backoff
                        else:
                            logger.exception("HTTP error occurred")
                            raise
                    except Exception as e:
                        logger.exception("Unexpected error occurred during request")
                    raise
            except requests.exceptions.RequestException as e:
                raise RuntimeError("Failed to get a valid response after 3 attempts")
            
            resp.raise_for_status()
            return resp.json()["content"]

        """
        If we aren't using a remote GPT service, call the OpenAI API directly and log the call.
        """
        # Define a nested function to log OpenAI call details, easier for scoping for now
        def _log_openai_call(m):
            logger.info(
                "[OPENAI CALL] model=%s max_tokens=%d temp=%.2f prompt_len=%d, prompt_tokens=%d",
                m, max_tokens, temperature, len(prompt), self.count_tokens(prompt)
            )

        _log_openai_call(model)

        try:
            start = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            duration = time.time() - start
            resp = resp.choices[0].message.content.strip()        
            total_tokens = self.count_tokens(prompt)

            logger.info(
                "[OPENAI CALL DURATION] %.2fs, tokens_used=%s",
                duration,
                total_tokens if total_tokens is not None else "n/a"
            )

            return resp

        except openai.BadRequestError as e:
            msg = str(e)
            if (
                "max_tokens is too large" in msg and
                "This model supports at most 16384" in msg and
                model == "gpt-4o-mini"
            ):
                # Log fallback and retry
                logger.warning("Token limit exceeded for gpt-4o-mini. Retrying with gpt-4o.")
                return self.send_OpenAI_request(
                    prompt=prompt,
                    client=client,
                    model="gpt-4o",
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            else:
                logger.exception("OpenAI BadRequestError not related to token size or model was not gpt-4o-mini.")
                raise

        except Exception as e:
            logger.exception("OpenAI API call failed")
            raise
