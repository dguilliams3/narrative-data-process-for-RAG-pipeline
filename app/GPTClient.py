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
        max_context_tokens=GPT_MAX_CONTEXT_TOKENS
    ):
        # API + model settings
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=self.api_key)

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

    def summarize_text(self, text: str, max_tokens: int=GPT_SUMMARIZER_MAX_TOKENS, role=GPT_SUMMARIZER_ROLE, temperature=GPT_SUMMARIZER_TEMPERATURE) -> str:
        """
        Utility to summarize arbitrary text via GPT.
        """
        prompt = (
            f"""[ROLE]
            {role}
            [END ROLE]
            [TEXT-TO-SUMMARIZE]
            f"{text}
            [END TEXT-TO-SUMMARIZE]
            """
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return resp.choices[0].message.content

    def send_prompt(self, user_prompt: str):
        """
        Main user-facing call. Appends prompt to chunker,
        constructs full prompt, calls API, adds assistant reply
        back into chunker.
        """
        # Step 1: add user message
        self.chunker.add_message("User", f"{user_prompt}\n[END USER PROMPT]")

        # Step 2: assemble full prompt
        full_prompt = (
            f"""[ROLE]
            {self.role}
            [END ROLE]
            [CONVERSATION CONTEXT AND HISTORY]
            Here is the conversation so far.
            **NOTE: The user's most recent prompt is at the very end of this context.**:
            {self.chunker.get_context()}
            [END CONVERSATION CONTEXT AND HISTORY]
            Now, please reply to the user's most recent message.
            """
        )

        # Step 3: call OpenAI
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        assistant_reply = resp.choices[0].message.content
        logging.info("Assistant reply: %s", assistant_reply)

        # Step 4: add assistant reply to chunker
        self.chunker.add_message("GPT", assistant_reply)

        return resp

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

Please return just a summary of the RAG-RETRIEVED CONTENT, a robust and comprehensive one, of what you think relevant to the chat given the context. Remember that some of the RAG results may be out of scope, but also note the chronology and relevant characters and tags if they relate to the prompt and context. Note that this will be added to the context, so don't add in anything about what the prompt and context are saying are currently happening - just give a relevant summary from what was RAG-RETRIEVED.
    """.strip()
        logger.info("Fine‑tuned GPT prompt:\n%s", prompt)

        # 3) call the fine‑tuned model
        resp = self.summarizer.chat.completions.create(
            model=GPT_FINE_TUNED_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )

        # 4) extract and log the summary
        summary = resp.choices[0].message.content.strip()
        logger.info("Fine‑tuned GPT response:\n%s", summary)

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

        # 4) Wrap with your RAG header/footer
        header = "\n\n" + "-"*20 + "[RAG LORE]" + "-"*20 + "\n"
        footer = "\n" + "-"*20 + "[END RAG LORE]" + "-"*20 + "\n\n"

        # 5) Add as a single chunk into the chunker
        # (This is a single chunk, so we don't need to worry about trimming)
        self.chunker.add_message("RAG_LORE", header + combined_summaries + footer)

        # 6) Return the human‑readable version if you like
        return combined_summaries, file_names