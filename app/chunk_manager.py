import tiktoken
import logging

class ChunkManager:
    """
    Manages breaking conversation/history into chunks,
    summarizing old chunks when token limits are reached.
    """
    def __init__(self, tokenizer_model: str, max_total_tokens: int, summarizer_client):
        """
        :param tokenizer_model: name of model for tiktoken
        :param max_total_tokens: overall context token limit
        :param summarizer_client: GPTClient-like instance to call for summarization
        """
        self.encoding = tiktoken.encoding_for_model(tokenizer_model)
        self.max_total_tokens = max_total_tokens
        self.summarizer = summarizer_client

        # Each entry is either a raw chunk (str) or a summary (str)
        self.chunks = []

    def _count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def add_message(self, speaker: str, message: str):
        """
        Add a new User/Assistant line to context. If the
        rolling total would exceed the limit, summarize the
        earliest chunk.
        """
        entry = f"{speaker}: {message}"
        entry_tokens = self._count_tokens(entry)

        # If single entry itself > max, we have to summarize it immediately:
        if entry_tokens > self.max_total_tokens:
            logging.info("Single entry too large — summarizing it directly...")
            summary = self.summarizer.summarize_text(entry, max_tokens=self.max_total_tokens)
            self.chunks.append(f"[SUMMARIZED]\n{summary}[END SUMMARY]")
            return

        # Otherwise, append and then trim if needed:
        self.chunks.append(entry)
        self._trim_context_if_needed()

    def get_context(self) -> str:
        """
        Return the full context string to prepend to the next prompt.
        """
        return "\n".join(self.chunks)

    def _trim_context_if_needed(self):
        """
        If total tokens across all chunks exceeds max_total_tokens,
        summarize the oldest raw chunk and replace it with its summary.
        """
        total = sum(self._count_tokens(c) for c in self.chunks)
        while total > self.max_total_tokens and self.chunks:
            # find first chunk that isn't already a summary
            for idx, chunk in enumerate(self.chunks):
                if not chunk.startswith("[SUMMARIZED]"):
                    logging.info("Context too large, summarizing chunk %d", idx)
                    summary = self.summarizer.summarize_text(chunk, max_tokens=self.max_total_tokens)
                    self.chunks[idx] = f"[SUMMARIZED]\n{summary}[END SUMMARY]"
                    break
            total = sum(self._count_tokens(c) for c in self.chunks)
