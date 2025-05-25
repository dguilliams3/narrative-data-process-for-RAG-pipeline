import unittest
from unittest.mock import MagicMock
from chunk_manager import ChunkManager
import tiktoken
import logging
from logging_utils import configure_logging, ensure_log_dir
import os

# Set up logging
LOG_FILE = os.path.join(os.path.dirname(__file__), "logs", "test_chunk_manager.log")
ensure_log_dir(LOG_FILE)
configure_logging(LOG_FILE)
logger = logging.getLogger(__name__)

class TestChunkManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up logging for the test class"""
        logger.info("\n\n=== Starting ChunkManager Tests ===")

    def setUp(self):
        logger.info("\n--- Setting up new test ---")
        # Mock the tokenizer and summarizer
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.encode = lambda text: text.split()  # Mock tokenization by splitting on spaces
        self.mock_summarizer = MagicMock()
        self.mock_summarizer.summarize_text = lambda text, max_tokens: f"Summary of: {text[:max_tokens]}"

        # Patch tiktoken to return the mock tokenizer
        tiktoken.encoding_for_model = MagicMock(return_value=self.mock_tokenizer)

        # Initialize ChunkManager
        self.chunk_manager = ChunkManager(
            tokenizer_model="mock-model",
            max_total_tokens=10,
            summarizer_client=self.mock_summarizer
        )
        logger.info("ChunkManager initialized with mock_model and max_tokens=10")

    def test_add_message_within_limit(self):
        logger.info("Running test: add_message_within_limit")
        self.chunk_manager.add_message("User", "Hello world")
        logger.info(f"Current chunks: {self.chunk_manager.chunks}")
        self.assertEqual(self.chunk_manager.chunks, ["User: Hello world"])
        logger.info("Test passed: Message added successfully within limit")

    def test_add_message_exceeds_limit(self):
        logger.info("Running test: add_message_exceeds_limit")
        test_message = "This is a very long message that exceeds the token limit"
        self.chunk_manager.add_message("User", test_message)
        logger.info(f"Current chunks after adding long message: {self.chunk_manager.chunks}")
        self.assertTrue(self.chunk_manager.chunks[0].startswith("[SUMMARIZED]"))
        self.assertTrue(self.chunk_manager.chunks[0].endswith("[END SUMMARY]"))
        logger.info("Test passed: Long message was properly summarized")

    def test_add_message_triggers_trim(self):
        logger.info("Running test: add_message_triggers_trim")
        messages = [
            ("User", "Message one"),
            ("Assistant", "Message two"),
            ("User", "Message three"),
            ("Assistant", "Message four")
        ]
        
        for role, message in messages:
            logger.info(f"Adding message - {role}: {message}")
            self.chunk_manager.add_message(role, message)
            logger.info(f"Current chunks: {self.chunk_manager.chunks}")
        
        self.assertTrue(any(chunk.startswith("[SUMMARIZED]") for chunk in self.chunk_manager.chunks))
        logger.info("Test passed: Messages were properly trimmed and summarized")

    def tearDown(self):
        logger.info("Cleaning up test\n")

    @classmethod
    def tearDownClass(cls):
        logger.info("=== Completed All ChunkManager Tests ===\n")

if __name__ == "__main__":
    logger.info("Starting test suite execution")
    unittest.main()