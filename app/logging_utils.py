import logging
import os

DEFAULT_LOG_FILE_PATH = "./app.log"
# Handle any unexpected characters
class UTF8Formatter(logging.Formatter):
    def format(self, record):
        # Ensure the message is encoded in UTF-8
        if isinstance(record.msg, str):
            record.msg = record.msg.encode('utf-8', 'replace').decode('utf-8')
        return super().format(record)

def configure_logging(log_file_path=DEFAULT_LOG_FILE_PATH):
  # Remove any previously configured handlers
  for handler in logging.root.handlers[:]:
      logging.root.removeHandler(handler)
      
  # Configure logging
  logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s - %(levelname)s - %(message)s',
      handlers=[
          logging.FileHandler(log_file_path, encoding='utf-8'),
          logging.StreamHandler()
      ]
  )

  # Set the custom formatter for all handlers
  for handler in logging.getLogger().handlers:
      handler.setFormatter(UTF8Formatter())

def ensure_log_dir(path=DEFAULT_LOG_FILE_PATH):
    log_dir = os.path.dirname(path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)