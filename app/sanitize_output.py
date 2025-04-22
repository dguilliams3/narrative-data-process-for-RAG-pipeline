import os
import json
import re

REPLACEMENTS = json.loads(os.getenv("REPLACEMENTS_JSON", "{}"))

def sanitize_text(text: str) -> str:
    if REPLACEMENTS:
        for pattern, replacement in REPLACEMENTS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
