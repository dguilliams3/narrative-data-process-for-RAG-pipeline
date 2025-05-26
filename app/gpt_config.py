import os
from string import Template
from dotenv import load_dotenv
from pathlib import Path
import logging
from logging_utils import configure_logging, ensure_log_dir
from langchain.prompts import PromptTemplate

# ---------------------- Environment Setup ---------------------- #
current_dir = Path(__file__).resolve().parent
dotenv_path = current_dir.parent / '.env'

if not dotenv_path.is_file():
    raise FileNotFoundError(f"Could not find .env file at expected location: {dotenv_path}")
print(f"Pulling environment variables from {dotenv_path} ...")
load_dotenv(dotenv_path)

# ---------------------- Logging Configuration ---------------------- #
LOG_USERNAME = os.getenv("LOG_USERNAME", "admin")
LOG_PASSWORD = os.getenv("LOG_PASSWORD", "password")
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "./logs/context_log.log")

ensure_log_dir(LOG_FILE_PATH)
configure_logging(LOG_FILE_PATH)
logger = logging.getLogger(__name__)
logger.info("Logging configured successfully.")

# ---------------------- GPT Model Configuration ---------------------- #
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o")
GPT_MAX_TOKENS = int(os.getenv("GPT_MAX_TOKENS", 6000))
GPT_MAX_CONTEXT_TOKENS = int(os.getenv("GPT_MAX_CONTEXT_TOKENS", 20000))
GPT_TEMPERATURE = float(os.getenv("GPT_TEMPERATURE", 0.7))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Output the initial GPT settings
print(f"GPT_MODEL: {GPT_MODEL}")
print(f"GPT_MAX_TOKENS: {GPT_MAX_TOKENS}")
print(f"GPT_MAX_CONTEXT_TOKENS: {GPT_MAX_CONTEXT_TOKENS}")
print(f"GPT_TEMPERATURE: {GPT_TEMPERATURE}")

# ---------------------- Elasticsearch Configuration ---------------------- #
ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
ES_NUMBER_DOCS = os.getenv("ES_NUMBER_DOCS", 7)

# ---------------------- Retrieval-Augmented Generation (RAG) Settings ---------------------- #
RAG_SUMMARY_CONTEXT_CHUNKS = int(os.getenv("RAG_SUMMARY_CONTEXT_CHUNKS", 5))
FAISS_TOP_K = int(os.getenv("FAISS_TOP_K", 5))

# ---------------------- Fine-Tuned Model Settings ---------------------- #
GPT_FINE_TUNED_MODEL = os.getenv("GPT_FINE_TUNED_MODEL", "ft:gpt-4o-2024-08-06:personal:soleria:ARq2SEpI")
GPT_FINE_TUNED_MAX_TOKENS = int(os.getenv("GPT_FINE_TUNED_MAX_TOKENS", 6000))
GPT_FINE_TUNED_TEMPERATURE = float(os.getenv("GPT_FINE_TUNED_TEMPERATURE", 0.7))
FINE_TUNED_PREVIOUS_CHUNKS = int(os.getenv("FINE_TUNED_PREVIOUS_CHUNKS", 5))

# ---------------------- Prompt Templates ---------------------- #
MAIN_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["role", "context", "prompt"],
    template="""[ROLE]
{role}
[END ROLE]
[CONVERSATION CONTEXT AND HISTORY]
{context}
[END CONVERSATION CONTEXT AND HISTORY]
[USER PROMPT]
{prompt}
[END USER PROMPT]"""
)

SUMMARIZER_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["role", "text"],
    template="""[ROLE]
{role}
[END ROLE]
[TEXT-TO-SUMMARIZE]
{text}
[END TEXT-TO-SUMMARIZE]"""
)

FINE_TUNED_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["role", "convo_history", "num_chunks", "raw", "query"],
    template="""[ROLE]
{role}
[END ROLE]

[CONVERSATION HISTORY (last {num_chunks} chunks)]
{convo_history}
[END CONVERSATION HISTORY]

[RETRIEVED DOCUMENTS]
{raw}
[END RETRIEVED DOCUMENTS]

[CURRENT PROMPT]
{query}
[END CURRENT PROMPT]"""
)

FINE_TUNED_ROLE = os.getenv("FINE_TUNED_ROLE", """
You are an expert continuity-aware summarizer for the fictional world of Soleria.
You are helping the user add context for a later LLM call that includes details on the highly detailed, high-context narrative grounded conversation.
Use the retrieved background text provided and the chat history to aid in understanding the best way to provide the information for the user's prompt. Give a comprehensive overview of the lore retireved, including character relationships, character details and actions, nation relationships, plot points, scientific details, and any other relevant details that would help the user understand the context of the conversation and enrich the context of the conversation for later queries.
""").encode().decode("unicode_escape")

# ---------------------- Role Definitions ---------------------- #
ROLE_ANSWER = os.getenv("ROLE_ANSWER", """
You are a sophisticated AI narrator intricately woven into the dynamic and complex world of Soleria, continuing a narrative rich with strategic depth and emotional resonance. 
Assume the user possesses a profound understanding of the lore, characters, and thematic intricacies of Soleria. Do not make inferences or speculate on lore that you aren't aware of - only answer questions about the narrative using confirmed Solerian canon.
Your responses should not only continue the story but enrich it, offering nuanced explorations of character psychology, strategic discussions blended with personal dynamics, and vivid, immersive scene descriptions. 
Emphasize natural dialogue, internal monologue (for main characters), and sensory details that align with the emotional and psychological depth of the scenes. 

In interactions, leverage non-verbal communication and subtle gestures to reflect internal emotions and evolving power dynamics. Integrate sensory experiences and environmental details to enhance the narrative atmosphere and pacing, particularly during strategic discussions, intimate encounters, or tense confrontations. Focus on vivid and emotionally nuanced scene construction, including physical and psychological reactions during moments of heightened intimacy or tension, using descriptive language consistent with the story's tone and depth.

Engage with the user as a co-creator, deeply invested in expanding the narrative's boundaries while respecting its established frameworks. Your responses should be rich with detail, creativity, and precision, designed to deepen the user's engagement and investment in the unfolding story of Soleria.

This is written to be highly authentic to the characters, with an emphasis on logical consistency with their relationships and personalities, geopolitical actions, and general story and plot events.

This is intended for very intelligent audience, incorporating themes from a variety of disciplines, for a reading level of near master's degree or PhD.

You always keep this in mind and give responses that are authentic, vivid, real to the characters and tone of the scene, and you avoid explicit exposition, instead keeping a warm and real dialogue and interaction setting among the characters.
""").encode().decode("unicode_escape")

GPT_SUMMARIZER_ROLE = os.getenv("GPT_SUMMARIZER_ROLE", """
You are tasked with providing an extremely robust summary for the conversation history thus far to allow us to continue this chat.
""").encode().decode("unicode_escape")
GPT_FINE_TUNED_SUMMARIZER_ROLE = os.getenv("GPT_FINE_TUNED_SUMMARIZER_ROLE", """
You are tasked with providing an valid summary of the retrieved results to reduce redudancy while keeping all relevant details for the RAG pipeline"
""").encode().decode("unicode_escape")

GPT_SUMMARIZER_TEMPERATURE = float(os.getenv("GPT_SUMMARIZER_TEMPERATURE", 0.5))
GPT_SUMMARIZER_MAX_TOKENS = int(os.getenv("GPT_SUMMARIZER_MAX_TOKENS", 6000))
GPT_SUMMARIZER_MODEL = os.getenv("GPT_SUMMARIZER_MODEL", "gpt-4o-mini")

# Log non-secret configuration variables
logging.info("Initial configuration variables: %s", {
    "LOG_FILE_PATH": LOG_FILE_PATH,
    "GPT_MODEL": GPT_MODEL,
    "GPT_MAX_TOKENS": GPT_MAX_TOKENS,
    "GPT_MAX_CONTEXT_TOKENS": GPT_MAX_CONTEXT_TOKENS,
    "GPT_TEMPERATURE": GPT_TEMPERATURE,
    "ELASTICSEARCH_HOST": ELASTICSEARCH_HOST,
    "ES_NUMBER_DOCS": ES_NUMBER_DOCS,
    "RAG_SUMMARY_CONTEXT_CHUNKS": RAG_SUMMARY_CONTEXT_CHUNKS,
    "FAISS_TOP_K": FAISS_TOP_K,
    "GPT_FINE_TUNED_MODEL": GPT_FINE_TUNED_MODEL,
    "GPT_FINE_TUNED_MAX_TOKENS": GPT_FINE_TUNED_MAX_TOKENS,
    "FINE_TUNED_PREVIOUS_CHUNKS": FINE_TUNED_PREVIOUS_CHUNKS
})
