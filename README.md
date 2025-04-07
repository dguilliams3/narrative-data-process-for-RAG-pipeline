# RAG Narrative Data Management Pipeline

## Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline that transforms a large corpus of raw narrative (lore) documents into a finely tuned, context-aware AI system. The pipeline consists of several key stages:

- **Document Summarization:** Using the `facebook/bart-large-cnn` model to distill lengthy texts into concise summaries.
- **Embedding & Indexing:** Generating dense vector embeddings with SentenceTransformer (`all-MiniLM-L6-v2`) for semantic search via FAISS, and indexing enriched data in Elasticsearch.
- **QA Pair Generation:** Creating question-answer pairs using both a Hugging Face T5-based model (`valhalla/t5-base-qg-hl`) and OpenAI’s GPT-4/GPT-4o API.
- **Tagging and Metadata Enrichment:** Enhancing the data with named entity recognition using `spaCy` (`en_core_web_trf`), topic modeling via BERTopic (`all-mpnet-base-v2`), and chronology inference.
- **Fine-Tuning:** Using the curated QA dataset to fine-tune GPT-4 for highly context-aware responses.
- **Future Enhancements:** Plans for dynamic ranking, function-calling for query routing, and extended scalability.

---

## Project Structure

The project is organized into several scripts, each responsible for a specific stage in the pipeline. Below is a brief overview of the main scripts and their roles:

- **Document Summarization:**
  - `summarize_documents_with_batch_processing.py`
  - `summarize_documents_with_batch_processing_only_modified_since_last_processing.py`
- **Embedding & Indexing:**
  - `load_documents_for_sentence_transformers.py`
- **QA Pair Generation:**
  - `generate_qa_pairs.py` (Hugging Face pipeline with `valhalla/t5-base-qg-hl`)
  - `generate_qa_pairs_via_GPT.py` (OpenAI GPT-4/GPT-4o with token management and error handling)
  - `cleanup_qa_results_json.py` (Cleans up QA output JSON)
- **Tagging & Metadata Enrichment:**
  - `create_entity_tags_with_spacy_in_elasticsearch.py`
  - `enrich_narrative_data_with_roles_dependencies_and_keywords_by_file_path.py`
  - `get_word_bank_for_role_types_and_contexts.py`
  - `chronological_inference_by_file_and_folder_names_and_file_content.py`
  - `add_chronology_to_es_index.py`
- **Fine-Tuning:**
  - (Configured via fine-tuning configuration files and scripts using OpenAI’s API / Hugging Face Trainer)

---

## Detailed Pipeline Description

### 1. Document Summarization

- **Purpose:**  
  Transform lengthy, raw lore documents into concise summaries.

- **Implementation Details:**
  - **Model:** `facebook/bart-large-cnn`  
  - **Approach:**  
    - Documents are split into ~1500-character chunks.
    - Chunks with fewer than 50 tokens are skipped.
    - Batch processing is implemented via multiprocessing.
  
- **Code Snippet:**

  ```python
  from transformers import pipeline

  summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
  # Example: Summarize a document chunk
  summary = summarizer("Long text chunk from the lore document...", max_length=150, min_length=50)
  print(summary)
  ```

- **Scripts Involved:**  
  - `summarize_documents_with_batch_processing.py`
  - `summarize_documents_with_batch_processing_only_modified_since_last_processing.py`

---

### 2. Embedding & FAISS Indexing

- **Purpose:**  
  Convert summaries into dense vector embeddings for rapid semantic search.

- **Implementation Details:**
  - **Model:** `all-MiniLM-L6-v2` (SentenceTransformer)
  
- **Code Snippet:**

  ```python
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer('all-MiniLM-L6-v2')
  embeddings = model.encode(["This is a document summary..."])
  # Embeddings are then indexed in a FAISS index for similarity search.
  ```

- **Script Involved:**  
  - `load_documents_for_sentence_transformers.py`

---

### 3. Elasticsearch Integration for Summaries

- **Purpose:**  
  Store enriched summaries along with metadata (filename, last_modified, tags) in Elasticsearch for flexible and precise querying.

- **Implementation Details:**
  - **Approach:**  
    - An Elasticsearch index is created with defined mappings.
    - Bulk indexing is performed using Python scripts (utilizing `elasticsearch.helpers.bulk()`).

- **Notes:**  
  - Special attention is given to handling inconsistent document IDs; a deduplication script is planned.

---

### 4. QA Pair Generation & Cleanup

#### 4.1. Hugging Face Pipeline Approach

- **Purpose:**  
  Generate QA pairs using a T5-based model.

- **Implementation Details:**
  - **Model:** `valhalla/t5-base-qg-hl`  
  - **Technique:**  
    - Use the Hugging Face `pipeline` with PyTorch’s DataLoader for batch processing.
    - GPU acceleration enabled via `device=0`.

- **Script Involved:**  
  - `generate_qa_pairs.py`

- **Code Snippet:**

  ```python
  from transformers import pipeline
  question_generator = pipeline('question-generation', model='valhalla/t5-base-qg-hl', device=0)
  questions = question_generator("Document summary text here...")
  print(questions)
  ```

#### 4.2. GPT-Based Approach

- **Purpose:**  
  Generate QA pairs with OpenAI’s GPT-4/GPT-4o API for more complex contexts.

- **Implementation Details:**
  - **Models:** GPT-4, dynamically switching to GPT-4o if token limits are exceeded.
  - **Techniques:**
    - Token counting via `tiktoken`.
    - Splitting text into manageable chunks with `split_text_by_token_limit()`.
    - Robust error handling and retry logic with logging (outputting to `debug_log.txt`).

- **Script Involved:**  
  - `generate_qa_pairs_via_GPT.py`

- **Code Snippet:**

  ```python
  import tiktoken
  def count_tokens(text):
      encoding = tiktoken.encoding_for_model("gpt-4")
      return len(encoding.encode(text))
  
  # Call generate_questions_gpt() from the script to process document chunks.
  ```

#### 4.3. QA Output Cleanup

- **Purpose:**  
  Fix formatting errors in the QA output JSON to ensure data consistency.

- **Implementation Details:**
  - **Technique:**  
    - The function `fix_json_format` in `cleanup_qa_results_json.py` attempts direct JSON parsing and, if that fails, performs string replacements (e.g., replacing '}{' with '},{').

- **Script Involved:**  
  - `cleanup_qa_results_json.py`

- **Code Snippet:**

  ```python
  import json
  def fix_json_format(raw_text):
      try:
          return json.loads(raw_text)
      except json.JSONDecodeError:
          corrected_text = raw_text.replace('}{', '},{')
          return json.loads(corrected_text)
  ```

- **Output:**  
  - Corrected QA pairs are saved to `corrected_entries.json`; problematic entries are logged in `still_errors.json`.

---

### 5. Tagging and Metadata Enrichment

#### 5.1. Entity Extraction

- **Purpose:**  
  Extract and deduplicate named entities for metadata enrichment.

- **Implementation Details:**
  - **Model:** `en_core_web_trf` (spaCy)
  - **Technique:**  
    - Process text using `nlp(text)` from spaCy.
    - Extract entities of types `PERSON`, `ORG`, and `GPE`.
    - Deduplicate and store results in the Elasticsearch field `entity_tags`.

- **Script Involved:**  
  - `create_entity_tags_with_spacy_in_elasticsearch.py`

- **Code Snippet:**

  ```python
  import spacy
  nlp = spacy.load("en_core_web_trf")
  doc = nlp("Sample summary text about David and Durmston.")
  entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]
  print(set(entities))  # Deduplicated entities
  ```

#### 5.2. Topic Modeling

- **Purpose:**  
  Extract thematic keywords from document summaries.

- **Implementation Details:**
  - **Model:** `all-mpnet-base-v2` with BERTopic
  - **Technique:**  
    - Generate embeddings using `SentenceTransformer("all-mpnet-base-v2")`.
    - Use BERTopic to cluster embeddings (with parameters like `min_topic_size=3` and dynamic topic adjustment).
    - Extract representative keywords and store them in the Elasticsearch field `tags`.

- **Script Involved:**  
  - Integrated into the tagging process (documented in our pipeline).

#### 5.3. Chronology Tagging

- **Purpose:**  
  Infer and assign temporal metadata to documents.

- **Implementation Details:**
  - **Technique:**  
    - Use regex to extract date-like strings.
    - Reference `books_to_months_mapping.json` to map book titles to chronology ranges.
    - Apply fallback mechanisms if direct dates are not found.
  
- **Script Involved:**  
  - `chronological_inference_by_file_and_folder_names_and_file_content.py`

- **Code Snippet:**

  ```python
  import re
  pattern = r"\d{4}-\d{2}"
  dates = re.findall(pattern, "Document filename or content with date 0001-02")
  print(dates)
  ```

- **Additional Script:**  
  - `add_chronology_to_es_index.py` updates the Elasticsearch index with inferred chronology.

#### 5.4. Deduplication for Elasticsearch

- **Purpose:**  
  Standardize document IDs and merge duplicate entries.

- **Planned Resolution:**  
  - Develop a deduplication script using Python and `elasticsearch.helpers.bulk()`.

---

### 6. Fine-Tuning Process

#### 6.1. Data Curation and Preparation

- **QA Pair Generation:**  
  - **Scripts:**  
    - `generate_qa_pairs.py`
    - `generate_qa_pairs_via_GPT.py`
  - **Data Cleaning:**  
    - `cleanup_qa_results_json.py`
  - **Outcome:**  
    - QA pairs are standardized with clear labels (filename, question, answer) and enriched with metadata (chronology, entity tags).

#### 6.2. Training Process Specifics

- **Base Model:**  
  - GPT-4 is used, with dynamic fallback to GPT-4o for larger contexts.
- **Frameworks:**  
  - OpenAI’s fine-tuning API is leveraged.
  - Experiments conducted with Hugging Face Transformers’ Trainer API.
- **Environment and Tools:**  
  - OpenAI Python client, detailed logging via Python’s `logging` module (logs stored in `debug_log.txt`), and configuration files in JSON/YAML.
- **Hyperparameters:**  
  - Learning rates between 1e-5 and 5e-5.
  - Training over 3–5 epochs with early stopping to mitigate overfitting.
  
#### 6.3. Iterative Adjustments and Challenges

- **Overfitting:**  
  - Addressed by reducing epochs and implementing early stopping.
- **Noisy Data:**  
  - Handled by rigorous cleaning using `cleanup_qa_results_json.py`.
- **Token Limit Issues:**  
  - Solved by splitting long texts and switching to GPT-4o when needed.
- **Documentation and Artifacts:**  
  - Configuration files, key scripts, and logs (e.g., `debug_log.txt`) are maintained.

---

### 7. Additional Technical and Process Insights

- **Error Handling:**  
  - Robust error handling across summarization, QA extraction, and fine-tuning, including retry mechanisms and incremental saving.
- **Performance and Cost Trade-offs:**  
  - Notably, switching from GPT-4 (or GPT-4-0613) to GPT-4o when needed.
- **Iterative Refinement:**  
  - Use of intermediate JSON files (e.g., `document_summaries_temp.json`, `questions_temp.json`) for debugging.
- **Inter-Phase Dependencies:**  
  - Seamless flow from summarization to FAISS and Elasticsearch indexing, then to QA extraction, tagging, and finally fine-tuning.

---

### 8. Future Enhancements and Roadmap

- **Embedding QA Questions into FAISS:**  
  - Use models like MiniLM for embedding QA pairs.
- **Dynamic Ranking Prototypes:**  
  - Experiment with Elasticsearch scoring functions that leverage enriched metadata.
- **Function-Calling for Query Routing:**  
  - Integrate retrieval functions such as `search_FAISS()` and `search_ES()`.
- **Advanced Semantic Search:**  
  - Implement Elasticsearch’s kNN search capability.
- **Continuous Evaluation and Scaling:**  
  - Regular performance assessments, refining tagging algorithms, and scaling the system.
- **Extended Future Directions:**  
  - Adapt the pipeline to different domains.
  - Incorporate real-time feedback loops.
  - Enhance multilingual support and domain-specific customizations.

---

### 9. Summary of the Updated Map

1. **Project Initiation:**  
   - Raw lore documents are collected, forming the foundation for all subsequent processing.

2. **Document Summarization:**  
   - Raw texts are split, processed through `facebook/bart-large-cnn` with GPU acceleration, and incrementally saved using batch processing scripts.

3. **FAISS Vector Indexing:**  
   - Summaries are embedded using SentenceTransformer (`all-MiniLM-L6-v2`) and indexed in FAISS, supported by `load_documents_for_sentence_transformers.py`.

4. **Elasticsearch for Summaries:**  
   - An Elasticsearch index is created with mappings for filename, summary, modification date, and tags; bulk indexing is performed with plans for deduplication.

5. **QA Extraction & Full-Text Retrieval:**  
   - Full documents are parsed with token-aware chunking (using `tiktoken`).
   - QA pairs are generated via:
     - **Hugging Face pipeline:** `generate_qa_pairs.py` using `valhalla/t5-base-qg-hl`.
     - **GPT API approach:** `generate_qa_pairs_via_GPT.py` using GPT-4/GPT-4o with robust error handling and token management.
   - QA output is cleaned using `cleanup_qa_results_json.py` and saved to `questions_output.json`.

6. **Tagging and Metadata Enrichment:**  
   - **Entity Extraction:** Using `spaCy` with `en_core_web_trf` to extract and deduplicate entities, stored in `entity_tags`.
   - **Topic Modeling:** Using BERTopic with `SentenceTransformer("all-mpnet-base-v2")` to generate thematic keywords stored in `tags`.
   - **Chronology Tagging:** Using regex-based extraction and mapping via `books_to_months_mapping.json` in `chronological_inference_by_file_and_folder_names_and_file_content.py`, with output in `chronology_map_with_books.json`.
   - **Deduplication:** Planned script to resolve inconsistencies in document IDs.

7. **Fine-Tuning Process:**  
   - QA pairs from the generation and cleanup processes serve as the training dataset.
   - Fine-tuning is performed on GPT-4 (with fallback to GPT-4o) via OpenAI’s API and Hugging Face Trainer API.
   - Detailed logging, configuration, and hyperparameter tuning (learning rates, epochs, early stopping) are used to refine the model.
   - Challenges such as overfitting and noisy data are addressed iteratively.

8. **Additional Technical and Process Insights:**  
   - Robust error handling, iterative refinement, and inter-phase dependencies are emphasized.
   - Intermediate checkpoints ensure reliable debugging and process improvement.

9. **Future Enhancements and Roadmap:**  
   - Embedding QA pairs into FAISS, dynamic ranking prototypes, function-calling for query routing, advanced semantic search, continuous evaluation, and extended directions for cross-domain applications and multilingual support.

---

# **Scripts and Models: Quick Reference**

### Scripts
- **summarize_documents_with_batch_processing.py:**  
  Uses `facebook/bart-large-cnn` for batch summarization.
- **summarize_documents_with_batch_processing_only_modified_since_last_processing.py:**  
  Processes only modified files.
- **load_documents_for_sentence_transformers.py:**  
  Generates embeddings using SentenceTransformer (`all-MiniLM-L6-v2`) for FAISS indexing.
- **generate_qa_pairs.py:**  
  Uses Hugging Face pipeline with `valhalla/t5-base-qg-hl` to generate QA pairs.
- **generate_qa_pairs_via_GPT.py:**  
  Uses OpenAI’s GPT-4 (switching to GPT-4o as needed) with token management, logging, and error handling.
- **cleanup_qa_results_json.py:**  
  Fixes JSON formatting errors in QA output.
- **chronological_inference_by_file_and_folder_names_and_file_content.py:**  
  Infers chronology using regex and mapping files.
- **add_chronology_to_es_index.py:**  
  Updates Elasticsearch with inferred chronology.
- **create_entity_tags_with_spacy_in_elasticsearch.py:**  
  Extracts and deduplicates named entities using `en_core_web_trf`.
- **enrich_narrative_data_with_roles_dependencies_and_keywords_by_file_path.py:**  
  Enriches metadata with role and influence mapping.
- **get_word_bank_for_role_types_and_contexts.py:**  
  Maintains a word bank for standardized role types.
  
### Models
- **facebook/bart-large-cnn:** Summarization of raw documents.
- **all-MiniLM-L6-v2:** Embedding generation for semantic search (FAISS).
- **valhalla/t5-base-qg-hl:** Question-generation for QA pair creation.
- **GPT-4 / GPT-4o:** Generating QA pairs from full-text documents with dynamic token handling.
- **en_core_web_trf:** Named entity extraction for metadata enrichment.
- **all-mpnet-base-v2:** Used with BERTopic for topic modeling and extracting thematic keywords.
- **KeyBERT:** (Experimentally used for improved keyword extraction over basic BERT token classification).
