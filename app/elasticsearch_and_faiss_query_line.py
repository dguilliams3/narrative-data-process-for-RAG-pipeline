import os, json
import numpy as np
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
# from transformers import pipeline
# from transformers import BertTokenizer, BertModel, BertForTokenClassification
from logging_utils import *
from keybert import KeyBERT
# import torch
import faiss


from gpt_config import *

SENTANCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'
# # Load tokenizer and model with the specified device
# tokenizer = AutoTokenizer.from_pretrained('distilbert-base-nli-mean-tokens')
# model = AutoModel.from_pretrained('distilbert-base-nli-mean-tokens')

# # Check if GPU is available and move the model to GPU
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

# Initialize KeyBERT with the specified device
kw_model = KeyBERT(model='distilbert-base-nli-mean-tokens')

# BERT/transformers-based keyword extraction and QA are disabled for now:
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertModel.from_pretrained('bert-base-uncased')
# device = 0 if torch.cuda.is_available() else "cpu"
# logging.info(f"Device: {device}")
device = "cpu"
model = SentenceTransformer(SENTANCE_TRANSFORMER_MODEL)

# Elasticsearch setup - replace with your actual endpoint
es = Elasticsearch(ELASTICSEARCH_HOST, basic_auth=("elastic", os.getenv("ES_PASSWORD")), verify_certs=False)

# Check if GPU is available for encoding queries
# device = 0 if torch.cuda.is_available() else "cpu"
logging.info(f"Device: {device}")

# Load the SentenceTransformer model for encoding queries
# model = SentenceTransformer(SENTANCE_TRANSFORMER_MODEL, device=device)

def find_file(filename):
    """Search for FAISS index file in the project directory and its subdirectories."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up one level from 'app'
        
    for root, dirs, files in os.walk(project_root):
        if filename in files:
            return os.path.join(root, filename)
    return None

# Load the FAISS index and filename mapping
faiss_path = find_file("faiss_index.bin")
if faiss_path:
    logging.info(f"Found FAISS index at: {faiss_path}")
    index = faiss.read_index(faiss_path)
else:
    raise FileNotFoundError("Could not find faiss_index.bin in project directory")

with open(find_file("index_to_filename.json"), "r", encoding="utf-8") as f:
	index_to_filename = json.load(f)  # Updated filename mapping
	logging.info("Loaded index_to_filename mapping from location {index_to_filename}")

# Function to query FAISS and return document paths
def query_faiss(query, top_k=FAISS_TOP_K):
	# Step 1: Encode the query into an embedding
	query_embedding = model.encode([query])
	logging.info(f"Query embedding shape: {query_embedding.shape}")
	
	# Step 2: Run FAISS query to get top K indices and distances
	distances, indices = index.search(np.array(query_embedding), top_k)
	
	# Print distances and indices for debugging
	logging.info(f"Distances: {distances}")
	logging.info(f"Indices: {indices}")
	
	# Check and convert indices to strings
	faiss_results = []
	for idx in indices[0]:
		idx_str = str(idx)
		if idx_str in index_to_filename:
			faiss_results.append(index_to_filename[idx_str])
		else:
			logging.info(f"Index {idx_str} not found in index_to_filename")

	logging.info("\nFAISS Retrieved Document Paths:")
	for doc_path in faiss_results:
		logging.info(doc_path)  # Print each document path retrieved by FAISS
	
	return faiss_results

def retrieve_documents_from_elasticsearch(faiss_docs, prompt):
    # Try to extract keywords from the prompt using KeyBERT
    # keywords = extract_keywords_with_keybert(prompt)
    keywords = []
    if not keywords:
        # Fallback: use filenames from FAISS results
        filenames = [os.path.basename(doc_path) for doc_path in faiss_docs]
        logging.info("-" * 40 + "\nUsing fallback: querying Elasticsearch by filename.")
        for filename in filenames:
            logging.info(f"Filename being queried in Elasticsearch: {filename}")
        es_query = {
            "_source": True,
            "query": {
                "bool": {
                    "should": [
                        {"match_phrase": {"file_path": filename}} for filename in filenames
                    ],
                    "minimum_should_match": 1
                }
            },
            "size": len(faiss_docs)
        }
    else:
        logging.info("-" * 40 + "\nUsing KeyBERT keywords for Elasticsearch query.")
        logging.info(f"Extracted Keywords: {keywords}")
        es_query = {
            "_source": True,
            "query": {
                "bool": {
                    "should": [
                        {"match": {"summary": keyword}} for keyword in keywords
                    ],
                    "minimum_should_match": 1
                }
            },
            "size": ES_NUMBER_DOCS  # Adjust size as needed
        }
    
    logging.info(f"Elasticsearch query:\n{es_query}\n" + "-" * 20)
    
    try:
        response = es.search(index="summaries", body=es_query)
        logging.info(f"Number of documents retrieved: {len(response['hits']['hits'])}")
        documents = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            documents.append({
                'filename': source.get('filename', 'Unknown'),
                'summary': source.get('summary', 'No summary available')
            })
            logging.info("Returned document from Elasticsearch: %s", documents[-1]['summary'][:1000])
        return documents
    except Exception as e:
        logging.info("Error retrieving documents from Elasticsearch: %s", e)
        return []

# def retrieve_documents_from_elasticsearch(prompt):
# 	keywords = extract_keywords_with_keybert(prompt)
# 	es_query = {
# 		"_source": True,  # This fetches all fields from the documents
# 		"query": {
# 			"bool": {
# 				"should": [
# 					{"match": {"summary": keyword}} for keyword in keywords
# 				],
# 				"minimum_should_match": 1
# 			}
# 		},
# 		"size": 10
# 	}
	
# 	try:
# 		response = es.search(index="summaries", body=es_query)
# 		logging.info(f"Number of documents retrieved: {len(response['hits']['hits'])}")
# 		# documents = [{'filename': hit["_source"].get('filename', 'Unknown'), 
# 		# 			  'summary': hit["_source"].get('summary', 'No summary available')} for hit in response['hits']['hits']]
# 		documents = []
# 		for hit in response["hits"]["hits"]:
# 			source = hit["_source"]
# 			documents.append({
# 				'filename': source.get('filename', 'Unknown'),
# 				'summary': source.get('summary', 'No summary available')
# 			})
# 		return documents
# 	except Exception as e:
# 		logging.info("Error retrieving documents from Elasticsearch: %s", e)
# 		return []

## BERT/transformers-based QA pipeline is disabled for now:
# def test_qa_pipeline(): # Run a test query
#     # Load the QA pipeline on the selected device
#     qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", device=device)
#     query = "How does Dalakash influence Soleria's governance?"
#
#     # Retrieve documents based on the test query
#     faiss_results = query_faiss(query)
#     es_results = retrieve_documents_from_elasticsearch(faiss_results)
#
#     # Print the results for inspection
#     logging.info("\nRetrieved Documents and Summaries:")
#     for filename, summary in es_results: 
#         logging.info(f"Filename: {filename}")
#         logging.info(f"Summary: {summary}\n{'-'*60}\n")
#
#     logging.info(f"\n{'-'*100}\nAttempts to answer question:\n{query}")
#     # Perform QA on retrieved documents
#     for i, (filename, summary) in enumerate(es_results):
#         logging.info(f"\n{'-'*40}\nResult {i + 1}: {filename}")
#         doc_content = summary or "Summary not available"
#         answer = qa_pipeline(question=query, context=doc_content)
#         # Print the answer
#         logging.info(f"Answer: {answer['answer']} (Confidence: {answer['score']:.2f})")
#
#         if answer['score'] < 0.5:
#             logging.info("Low confidence in answer; additional context may be required.")
#         
#         logging.info(f"\n{'-'*40}")

def extract_keywords_with_keybert(text, top_n=5):
	logging.info(f"\n\n{'-'*40}\nEXTRACTING KEYWORDS WITH KEYBERT...\nText:\n{text}\n\n")
	keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', use_maxsum=True, top_n=top_n)
	logging.info(f"\n\nKEYWORDS FROM BERT:\n{keywords}\n\n{'-'*40}")
	return [kw[0] for kw in keywords]  # Return only the keywords