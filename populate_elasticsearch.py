import json
from elasticsearch import Elasticsearch, helpers
import os
from dotenv import load_dotenv
from elasticsearch.helpers import BulkIndexError

load_dotenv()

ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
ES_PASSWORD = os.getenv("ES_PASSWORD")

# Create Elasticsearch client
es_client = Elasticsearch(
    ELASTICSEARCH_HOST,
    basic_auth=("elastic", ES_PASSWORD),
    verify_certs=False
)

# Mapping from JSON filename to index name
json_files = {
    "summaries.json": "summaries",
}


def load_json_docs(file_path, index_name):
    """
    Generator that yields each document to be bulk-indexed.
    Assumes each line in file_path is a separate JSON object.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines

            doc = json.loads(line)
            # Grab optional custom ID if present
            doc_id = doc.pop("_id", None)
            # Remove any outer ES metadata
            doc.pop("_index", None)
            doc.pop("_score", None)
            doc.pop("_ignored", None)

            # Extract the actual content from _source if it exists
            if "_source" in doc:
                actual_body = doc["_source"]
                doc.pop("_source", None)
            else:
                actual_body = doc

            # Remove any nested "_source" keys from actual_body (if present)
            if isinstance(actual_body, dict) and "_source" in actual_body:
                actual_body.pop("_source", None)

            yield {
                "_index": index_name,
                "_id": doc_id,             # only set if we had one
                "_source": actual_body     # the cleaned doc content for ES
            }

def populate_es_index(es_client, json_files):
    for file_name, index_name in json_files.items():
        print(f"Indexing data from {file_name} into index '{index_name}'")

        # Create index if it doesn't exist
        if not es_client.indices.exists(index=index_name):
            es_client.indices.create(index=index_name)

        # Prepare generator of docs from JSON lines
        docs = load_json_docs(file_name, index_name)

        # Bulk-index them
        try:
            helpers.bulk(es_client, docs)
            print(f"Successfully indexed documents from {file_name} into '{index_name}'.")
        except BulkIndexError as bie:
            # If partial success/failure, record errors
            with open("es_bulk_errors.json", "w", encoding="utf-8") as f:
                json.dump(bie.errors, f, indent=2)
            print("Bulk indexing failed for some docs. See 'es_bulk_errors.json' for details.")


if __name__ == "__main__":
    populate_es_index(es_client, json_files)
    print("Done.")
