from llama_index import (
    GPTSimpleVectorIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
)
from langchain.chat_models import ChatOpenAI
from firebase_admin import storage
import os
from firebase_utils import db
import requests
from urllib.parse import unquote, urlparse
import logging

storage_url = os.getenv("FIREBASE_STORAGE_BUCKET_URL")
docs_ref = db.collection("documents")

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def index_docs(files, index_name):
    try:
        # Defining the parameters for the index
        max_input_size = 4096
        num_outputs = 1024
        max_chunk_overlap = 20

        prompt_helper = PromptHelper(
            max_input_size,
            num_outputs,
            max_chunk_overlap,
        )

        llm_predictor = LLMPredictor(
            llm=ChatOpenAI(
                temperature=0.3, model_name="gpt-3.5-turbo", max_tokens=num_outputs
            )
        )

        service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor, prompt_helper=prompt_helper
        )

        urls = [f["url"] for f in files]
        documents = download_files_and_create_documents(urls)

        print("documents:", documents)

    except Exception as e:
        print("Error loading service context", e)
        return (
            f"Error loading service context or documents. Please inform your developer. Error: {e}",
            "",
        )

    try:
        # Download the index.json file from Firebase Storage
        bucket = storage.bucket(name=storage_url)
        blob = bucket.blob(f"gptIndices/{index_name}.json")

        # Check if the file exists in Firebase Storage
        if blob.exists():
            print(f"{index_name}.json exists in Firebase Storage")
            index_json_data = blob.download_as_text()
            index = GPTSimpleVectorIndex.load_from_string(
                index_json_data, service_context=service_context
            )

            for doc in documents:
                doc_id = doc.get_doc_id()
                extra_info = doc.extra_info
                url = extra_info["url"]

                index.insert(doc, service_context=service_context)

                for file in files:
                    if file["url"] == url:
                        document_id = file["id"]
                        doc_ref = docs_ref.document(document_id)
                        doc_ref.update(
                            {
                                "doc_id": doc_id,
                                "indexed": True,
                                "index_name": index_name,
                                "index_path": f"gptIndices/{index_name}.json",
                            }
                        )
                        break

            # Serialize the index to a JSON string
            index_json_data = index.save_to_string()

            # Upload the JSON string to Firebase Storage
            blob = bucket.blob(f"gptIndices/{index_name}.json")
            blob.upload_from_string(index_json_data)

            print(f"{index_name}.json saved to Firebase Storage")
            return "Indexed successfully"

        else:
            print("File does not exist in Firebase Storage")
            # Index file doesn't exist, so we'll create a new index from scratch

            index = GPTSimpleVectorIndex.from_documents(
                documents, service_context=service_context
            )

            for doc in documents:
                doc_id = doc.get_doc_id()
                extra_info = doc.extra_info
                url = extra_info["url"]

                for file in files:
                    if file["url"] == url:
                        document_id = file["id"]
                        doc_ref = docs_ref.document(document_id)
                        doc_ref.update(
                            {
                                "doc_id": doc_id,
                                "indexed": True,
                                "index_name": index_name,
                                "index_path": f"gptIndices/{index_name}.json",
                            }
                        )
                        break

            # Serialize the index to a JSON string
            index_json_data = index.save_to_string()

            # Upload the JSON string to Firebase Storage
            blob = bucket.blob(f"gptIndices/{index_name}.json")
            blob.upload_from_string(index_json_data)

            print(f"{index_name}.json saved to Firebase Storage")
            return "Indexed successfully"

    except Exception as e:
        print("Error loading index.json:", e)
        return "Error loading index. Inform your developer", ""


def download_files_and_create_documents(url_list, timeout=10):
    documents = []

    for url in url_list:
        # Extract the file name from the URL
        url_path = urlparse(url).path
        file_name = unquote(os.path.basename(url_path)).split("?")[0]

        # Get the portion of the file name after the '/'
        file_name_after_slash = file_name.split("/")[-1]

        try:
            # Download the file with a timeout
            response = requests.get(url, timeout=timeout)

            # Ensure the response is valid
            if response.status_code == 200:
                # Save the file to the /tmp folder in Google Cloud Functions
                tmp_file_path = f"/tmp/{file_name_after_slash}"
                with open(tmp_file_path, "wb") as f:
                    f.write(response.content)
                    logging.info(
                        f"Successfully downloaded {file_name_after_slash} to /tmp"
                    )

                # Load the file and create a Document object
                document = SimpleDirectoryReader(
                    input_files=[tmp_file_path]
                ).load_data()[0]
                document.doc_id = f"doc_id_{file_name_after_slash}"
                document.extra_info = {"url": url}
                documents.append(document)

            else:
                logging.error(
                    f"Failed to download {file_name_after_slash}: {response.status_code}"
                )
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading {file_name_after_slash}: {e}")

    return documents
