from custom_class import CustomGPTSimpleVectorIndex
from firebase_admin import storage
import os
from firebase_utils import db
import logging
from service_context import load_service_context

storage_url = os.getenv("FIREBASE_STORAGE_BUCKET_URL")
docs_ref = db.collection("documents")
# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def delete_doc_fn(file):
    service_context = load_service_context()

    index_name = file["index_name"]
    doc_id = file["doc_id"]
    print("index_name: ", index_name)
    print("doc_id: ", doc_id)

    try:
        # Download the index.json file from Firebase Storage
        bucket = storage.bucket(name=storage_url)
        blob = bucket.blob(f"gptIndices/{index_name}.json")

        # Check if the file exists in Firebase Storage
        if blob.exists():
            print(f"{index_name}.json exists in Firebase Storage")
            index_json_data = blob.download_as_text()
            index = CustomGPTSimpleVectorIndex.load_from_string(
                index_json_data, service_context=service_context
            )

            index.delete(doc_id)

            index_json_data = index.save_to_string()

            # Upload the JSON string to Firebase Storage
            blob = bucket.blob(f"gptIndices/{index_name}.json")
            blob.upload_from_string(index_json_data)

            print(f"{index_name}.json saved to Firebase Storage")
            return "Delete document successfully"

        else:
            print("File does not exist in Firebase Storage")

            return "Nothing to delete. Please check the index exists"

    except Exception as e:
        print("Error loading index", e)
        return "Error loading index. Inform your developer", ""
