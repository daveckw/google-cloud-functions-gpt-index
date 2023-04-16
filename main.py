import os
from flask import jsonify, make_response
import firebase_admin
from firebase_admin import firestore
from index_docs_fn import index_docs

# Get the value of OPENAI_API_KEY from the environment
api_key = os.getenv("OPENAI_API_KEY")
storage_url = os.getenv("FIREBASE_STORAGE_BUCKET_URL")
access_key = os.getenv("ACCESS_KEY")
# Use the API key in your code
os.environ["OPENAI_API_KEY"] = api_key

FIREBASE_STORAGE_BUCKET_URL = "whatsapp-api-eea64.appspot.com"
firebase_admin.initialize_app(options={"storageBucket": FIREBASE_STORAGE_BUCKET_URL})
db = firestore.client()


def index_documents(request):
    # Set CORS headers for the preflight request
    if request.method == "OPTIONS":
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for a 3600s
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
        }
        return "", 204, headers

    # Set CORS headers for the main request
    headers = {"Access-Control-Allow-Origin": "*"}

    input_text = request.args.get("input_text", "")
    try:
        if input_text == access_key:
            # Fetch documents with indexed == false
            docs_ref = db.collection("documents")
            query = docs_ref.where("indexed", "==", False)
            results = query.stream()
            documents = [{"id": doc.id, **doc.to_dict()} for doc in results]
            index_docs(documents)

            return make_response(jsonify({"documents": documents}), 200, headers)
        else:
            response = "Please provide a valid input text"
            return make_response(
                jsonify({"response": response}),
                400,
                headers,
            )
    except KeyError as e:
        print(f"KeyError encountered: {e} {input_text}")
        response = "An error occurred. Please try again later."
    return make_response(
        jsonify({"response": response}),
        500,  # Change this status code to 500 to indicate a server-side error
        headers,
    )
