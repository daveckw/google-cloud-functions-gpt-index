import os
from flask import jsonify, make_response
from index_docs_fn import index_docs
from firebase_utils import db
from chatbot_fn import chatbot_fn
from delete_doc_fn import delete_doc_fn

api_key = os.getenv("OPENAI_API_KEY")
access_key = os.getenv("ACCESS_KEY")
os.environ["OPENAI_API_KEY"] = api_key


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
    index_name = request.args.get("index_name", "")
    if not index_name:
        response = "Please provide a valid index name"
        return make_response(
            jsonify({"response": response}),
            400,
            headers,
        )
    try:
        if input_text == access_key:
            # Fetch documents with indexed == false
            docs_ref = db.collection("documents")
            query = docs_ref.where("indexed", "==", False)
            results = query.stream()
            documents = [{"id": doc.id, **doc.to_dict()} for doc in results]

            index_docs(documents, index_name)

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


def chatbot(request):
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
        }
        return "", 204, headers
    headers = {"Access-Control-Allow-Origin": "*"}

    input_text = request.args.get("input_text", "")
    index_name = request.args.get("index_name", "")

    print(input_text)
    try:
        if input_text:
            response = chatbot_fn(input_text, index_name)
            return make_response(jsonify({"response": response}), 200, headers)
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


def delete_doc_from_index(request):
    headers = {"Access-Control-Allow-Origin": "*"}

    if request.method == "OPTIONS":
        headers.update(
            {
                "Access-Control-Allow-Methods": "GET",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Max-Age": "3600",
            }
        )
        return "", 204, headers

    document_id = request.args.get("document_id", "")
    input_text = request.args.get("input_text", "")

    if input_text == access_key:
        doc_ref = db.collection("documents").document(document_id)
        doc = doc_ref.get()

        if doc.exists:
            print(f"Document data: {doc.to_dict()}")

            response = delete_doc_fn(doc.to_dict())

            return make_response(jsonify({"response": response}), 200, headers)

        else:
            response = f"No document found with ID: {document_id}"
            print(response)
            return make_response(
                jsonify({"response": response}),
                400,
                headers,
            )

    else:
        response = "Invalid request method or missing access key."
        return make_response(
            jsonify({"response": response}),
            400,
            headers,
        )
