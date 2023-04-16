from llama_index import (
    GPTSimpleVectorIndex,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
)
from llama_index.readers import BeautifulSoupWebReader
from langchain.chat_models import ChatOpenAI
from firebase_admin import storage


def index_docs(files):
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
                temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs
            )
        )

        service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor, prompt_helper=prompt_helper
        )

        urls = [f["url"] for f in files]
        # Llamaindex documents format
        loader = BeautifulSoupWebReader()
        documents = loader.load_data(urls=urls)
        print(documents)

    except Exception as e:
        print("Error loading service context", e)
        return (
            f"Error loading service context. Please inform your developer. Error: {e}",
            "",
        )

    try:
        # Download the index.json file from Firebase Storage
        bucket = storage.bucket(name="whatsapp-api-eea64.appspot.com")
        blob = bucket.blob("gptIndices/index.json")

        # Check if the file exists in Firebase Storage
        if blob.exists():
            index_json_data = blob.download_as_text()
            index = GPTSimpleVectorIndex.load_from_string(
                index_json_data, service_context=service_context
            )

            for doc in documents:
                index.insert(doc, service_context=service_context)

        else:
            print("File does not exist in Firebase Storage")
            # Index file doesn't exist, so we'll create a new index from scratch

            index = GPTSimpleVectorIndex.from_documents(
                documents, service_context=service_context
            )

            # Serialize the index to a JSON string
            index_json_data = index.save_to_string()

            # Upload the JSON string to Firebase Storage
            blob = bucket.blob("gptIndices/index.json")
            blob.upload_from_string(index_json_data)

            print("Index saved to Firebase Storage")

    except Exception as e:
        print("Error loading index.json:", e)
        return "Error loading index. Inform your developer", ""

    return "Indexed successfully"
