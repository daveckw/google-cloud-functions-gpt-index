# Document Indexing Service using ChatGPT 3.5 Turbo

This service is designed to index documents using the GPT-3.5-turbo model from OpenAI. It fetches documents from a Firebase Firestore database and performs indexing using the `llama_index` library. The indexed data is then stored in a Firebase Storage bucket.

## Requirements

- Python 3.10 or later
- `flask` library
- `firebase_admin` library
- `llama_index` library
- `BeautifulSoup` library
- `langchain` library

## Files

### main.py

This file contains the Flask server code and the `index_documents` function. It listens for incoming HTTP requests and performs document indexing based on the provided input text. If the input text is "your-password", the server fetches non-indexed documents from the Firestore database, calls the `index_docs` function, and returns the indexed documents in the HTTP response.

### index_docs_fn.py

This file contains the `index_docs` function, which is responsible for indexing the documents. It uses the `llama_index` library to create a GPTSimpleVectorIndex and indexes the documents using the GPT-3.5-turbo model. The BeautifulSoupWebReader is used to read the documents, which are then inserted into the GPTSimpleVectorIndex. The resulting index is saved in a Firebase Storage bucket.

### Setup

1. Install the required libraries:

`requirements.txt` as follows:
```bash
llama_index==0.5.15
langchain
python-dotenv
Flask==2.2.3
gunicorn==20.1.0
transformers
firebase_admin
bs4
fake_useragent
```

### Google Cloud Function

`gcloud auth login`
`gcloud config set project your-firebase-project-id`

### Deploy to Google Cloud Function using command line below

`gcloud functions deploy index_documents --runtime python310 --trigger-http --allow-unauthenticated --entry-point index_documents --source . --env-vars-file .env.yaml --memory 1024MB --region asia-southeast1`
`

### Environment Variables
`.env.yaml` in your main directory (remember to .gitignore it)
```
OPENAI_API_KEY: your openai key  
FIREBASE_STORAGE_BUCKET_URL: storage bucket  
ACCESS_KEY: your access key
```
