import firebase_admin
from firebase_admin import firestore
from firebase_admin import storage
import os

storage_url = os.getenv("FIREBASE_STORAGE_BUCKET_URL")

firebase_admin.initialize_app(options={"storageBucket": storage_url})
db = firestore.client()
bucket = storage.bucket(name=storage_url)
