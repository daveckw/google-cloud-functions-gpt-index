import firebase_admin
from firebase_admin import firestore
import os

storage_url = os.getenv("FIREBASE_STORAGE_BUCKET_URL")

firebase_admin.initialize_app(options={"storageBucket": storage_url})
db = firestore.client()
