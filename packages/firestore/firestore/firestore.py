from google.cloud import firestore
import os

def create_firestore_client():
    credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", None)
    print("GOOGLE_APPLICATION_CREDENTIALS:", credentials)
    # Create a Firestore client
    firestore_client = firestore.AsyncClient()
    return firestore_client