class FirestoreBaseClass:
    def __init__(self, firestore_client, collection_name):
        self.firestore_client = firestore_client
        self.collection_name = collection_name

    def get_document_by_id(self, document_id):
        collection_ref = self.firestore_client.collection(self.collection_name)
        document_ref = collection_ref.document(document_id)
        document = document_ref.get()
        return document.to_dict() if document.exists else None
    


class ChatConfig(FirestoreBaseClass):
    def __init__(self, firestore_client):
        super().__init__(firestore_client, "chat-config")