class FirestoreBaseClass:
    def __init__(self, firestore_client, collection_name):
        self.firestore_client = firestore_client
        self.collection_name = collection_name

    async def get_document_by_id(self, document_id):
        document_ref = self.firestore_client.collection(self.collection_name).document(document_id)
        document = await document_ref.get()
        return document.to_dict() if document.exists else None
    
    async def get_all_documents(self):
        documents = []
        async for document in self.firestore_client.collection(self.collection_name).stream():
            document_dict = document.to_dict()
            document_dict['id'] = document.id
            documents.append(document_dict)
        return documents
    


class ChatConfig(FirestoreBaseClass):
    def __init__(self, firestore_client):
        super().__init__(firestore_client, "chat-config")


class Model(FirestoreBaseClass):
    def __init__(self, firestore_client):
        super().__init__(firestore_client, "model")