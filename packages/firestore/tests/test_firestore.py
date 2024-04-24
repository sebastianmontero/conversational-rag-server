import unittest
from firestore.firestore import create_firestore_client
from firestore.models import ChatConfig



class TestFirestore(unittest.TestCase):
    def test_document_by_id(self):
        client = create_firestore_client()
        chatConfig = ChatConfig(client)

        doc = chatConfig.get_document_by_id("mEAXwvCUcUPOp22R99u1")
        print("doc:", doc)

if __name__ == '__main__':
    unittest.main()