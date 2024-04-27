from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from langserve import add_routes
from conversational_rag.chain import chain as conversational_rag_chain
from conversational_rag_external_history.chain import chain as conversational_rag_chain_external_history
from conversational_rag_external_history_pinecone.chain import chain as conversational_rag_chain_external_history_pinecone
from conversational_rag_external_history_pinecone_sources.chain import chain as conversational_rag_chain_external_history_pinecone_sources
from conversational_agent_external_history_pinecone.chain import chain as conversational_agent_chain_external_history_pinecone
from fastapi.middleware.cors import CORSMiddleware
from firestore.models import ChatConfig
from firestore.firestore import create_firestore_client

app = FastAPI()
firestore_client = create_firestore_client()


# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


@app.get("/chat-config/{id}")
async def get_chat_config(id: str):
    # Create an instance of ChatConfig
    chat_config = ChatConfig(firestore_client)

    # Get the chat config document by ID
    document = chat_config.get_document_by_id(id)

    if document:
        return document
    else:
        raise HTTPException(status_code=404, detail="Chat config not found")

# Edit this to add the chain you want to add
add_routes(app, conversational_rag_chain, path="/conversational-rag", playground_type="default")

add_routes(app, conversational_rag_chain_external_history, path="/conversational-rag-eh", playground_type="chat")

add_routes(app, conversational_rag_chain_external_history_pinecone, path="/conversational-rag-ehp", playground_type="default")

add_routes(app, conversational_rag_chain_external_history_pinecone_sources, path="/conversational-rag-ehps", playground_type="default")

add_routes(app, conversational_agent_chain_external_history_pinecone, path="/conversational-agent-ehp", playground_type="default")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
