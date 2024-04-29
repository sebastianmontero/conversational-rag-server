from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from langserve import add_routes
from conversational_rag.chain import chain as conversational_rag_chain
from conversational_rag_external_history.chain import (
    chain as conversational_rag_chain_external_history,
)
from conversational_rag_external_history_pinecone.chain import (
    chain as conversational_rag_chain_external_history_pinecone,
)
from conversational_rag_external_history_pinecone_sources.chain import (
    chain as conversational_rag_chain_external_history_pinecone_sources,
)
from conversational_agent_external_history_pinecone.chain import (
    chain as conversational_agent_chain_external_history_pinecone,
)
from fastapi.middleware.cors import CORSMiddleware
from firestore.models import ChatConfig, Model
from firestore.firestore import create_firestore_client
from typing import Dict, Any
from starlette.requests import Request

app = FastAPI()
firestore_client = create_firestore_client()
chat_config = ChatConfig(firestore_client)
model = Model(firestore_client)

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
    # Get the chat config document by ID
    document = await chat_config.get_document_by_id(id)

    if document:
        return document
    else:
        raise HTTPException(status_code=404, detail="Chat config not found")


@app.get("/models")
async def get_models():
    models = await model.get_all_documents()
    return models


async def config_modifier(config: Dict[str, Any], req: Request) -> Dict[str, Any]:
    print(config)
    body = await req.json()
    print(body)
    if "id" not in body:
        raise HTTPException(status_code=400, detail="Missing 'id' parameter")
    document = await chat_config.get_document_by_id(body["id"])
    if document is None:
        raise HTTPException(status_code=400, detail="Invalid 'id' parameter")
    # set config configurable->search-parameters->filter->namespace" to document["namespace"] use setdefault
    config.setdefault("configurable", {}).setdefault(
        "search-parameters", {}
    ).setdefault("filter", {})["namespace"] = document["namespace"]
    print("config:", config)
    return config


# Edit this to add the chain you want to add
add_routes(
    app, conversational_rag_chain, path="/conversational-rag", playground_type="default"
)

add_routes(
    app,
    conversational_rag_chain_external_history,
    path="/conversational-rag-eh",
    playground_type="chat",
)

add_routes(
    app,
    conversational_rag_chain_external_history_pinecone,
    path="/conversational-rag-ehp",
    playground_type="default",
)

add_routes(
    app,
    conversational_rag_chain_external_history_pinecone_sources,
    path="/conversational-rag-ehps",
    playground_type="default",
    per_req_config_modifier=config_modifier,
)

add_routes(
    app,
    conversational_agent_chain_external_history_pinecone,
    path="/conversational-agent-ehp",
    playground_type="default",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
