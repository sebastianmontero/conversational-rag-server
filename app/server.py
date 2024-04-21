from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from conversational_rag.chain import chain as conversational_rag_chain
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


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


# Edit this to add the chain you want to add
add_routes(app, conversational_rag_chain, path="/conversational-rag", playground_type="default")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
