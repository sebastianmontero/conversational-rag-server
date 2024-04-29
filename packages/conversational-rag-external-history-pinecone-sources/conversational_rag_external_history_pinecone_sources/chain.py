from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessageChunk
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableMap,
    ConfigurableField,
    RunnableGenerator,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_cohere import CohereRerank
from typing import List, Tuple, AsyncIterator
from langserve.pydantic_v1 import BaseModel, Field
from langchain_pinecone import PineconeVectorStore
from operator import itemgetter
from langchain_core.runnables.utils import AddableDict
from langchain_core.runnables.configurable import RunnableConfigurableAlternatives
from langchain_groq import ChatGroq
import os


if os.environ.get("OPENAI_API_KEY", None) is None:
    raise Exception("Missing `OPENAI_API_KEY` environment variable.")

if os.environ.get("COHERE_API_KEY", None) is None:
    raise Exception("Missing `COHERE_API_KEY` environment variable.")

if os.environ.get("PINECONE_API_KEY", None) is None:
    raise Exception("Missing `PINECONE_API_KEY` environment variable.")

# if os.environ.get("PINECONE_ENVIRONMENT", None) is None:
#     raise Exception("Missing `PINECONE_ENVIRONMENT` environment variable.")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX", None)
if PINECONE_INDEX_NAME is None:
    raise Exception("Missing `PINECONE_INDEX` environment variable.")


def get_model(model_configurable_id: str) -> RunnableConfigurableAlternatives:
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0).configurable_alternatives(
        ConfigurableField(id=model_configurable_id),
        default_key="gpt35",
        gpt4=ChatOpenAI(model="gpt-4", temperature=0),
        llama38b=ChatGroq(model="llama3-8b-8192", temperature=0),
        llama370b=ChatGroq(model="llama3-70b-8192", temperature=0),
        mixtral7b=ChatGroq(model="mixtral-8x7b-32768", temperature=0),
        gemma7b=ChatGroq(model="gemma-7b-it", temperature=0),
    )


contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is. Just return the question nothing else.
    Do not add any information that is not in the chat history."""
# contextualize_q_system_prompt ="""Given the following conversation and a follow up question, rephrase the
# follow up question to be a standalone question."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

contextualize_model = get_model("contextualize-model")

# contextualize_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0).configurable_fields(
#     model_name=ConfigurableField(
#         id="contextualize-model",
#         name="contextualize model to use",
#         description="The model to use for contextualization. Options: gpt-3.5-turbo, gpt-4",
#     )
# )

contextualize_chain = contextualize_q_prompt | contextualize_model | StrOutputParser()

vectorstore = PineconeVectorStore.from_existing_index(
    PINECONE_INDEX_NAME, OpenAIEmbeddings(), text_key="text", namespace="main"
)

# {"filter":{"title":"Avatar"}}
retriever = vectorstore.as_retriever().configurable_fields(
    search_kwargs=ConfigurableField(
        id="search-parameters",
        name="search parameters",
        description="Set additional search parameters for the retriever",
    )
)
print(type(retriever))
# compression_retriever = ContextualCompressionRetriever(base_retriever=retriever, base_compressor=CohereRerank())
# template = """Answer the question based only on the following context:
template = """Provide a coherent and easy to understand answer to the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
parser = StrOutputParser()

# model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
model = get_model("model")


# User input
class ChatHistory(BaseModel):
    """Chat history with the bot."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )
    input: str


answer_chain = prompt | model | parser


async def extract_sources(
    chunks: AsyncIterator[AddableDict],
) -> AsyncIterator[AddableDict]:
    async for chunk in chunks:
        parsed = {}
        if "context" in chunk:
            sources = []
            for doc in chunk["context"]:
                sources.append(doc.metadata["source"])
            parsed["sources"] = sources
        if "answer" in chunk:
            parsed["answer"] = chunk["answer"]
        if len(parsed) > 0:
            yield parsed


chain = (
    contextualize_chain
    | RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=answer_chain)
    | RunnableGenerator(extract_sources)
)

# chain = contextualize_chain | {"context": retriever, "question": RunnablePassthrough()} | prompt | model | parser
# chain
