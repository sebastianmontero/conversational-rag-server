from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableMap, ConfigurableField, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_cohere import CohereRerank
from typing import List, Tuple
from langserve.pydantic_v1 import BaseModel, Field
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.callbacks import FinalStreamingStdOutCallbackHandler
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


# contextualize_q_system_prompt ="""Given the following conversation and a follow up question, rephrase the 
# follow up question to be a standalone question."""
# contextualize_q_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", contextualize_q_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )

# contextualize_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# contextualize_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0).configurable_fields(
#     model_name=ConfigurableField(
#         id="contextualize-model",
#         name="contextualize model to use",
#         description="The model to use for contextualization. Options: gpt-3.5-turbo, gpt-4",
#     )
# )

# contextualize_chain = contextualize_q_prompt | contextualize_model | StrOutputParser()


films_vectorstore = PineconeVectorStore.from_existing_index(
    PINECONE_INDEX_NAME, OpenAIEmbeddings(), text_key="summary"
)

# {"filter":{"title":"Avatar"}}
films_retriever = films_vectorstore.as_retriever()

films_tool =create_retriever_tool(films_retriever, "film_search","Find information about films")

investment_vectorstore = PineconeVectorStore.from_existing_index(
    PINECONE_INDEX_NAME, OpenAIEmbeddings(), text_key="text", namespace="lyn-alden"
)

# {"filter":{"title":"Avatar"}}
investment_retriever = investment_vectorstore.as_retriever()

investment_tool =create_retriever_tool(investment_retriever, "investment_search","Find information about investing, the economy and bitcoin")

tools = [films_tool, investment_tool]

prompt = hub.pull("hwchase17/openai-functions-agent")

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True, callbacks=[FinalStreamingStdOutCallbackHandler()])

agent = create_tool_calling_agent(model, tools, prompt)

class ChatHistory(BaseModel):
    """Chat history with the bot."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "input", "output": "output"}},
    )
    input: str

chain = AgentExecutor(agent=agent, tools=tools, verbose=True).with_types(input_type=ChatHistory)
