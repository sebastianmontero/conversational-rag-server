from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableMap, ConfigurableField
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_cohere import CohereRerank
import os


if os.environ.get("OPENAI_API_KEY", None) is None:
    raise Exception("Missing `OPENAI_API_KEY` environment variable.")

if os.environ.get("COHERE_API_KEY", None) is None:
    raise Exception("Missing `COHERE_API_KEY` environment variable.")

# if os.environ.get("PINECONE_API_KEY", None) is None:
#     raise Exception("Missing `PINECONE_API_KEY` environment variable.")

# if os.environ.get("PINECONE_ENVIRONMENT", None) is None:
#     raise Exception("Missing `PINECONE_ENVIRONMENT` environment variable.")

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    print("history:", store[session_id])
    return store[session_id]


contextualize_q_system_prompt ="""Given the following conversation and a follow up question, rephrase the 
follow up question to be a standalone question, in its original language."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
contextualize_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0).configurable_fields(
    model_name=ConfigurableField(
        id="contextualize-model",
        name="contextualize model to use",
        description="The model to use for contextualization. Options: gpt-3.5-turbo, gpt-4",
    )
)

contextualize_chain = contextualize_q_prompt | contextualize_model | StrOutputParser()

vectorstore = DocArrayInMemorySearch.from_texts(["harrison worked at kensho", "harrison is 30 years old", "harrison lives in new york", "sofia lives in brazil"], embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
compression_retriever = ContextualCompressionRetriever(base_retriever=retriever, base_compressor=CohereRerank())
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
parser = StrOutputParser()

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

base_chain = contextualize_chain | {"context": compression_retriever, "question": RunnablePassthrough()} | prompt | model | parser
chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)