import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")


def load_and_split_pdf(file_path):
    """Load PDF fle & split it into chunks"""
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)


def setup_vectorstore(documents):
    """Init pinecone vectorstore and add documents"""
    if not documents:
        raise ValueError("No documents provided by user")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    vectorstore = PineconeVectorStore(index, embedding=embeddings, text_key="content")
    vectorstore.add_documents(documents)
    return vectorstore


def build_chain(retriever):
    """Build retrieval QA chain"""
    prompt = ChatPromptTemplate.from_template(
        """You are an expert PDF assistant.
Given the context from the PDF and a user question, generate helpful answer.

Context:
{context}

Question:
{input}

Answer:"""
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    output_parser = StrOutputParser()

    chain = (
        RunnableMap({
            "input": lambda x: x,
            "context": lambda x: "\n\n".join(
    str(doc.page_content) for doc in retriever.invoke(x) if doc.page_content)})
        | prompt
        | llm
        | output_parser
    )
    return chain
