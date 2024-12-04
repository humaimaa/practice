import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
import shutil
load_dotenv()


def load_chunks(chunks:list):
    embeddings=OpenAIEmbeddings()
    #initializing our  vectore store
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_store",
        collection_name="test_collection"
        )
    
    print("adding documents to chroma")
    Chroma.add_documents(vectorstore, chunks)
    print("documnets added to chroma")


def retriver(question):
    embeddings=OpenAIEmbeddings()
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_store",
        collection_name="test_collection"
        )
    docs = vectorstore.similarity_search(question,k=3)
    return docs
 