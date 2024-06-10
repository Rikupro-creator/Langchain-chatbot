import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

os.environ["OPENAI_API_KEY"] = "sk-proj-wTEuxQogytTYz1e3Yl6wT3BlbkFJqTliRDaIB43DS5PoPTpo"

def qa(query):
    # Load document
    loader = PyPDFLoader("C:\\Users\\hp\\Downloads\\ken127322_merged.pdf")
    documents = loader.load()
    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # Select which embeddings to use
    embeddings = OpenAIEmbeddings()
    # Create the vectorstore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    # Expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    # Create a chain to answer questions
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type='map_reduce', retriever=retriever, return_source_documents=True)
    result = qa({"query": query})
    return result

st.title("Wakili.AI MVP")

# Input field for query
query = st.text_input("Enter your query:")

if st.button("Search"):
    if query:
        try:
            result = qa(query)
            st.write(result['result'])
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a query.")
