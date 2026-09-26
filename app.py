from dotenv import load_dotenv
load_dotenv();
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
import streamlit as st
from time import sleep

llm = ChatGroq(model=os.getenv("MODEL"))

if "vector_db" not in st.session_state:
    st.session_state.vertor_db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

def document_process(path):
    ## Document loading
    loader = PyPDFLoader("./Manjari Pandey 20062026.pdf")
    docs = loader.load()

    ## slpiting
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
    docs = splitter.split_documents(docs)

    ## embadings and vector stores
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )

    st.session_state.vector_db = vector_db
    st.session_state.document_uploaded = True

# ## user query
# query = "What is the name of patient? and name of doctors."
# documents = vector_db.similarity_search(query=query, k=2)
# ##print(len(documents), documents[0].page_content)

# context = ""
# for doc in documents:
#     context = context + doc.page_content + "\n\n"


# prompt = f"""You are a helpfull assistence and provide answer based on the provided context. context:{context}, question: {query}"""

#llm = ChatGroq(model=os.getenv("MODEL"))
# answer = llm.invoke(prompt)

st.subheader("📋 Document Q&A ChatBot = Ask Anything")

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

#### Document upload
if not st.session_state.document_uploaded:
    file = st.file_uploader(label="Select your PDF file", type="pdf")
    if file:
        with open("uploaded_document.pdf", "wb") as f:
            f.write(file.getvalue())
        
        with st.spinner("Procecessing...."):
            document_process("./uploaded_document.pdf")

        st.markdown("Document uploaded successfull")
        sleep(2)
        st.rerun()


if st.session_state.document_uploaded and st.session_state.vector_db :
    for oneMessage in st.session_state.messages:
        role = oneMessage["role"]
        content = oneMessage["content"]

        st.chat_message(role).markdown(content)
    
    query = st.chat_input("Ask Anuthing...")
    if query:

        st.session_state.messages.append({"role": "user", "content" : query})
        st.chat_message("user").markdown(query)

        documants = st.session_state.vector_db.similarity_search(query)
        context = ""

        for doc in documants:
            context += doc.page_content + "\n\n"

        prompt = f"""You are a helpfull assistence and provide answer based on the provided context. context:{context}, question: {query}"""
        result = llm.invoke(prompt)

        st.session_state.messages.append({"role": "AI", "content" : result.content})
        st.chat_message("ai").markdown(result.content)

#### Chat UI
