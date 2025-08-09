import asyncio
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader, UnstructuredCSVLoader,
    UnstructuredHTMLLoader, UnstructuredExcelLoader
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import (
    ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Asyncio event loop fix for deployment
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")


def get_file_extension(filepath):
    return os.path.splitext(filepath)[1].lower()

def load_document(filepath):
    ext = get_file_extension(filepath)

    if ext == ".pdf":
        loader = PyPDFLoader(filepath)
    elif ext == ".txt":
        loader = TextLoader(filepath)
    elif ext in [".doc", ".docx"]:
        loader = UnstructuredWordDocumentLoader(filepath)
    elif ext in [".ppt", ".pptx"]:
        loader = UnstructuredPowerPointLoader(filepath)
    elif ext == ".csv":
        loader = UnstructuredCSVLoader(filepath)
    elif ext == ".html":
        loader = UnstructuredHTMLLoader(filepath)
    elif ext in [".xls", ".xlsx"]:
        loader = UnstructuredExcelLoader(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return loader.load()


def main():
    st.set_page_config(page_title="📄 Document Chatbot", layout="centered")
    st.title("📄 Document Chatbot using LangChain + Gemini")

    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "main_chain" not in st.session_state:
        st.session_state.main_chain = None

    
    uploaded_file = st.file_uploader(
        "Upload a document to start chatting",
        type=["pdf", "txt", "docx", "pptx", "csv", "html", "xls", "xlsx"]
    )

    if uploaded_file and st.session_state.main_chain is None:
        file_ext = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_filepath = tmp_file.name

        st.success(f"✅ File '{uploaded_file.name}' uploaded successfully.")

        try:
            docs = load_document(tmp_filepath)
        except ValueError as ve:
            st.error(str(ve))
            return

        
        full_text = "\n".join(doc.page_content for doc in docs)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_text(full_text)
        documents = [Document(page_content=chunk) for chunk in chunks]

       
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        vectorstore = FAISS.from_documents(documents, embedding_model)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

       
        prompt = PromptTemplate(
            template="""
            You are a helpful assistant.
            Answer ONLY from the provided document context. If the context is insufficient, provide a logical answer related to the document.

            {context}
            Question: {question}
            """,
            input_variables=["context", "question"]
        )

        def format_docs(retrieved_docs):
            return "\n\n".join(doc.page_content for doc in retrieved_docs)

        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })

        output_parser = StrOutputParser()

        main_chain = parallel_chain | prompt | ChatGoogleGenerativeAI(
            model='gemini-2.0-flash',
            google_api_key=google_api_key
        ) | output_parser

        st.session_state.main_chain = main_chain

    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    
    if prompt := st.chat_input("Ask something about the document..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.main_chain:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = st.session_state.main_chain.invoke(prompt)
                    st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            with st.chat_message("assistant"):
                st.markdown("Please upload a document first.")

if __name__ == "__main__":
    main()
