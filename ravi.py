import streamlit as st
import fitz
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import FastEmbedEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
import os

# Function to extract text from a PDF file
def load_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text_data = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text_data.append(page.get_text("text"))

    return text_data

# Function to fetch and extract text from a URL
def fetch_url_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Check if the URL points to a PDF
            if url.lower().endswith(".pdf"):
                with open("temp.pdf", "wb") as f:
                    f.write(response.content)
                return load_pdf_text("temp.pdf")
            else:
                # Extract text from HTML
                soup = BeautifulSoup(response.text, "html.parser")
                return [soup.get_text()]
        else:
            st.error(f"Failed to fetch URL: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return []

# Streamlit UI setup
st.set_page_config(page_title="PDF/URL Q&A Bot", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #008000;
        color: #333333;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 PDF/URL Q&A Bot with Text Retrieval")

# Option to upload a PDF or submit a URL
option = st.radio("Choose an option:", ("Upload PDF", "Submit URL"))

if option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

    if uploaded_file is not None:
        with st.spinner("Processing your PDF..."):
            with open("uploaded_file.pdf", "wb") as f:
                f.write(uploaded_file.read())

            # Load text from the PDF
            texts = load_pdf_text("uploaded_file.pdf")

            # Prepare text chunks for embedding
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            text_chunks = text_splitter.create_documents(texts)
            text_embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
            text_db = FAISS.from_documents(text_chunks, text_embeddings)

            # Setup language model for Q&A
            os.environ['GROQ_API_KEY'] = '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'
            llm = ChatGroq(model_name='llama-3.1-8b-instant')
            memory = ConversationBufferMemory(memory_key='chat_history', return_messages=False)
            retriever = text_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            qa = ConversationalRetrievalChain.from_llm(
                llm=llm,
                memory=memory,
                retriever=retriever
            )

            st.success("PDF processed successfully!")

            # Sidebar for extracted text
            st.sidebar.subheader("📜 Extracted Text")
            for i, text in enumerate(texts):
                with st.sidebar.expander(f"Page {i + 1}"):
                    st.write(text)

            # Tabs for asking questions and exploring text
            tab1, tab2 = st.tabs(["💬 Ask Questions", "📜 Explore Text"])

            with tab1:
                query = st.text_input("Ask a question about the PDF:")
                if query:
                    with st.spinner("Searching for answers..."):
                        result = qa({"question": query})
                        answer = result['answer']
                        st.write("### 💡 Answer:", answer)

            with tab2:
                st.subheader("📜 Extracted Text")
                for i, text in enumerate(texts):
                    with st.expander(f"Page {i + 1}"):
                        st.write(text)

    else:
        st.info("Upload a PDF file to get started!")

else:  # Submit URL
    url = st.text_input("Enter the URL:")

    if url:
        with st.spinner("Fetching and processing URL content..."):
            # Fetch and extract text from the URL
            texts = fetch_url_content(url)

            if texts:
                # Prepare text chunks for embedding
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                text_chunks = text_splitter.create_documents(texts)
                text_embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
                text_db = FAISS.from_documents(text_chunks, text_embeddings)

                # Setup language model for Q&A
                os.environ['GROQ_API_KEY'] = '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'
                llm = ChatGroq(model_name='llama-3.1-8b-instant')
                memory = ConversationBufferMemory(memory_key='chat_history', return_messages=False)
                retriever = text_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                qa = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    memory=memory,
                    retriever=retriever
                )

                st.success("URL content processed successfully!")

                # Sidebar for extracted text
                st.sidebar.subheader("📜 Extracted Text")
                for i, text in enumerate(texts):
                    with st.sidebar.expander(f"Section {i + 1}"):
                        st.write(text)

                # Tabs for asking questions and exploring text
                tab1, tab2 = st.tabs(["💬 Ask Questions", "📜 Explore Text"])

                with tab1:
                    query = st.text_input("Ask a question about the URL content:")
                    if query:
                        with st.spinner("Searching for answers..."):
                            result = qa({"question": query})
                            answer = result['answer']
                            st.write("### 💡 Answer:", answer)

                with tab2:
                    st.subheader("📜 Extracted Text")
                    for i, text in enumerate(texts):
                        with st.expander(f"Section {i + 1}"):
                            st.write(text)
            else:
                st.error("No text could be extracted from the URL.")