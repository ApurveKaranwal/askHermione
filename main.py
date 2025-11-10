import streamlit as st
import os
import time
import tempfile
import io
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from collections import Counter
import re

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ------------------ Streamlit UI Setup ------------------ #
st.set_page_config(page_title="askHermione", layout="wide", page_icon="🎓")
st.title("🎓 askHermione")
st.caption("AI Study Assistant – Ask, Summarize, and Understand Your Notes Instantly 📚")

# ------------------ Session Variables ------------------ #
if "processed_notes" not in st.session_state:
    st.session_state.processed_notes = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ------------------ Cache Models ------------------ #
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource
def get_llm():
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant"
    )

# ------------------ Document Processor ------------------ #
def process_notes(files, subject):
    with st.spinner(f"Processing notes for {subject}..."):
        try:
            embeddings = get_embeddings()
            all_docs = []

            for file in files:
                ext = file.name.lower().split(".")[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as temp_file:
                    temp_file.write(file.read())
                    temp_path = temp_file.name

                if ext == "pdf":
                    loader = PyPDFLoader(temp_path)
                elif ext == "txt":
                    loader = TextLoader(temp_path)
                elif ext in ["docx", "doc"]:
                    loader = UnstructuredWordDocumentLoader(temp_path)
                else:
                    st.warning(f"Unsupported file type: {file.name}")
                    continue

                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file.name
                    doc.metadata["subject"] = subject
                all_docs.extend(docs)
                os.unlink(temp_path)

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = splitter.split_documents(all_docs)
            vector_store = FAISS.from_documents(splits, embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            return vector_store, retriever
        except Exception as e:
            st.error(f"Error processing notes: {str(e)}")
            return None, None

# ------------------ Helper Functions ------------------ #
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def extract_keywords(text, num_keywords=10):
    words = re.findall(r"\b[A-Za-z]{4,}\b", text.lower())
    common = Counter(words).most_common(num_keywords)
    return [w for w, _ in common]

# ------------------ Chains ------------------ #
def create_rag_chain(retriever):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful and intelligent academic assistant.
    Subject: {subject}

    Context: {context}

    Question: {question}

    Based on the provided notes, give a clear, educational, and concise explanation.
    If the information isn't available, say "I couldn’t find that information in your notes."
    """)

    # FIX: Using explicit lambda functions for input mapping to prevent 'dict' object error.
    rag_chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "question": lambda x: x["question"],
            "subject": lambda x: x["subject"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever

def summarize_documents(retriever, subject):
    llm = get_llm()
    docs = retriever.invoke("summary of all concepts")
    full_text = format_docs(docs)
    prompt = f"Summarize the key ideas from these {subject} notes:\n\n{full_text[:3000]}"
    return llm.invoke(prompt).content

# ------------------ Sidebar (Modified) ------------------ #
with st.sidebar:
    st.header("⚙️ App Control Panel")
    
    # 1. Upload Section
    with st.expander("📂 Upload & Process Notes", expanded=True):
        subject = st.text_input("1. Enter Subject Name (e.g., Physics)")
        uploaded_files = st.file_uploader(
            "2. Upload Notes (PDF, DOCX, TXT)", accept_multiple_files=True, type=["pdf", "txt", "docx"]
        )

        if uploaded_files and subject:
            if st.button("🚀 Start Processing Notes", use_container_width=True):
                vector_store, retriever = process_notes(uploaded_files, subject)
                if vector_store:
                    st.session_state.vector_store = vector_store
                    st.session_state.retriever = retriever
                    st.session_state.processed_notes = True
                    st.success(f"✅ Processed {len(uploaded_files)} file(s) for {subject}!")
        elif not st.session_state.processed_notes:
             st.info("Fill both fields and click 'Process Notes' to start.")

    # 2. Management Section
    with st.expander("🧹 Chat & Data Management"):
        if st.session_state.chat_history:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Chat History", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
            with col2:
                # Download button logic moved here
                chat_data = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])
                st.download_button("⬇️ Download Chat", data=chat_data,
                                file_name="course_notes_chat.txt", use_container_width=True)
        else:
            st.info("Chat history is empty.")

    st.markdown("---")
    st.info("Status: " + ("✅ Notes Ready" if st.session_state.processed_notes else "❌ Awaiting Notes"))

# ------------------ Main UI (Tabs Added) ------------------ #
tab1, tab2 = st.tabs(["💬 Q&A Chat", "🧠 Tools & Concepts"])

with tab1:
    st.markdown("### Ask Questions About Your Notes")
    
    # Display Chat History
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

    # Chat Input
    if user_input := st.chat_input("Ask a question from your notes..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        if not st.session_state.processed_notes:
            response = "Please upload and process your notes first from the sidebar."
            st.chat_message("assistant").write(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                try:
                    current_subject = st.session_state.get("subject") or subject 
                    
                    rag_chain, retriever = create_rag_chain(st.session_state.retriever)
                    
                    start = time.time()
                    response_text = rag_chain.invoke({"question": user_input, "subject": current_subject})
                    elapsed = time.time() - start
                    
                    placeholder.write(response_text)
                    st.caption(f"Response time: {elapsed:.2f} sec")

                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})

                    # Display relevant docs for transparency
                    relevant_docs = retriever.invoke(user_input)
                    with st.expander("View related note sections"):
                        for i, doc in enumerate(relevant_docs):
                            st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                            st.markdown(f"{doc.page_content[:300]}...")
                            st.markdown("---")
                except Exception as e:
                    err = f"⚠️ Error: {str(e)}"
                    placeholder.error(err)
                    st.session_state.chat_history.append({"role": "assistant", "content": err})

    if not st.session_state.processed_notes and not st.session_state.chat_history:
        st.info("👈 Use the sidebar to upload your course notes to begin chatting.")

with tab2:
    st.markdown("### Study Tools")
    if not st.session_state.processed_notes:
        st.warning("Please upload and process notes in the sidebar to use these tools.")
    else:
        st.markdown("Use these tools to generate insights about all your processed study material.")
        st.markdown("---")

        # Summarize Tool
        col_summary, col_keywords = st.columns(2)
        
        with col_summary:
            st.subheader("📄 Generate Full Summary")
            st.markdown("Get a concise overview of all uploaded documents.")
            summary_placeholder = st.empty()
            if st.button("Summarize Notes", key="btn_summary", use_container_width=True):
                with summary_placeholder.container():
                    with st.spinner("Generating summary..."):
                        summary = summarize_documents(st.session_state.retriever, subject)
                        st.session_state.chat_history.append({"role": "assistant", "content": f"📘 **Summary:** {summary}"})
                        st.success("Summary generated and added to chat!")
        
        # Keyword Tool
        with col_keywords:
            st.subheader("🔑 Extract Key Concepts")
            st.markdown("Identify the top concepts and terms across all documents.")
            keywords_placeholder = st.empty()
            if st.button("Extract Keywords", key="btn_keywords", use_container_width=True):
                with keywords_placeholder.container():
                    with st.spinner("Extracting keywords..."):
                        docs = st.session_state.retriever.invoke("all content for keyword extraction")
                        text = format_docs(docs)
                        keywords = extract_keywords(text)
                        st.session_state.chat_history.append({"role": "assistant", "content": f"🔑 **Key Concepts:** {', '.join(keywords)}"})
                        st.success("Keywords extracted and added to chat!")