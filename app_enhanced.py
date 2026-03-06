"""
Quest Analytics - Enhanced RAG Storytelling Assistant
=====================================================
An upgraded web UI with conversational memory, PDF management,
chat history persistence, and improved UX.

Run with:  streamlit run app_enhanced.py
"""

import os
import sys
import subprocess
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Ensure pypdf is available (Streamlit Cloud can silently drop packages)
try:
    import pypdf  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])

import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment & paths
# ---------------------------------------------------------------------------
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDFS_DIR = os.path.join(BASE_DIR, "pdfs")
UPLOADED_PDFS_DIR = os.path.join(BASE_DIR, "uploaded_pdfs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")

for d in (PDFS_DIR, UPLOADED_PDFS_DIR, CHROMA_DIR):
    os.makedirs(d, exist_ok=True)

# Support both .env (local) and st.secrets (Streamlit Cloud)
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
if not HF_API_KEY:
    HF_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", "")
    if HF_API_KEY:
        os.environ["HUGGINGFACE_API_KEY"] = HF_API_KEY

COLLECTION_NAME = "quest_analytics_papers"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Quest Analytics RAG Assistant (Enhanced)",
    page_icon="\U0001f52e",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%);
    }
    .main-header {
        background: linear-gradient(90deg, #e94560 0%, #c23152 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; font-weight: 700; }
    .main-header p { color: rgba(255,255,255,.85); margin: .3rem 0 0 0; font-size: 1rem; }
    .source-badge {
        display: inline-block;
        background: rgba(233,69,96,.15);
        color: #e94560;
        border: 1px solid rgba(233,69,96,.3);
        border-radius: 6px;
        padding: 2px 8px;
        margin: 2px 4px;
        font-size: .82rem;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16213e 0%, #1a1a2e 100%);
    }
    footer { visibility: hidden; }
    .chat-user {
        background: rgba(233,69,96,.12);
        border-left: 4px solid #e94560;
        border-radius: 8px;
        padding: .8rem 1rem;
        margin: .5rem 0;
    }
    .chat-bot {
        background: rgba(255,255,255,.05);
        border-left: 4px solid #4ecdc4;
        border-radius: 8px;
        padding: .8rem 1rem;
        margin: .5rem 0;
    }
    .stat-card {
        background: rgba(255,255,255,.06);
        border: 1px solid rgba(255,255,255,.1);
        border-radius: 10px;
        padding: .8rem;
        text-align: center;
    }
    .doc-row {
        background: rgba(255,255,255,.04);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 8px;
        padding: .6rem .8rem;
        margin: .3rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===================================================================
# Chat history persistence
# ===================================================================

def save_chat_history(history: list):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2, default=str)
    except Exception:
        pass


def load_chat_history() -> list:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []


# ===================================================================
# Heavy-lifting functions (cached)
# ===================================================================

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    from langchain_core.embeddings import Embeddings

    chroma_ef = DefaultEmbeddingFunction()

    class OnnxEmbeddings(Embeddings):
        def __init__(self, ef):
            self._ef = ef

        def embed_documents(self, texts):
            return [emb.tolist() for emb in self._ef(texts)]

        def embed_query(self, text):
            return self._ef([text])[0].tolist()

    return OnnxEmbeddings(chroma_ef)


@st.cache_resource(show_spinner=False)
def get_llm():
    from langchain_community.llms import HuggingFaceHub

    model_name = os.getenv("HF_MODEL_NAME", "google/flan-t5-base")
    api_key = os.getenv("HUGGINGFACE_API_KEY", "")

    if api_key:
        try:
            llm = HuggingFaceHub(
                repo_id=model_name,
                model_kwargs={"temperature": 0.7, "max_length": 512},
                huggingfacehub_api_token=api_key,
            )
            return llm, model_name
        except Exception:
            pass

    from langchain_core.language_models.llms import LLM

    class ExtractiveSummaryLLM(LLM):
        @property
        def _llm_type(self) -> str:
            return "extractive-summary-llm"

        def _call(self, prompt: str, stop: Optional[list[str]] = None,
                  run_manager: Any = None, **kwargs) -> str:
            question, context = "", ""
            if "Question:" in prompt and "Context:" in prompt:
                parts = prompt.split("Question:")
                context_part = parts[0]
                question = parts[1].split("Answer:")[0].strip() if len(parts) > 1 else ""
                if "Context:" in context_part:
                    context = context_part.split("Context:")[1].strip()
            elif "Chat History:" in prompt and "Follow Up Question:" in prompt:
                parts = prompt.split("Follow Up Question:")
                context = parts[0]
                question = parts[1].strip() if len(parts) > 1 else ""
            else:
                context = prompt

            stop_words = {
                "what", "is", "the", "a", "an", "are", "how", "should",
                "under", "for", "of", "and", "in", "to", "by", "with",
                "do", "does", "why", "it", "its", "that", "this", "be",
                "on", "at", "from", "or", "which", "was", "were", "has",
                "have", "been", "being", "can", "could", "will", "would",
            }
            question_keywords = set(question.lower().split()) - stop_words

            sentences = [s.strip() + "." for s in context.replace("\n", " ").split(".")
                         if len(s.strip()) > 20]
            if not sentences:
                return "No relevant information found in the provided context."

            scored = sorted(
                [(len(question_keywords & set(s.lower().split())), s) for s in sentences],
                key=lambda x: x[0],
                reverse=True,
            )
            top = [s for score, s in scored[:3] if score > 0]
            if not top:
                top = [scored[0][1]] if scored else ["No relevant information found."]
            answer = " ".join(top)
            return answer[:500] if len(answer) > 500 else answer

    return ExtractiveSummaryLLM(), "ExtractiveSummaryLLM (Fallback)"


# ===================================================================
# Vector DB helpers
# ===================================================================

def build_vectordb(documents, embedding_model):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )
    return vectordb, len(chunks)


def load_existing_vectordb(embedding_model):
    from langchain_community.vectorstores import Chroma

    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
    )
    count = vectordb._collection.count()
    if count == 0:
        return None, 0
    return vectordb, count


def get_vectordb_stats(embedding_model):
    """Return per-document chunk counts from the vector store."""
    from langchain_community.vectorstores import Chroma

    try:
        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME,
        )
        collection = vectordb._collection
        total = collection.count()
        if total == 0:
            return total, {}

        all_meta = collection.get(include=["metadatas"])
        doc_chunks: dict[str, int] = {}
        for meta in all_meta["metadatas"]:
            src = meta.get("source_file", meta.get("source", "Unknown"))
            if "/" in src or "\\" in src:
                src = Path(src).name
            doc_chunks[src] = doc_chunks.get(src, 0) + 1
        return total, doc_chunks
    except Exception:
        return 0, {}


def delete_document_chunks(embedding_model, source_filename: str) -> int:
    """Delete all chunks belonging to a specific source document."""
    from langchain_community.vectorstores import Chroma

    try:
        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME,
        )
        collection = vectordb._collection
        all_data = collection.get(include=["metadatas"])
        ids_to_delete = []
        for i, meta in enumerate(all_data["metadatas"]):
            src = meta.get("source_file", meta.get("source", ""))
            if "/" in src or "\\" in src:
                src = Path(src).name
            if src == source_filename:
                ids_to_delete.append(all_data["ids"][i])
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)
    except Exception:
        return 0


# ===================================================================
# PDF helpers
# ===================================================================

def list_all_pdfs():
    """Return all PDF paths from pdfs/, uploaded_pdfs/, and project root."""
    seen = {}
    for folder in (PDFS_DIR, UPLOADED_PDFS_DIR, BASE_DIR):
        for p in Path(folder).glob("*.pdf"):
            seen.setdefault(p.name, p)
    return list(seen.values())


def load_pdf_documents(pdf_paths):
    from langchain_community.document_loaders import PyPDFLoader

    all_docs = []
    meta = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = Path(pdf_path).name
        all_docs.extend(docs)
        meta.append({"filename": Path(pdf_path).name, "pages": len(docs)})
    return all_docs, meta


# ===================================================================
# Storytelling chains
# ===================================================================

def get_storytelling_chains(llm, retriever):
    from langchain_core.prompts import PromptTemplate
    from langchain_core.language_models.llms import LLM as BaseLLMClass

    prompts = {
        "kid": PromptTemplate(
            template=(
                "You are a friendly teacher explaining to an 8-12 year old kid.\n"
                "Use simple words, fun analogies, and make it feel like an adventure!\n"
                "Use examples kids understand (school, games, superheroes, toys).\n\n"
                "Context:\n{context}\n\nQuestion: {question}\n\n"
                "Fun and Simple Answer for Kids:"
            ),
            input_variables=["context", "question"],
        ),
        "adult": PromptTemplate(
            template=(
                "You are a practical advisor using real-life examples.\n"
                "Relate concepts to everyday things like Netflix accounts, online banking,\n"
                "shopping websites, and social media privacy settings.\n\n"
                "Context:\n{context}\n\nQuestion: {question}\n\n"
                "Practical Answer with Real-Life Examples:"
            ),
            input_variables=["context", "question"],
        ),
        "story": PromptTemplate(
            template=(
                "You are a storyteller explaining through narrative.\n"
                "Create a short story with characters, a setting, and a resolution.\n"
                "Use real historical context when possible.\n\n"
                "Context:\n{context}\n\nQuestion: {question}\n\n"
                "Story Answer:"
            ),
            input_variables=["context", "question"],
        ),
    }

    class StorytellingLLM(BaseLLMClass):
        base_llm: Any = None
        mode: str = "adult"

        @property
        def _llm_type(self) -> str:
            return f"storytelling-{self.mode}"

        def _call(self, prompt, stop=None, run_manager=None, **kwargs):
            raw = self.base_llm._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            if getattr(self.base_llm, "_llm_type", "") != "extractive-summary-llm":
                return raw
            return self._transform(raw)

        def _transform(self, text):
            if not text or text.startswith("No relevant"):
                return text
            if self.mode == "kid":
                return self._kid_transform(text)
            if self.mode == "story":
                return self._story_transform(text)
            return self._adult_transform(text)

        def _kid_transform(self, text):
            replacements = {
                "regulation": "set of rules", "Regulation": "Set of rules",
                "compliance": "following the rules", "Compliance": "Following the rules",
                "organizations": "companies", "organization": "company",
                "processing": "using", "personal data": "your personal info",
                "data protection": "keeping your info safe",
                "biometric data": "body info (like fingerprints!)",
                "implementation": "setting things up", "implement": "set up",
                "jurisdiction": "country", "breach": "leak",
                "penalties": "punishments", "consent": "permission",
                "fundamental": "super important", "principles": "rules",
                "legislation": "law", "violation": "breaking the rules",
                "violations": "breaking the rules",
                "supervisory authority": "data police",
                "transparency": "being open and honest",
                "accountability": "being responsible",
                "lawful basis": "a good reason", "enforceable": "that must be followed",
                "financial": "money-related",
            }
            result = text
            for old, new in replacements.items():
                result = result.replace(old, new)
            prefix = "Imagine this - it's like a superhero story for your data! "
            suffix = " Pretty cool, right? It's all about keeping people's secrets safe!"
            result = prefix + result
            if len(result) + len(suffix) < 550:
                result += suffix
            return result

        def _adult_transform(self, text):
            examples = {
                "data protection": "Think about how your bank protects your account details - ",
                "gdpr": "Like the privacy settings on your Netflix or Facebook account - ",
                "consent": "Similar to accepting cookies on websites or app permissions - ",
                "personal data": "This covers everything from your email to browsing history - ",
                "biometric": "Like Face ID on your iPhone or fingerprint login at your bank - ",
                "breach": "Similar to when companies get hacked and user data leaks - ",
                "penalties": "Companies face massive fines, like when tech giants were fined billions - ",
                "privacy by design": "Like how Apple builds privacy into iPhones from the start - ",
                "rights": "Like how you can delete your Amazon history or download Google data - ",
            }
            text_lower = text.lower()
            prefix = next(
                (ex for kw, ex in examples.items() if kw in text_lower),
                "In practical terms: ",
            )
            return prefix + text

        def _story_transform(self, text):
            intros = [
                ("In 2018, European leaders gathered to face a digital crisis. They saw "
                 "that people's personal information was being collected everywhere - and "
                 "something had to change. "),
                ("Picture a world where every click tells a story about you. In this "
                 "world, a group of determined lawmakers set out to protect citizens. "),
                ("Once upon a time in the European Union, citizens grew worried about "
                 "their digital lives. Their governments listened, and a great plan "
                 "was set in motion. "),
            ]
            tl = text.lower()
            if "2018" in tl or "regulation" in tl or "gdpr" in tl:
                intro = intros[0]
            elif "data" in tl or "personal" in tl:
                intro = intros[1]
            else:
                intro = intros[2]
            conclusion = " And so, the journey toward digital privacy continues to this day."
            result = intro + text
            if len(result) + len(conclusion) < 600:
                result += conclusion
            return result

    class ConversationalRetrievalQA:
        """QA chain with conversational memory and source tracking."""

        def __init__(self, llm, retriever, prompt, memory_key="chat_history"):
            self._llm = llm
            self._retriever = retriever
            self._prompt = prompt
            self._memory_key = memory_key

        def invoke(self, inputs, chat_history=None):
            query = inputs["query"]
            chat_history = chat_history or []

            # Build a standalone question incorporating chat history
            if chat_history:
                history_text = "\n".join(
                    f"Human: {h}\nAssistant: {a}" for h, a in chat_history[-3:]
                )
                condensed_prompt = (
                    f"Given this conversation history:\n{history_text}\n\n"
                    f"And a follow-up question: {query}\n\n"
                    f"Rephrase as a standalone question:"
                )
                standalone = self._llm._call(condensed_prompt)
                if not standalone or len(standalone.strip()) < 5:
                    standalone = query
            else:
                standalone = query

            try:
                docs = self._retriever.invoke(standalone)
            except AttributeError:
                docs = self._retriever.get_relevant_documents(standalone)

            context = "\n\n".join(doc.page_content for doc in docs)
            full_prompt = self._prompt.format(context=context, question=query)
            answer = self._llm._call(full_prompt)

            return {"result": answer, "source_documents": docs}

    chains = {}
    for mode_key in ("kid", "adult", "story"):
        mode_llm = StorytellingLLM(base_llm=llm, mode=mode_key)
        chains[mode_key] = ConversationalRetrievalQA(
            llm=mode_llm,
            retriever=retriever,
            prompt=prompts[mode_key],
        )
    return chains


# ===================================================================
# Session state initialization
# ===================================================================

if "history" not in st.session_state:
    st.session_state["history"] = load_chat_history()

if "conversation" not in st.session_state:
    st.session_state["conversation"] = []  # list of (human, ai) tuples for memory

if "delete_trigger" not in st.session_state:
    st.session_state["delete_trigger"] = None

# ===================================================================
# Sidebar with tabs
# ===================================================================

with st.sidebar:
    st.markdown("## \U0001f52e RAG Assistant Enhanced")
    st.caption("Upload \u00b7 Manage \u00b7 Chat History")

    tab_upload, tab_manage, tab_history = st.tabs([
        "\U0001f4e4 Upload", "\U0001f5c4 Manage", "\U0001f4ac History"
    ])

    # ---- UPLOAD TAB ----
    with tab_upload:
        st.markdown("### Upload PDFs")
        uploaded_files = st.file_uploader(
            "Drop PDF files here",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files to analyze",
        )

        if uploaded_files:
            saved_files = []
            for uf in uploaded_files:
                dest = os.path.join(UPLOADED_PDFS_DIR, uf.name)
                if not os.path.exists(dest):
                    with open(dest, "wb") as f:
                        f.write(uf.getbuffer())
                    saved_files.append(uf.name)

            if saved_files:
                st.success(f"Saved {len(saved_files)} new PDF(s): {', '.join(saved_files)}")
                for key in ("vectordb", "chains", "chunk_count"):
                    st.session_state.pop(key, None)

            # Auto-process uploaded files
            if saved_files and st.button("\U0001f504 Process & Index Now", use_container_width=True, key="process_btn"):
                with st.spinner("Processing uploaded PDFs..."):
                    if "embedding_model" not in st.session_state:
                        st.session_state["embedding_model"] = get_embedding_model()
                    emb = st.session_state["embedding_model"]
                    new_paths = [Path(UPLOADED_PDFS_DIR) / fn for fn in saved_files]
                    docs, meta = load_pdf_documents(new_paths)
                    if docs:
                        _, count = build_vectordb(docs, emb)
                        st.session_state.pop("vectordb", None)
                        st.session_state.pop("chains", None)
                        st.success(f"Indexed {count} chunks from {len(saved_files)} file(s)")
                        st.rerun()

        st.markdown("---")
        st.markdown("**Available PDFs:**")
        available_pdfs = list_all_pdfs()
        if available_pdfs:
            for pdf_path in available_pdfs:
                size_kb = pdf_path.stat().st_size / 1024
                st.markdown(
                    f"\U0001f4c4 **{pdf_path.name}** ({size_kb:.0f} KB)",
                )
        else:
            st.info("No PDFs found. Upload some above!")

        st.markdown("---")
        if st.button("\U0001f504 Rebuild Vector Database", use_container_width=True, key="rebuild_btn"):
            for key in ("vectordb", "chains", "chunk_count"):
                st.session_state.pop(key, None)
            st.rerun()

    # ---- MANAGE TAB ----
    with tab_manage:
        st.markdown("### Vector Database Stats")

        if "embedding_model" not in st.session_state:
            with st.spinner("Loading embedding model..."):
                st.session_state["embedding_model"] = get_embedding_model()
        emb = st.session_state["embedding_model"]

        if st.button("\U0001f504 Refresh Stats", use_container_width=True, key="refresh_stats"):
            st.session_state.pop("db_stats", None)

        total_chunks, doc_chunks = get_vectordb_stats(emb)
        st.session_state["db_stats"] = (total_chunks, doc_chunks)

        st.metric("Total Chunks Stored", total_chunks)
        st.metric("Documents Indexed", len(doc_chunks))

        if doc_chunks:
            st.markdown("---")
            st.markdown("### Per-Document Breakdown")
            for doc_name, count in sorted(doc_chunks.items()):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(
                        f'<div class="doc-row">\U0001f4c4 <strong>{doc_name}</strong>'
                        f' &mdash; {count} chunks</div>',
                        unsafe_allow_html=True,
                    )
                with col2:
                    if st.button(
                        "\U0001f5d1",
                        key=f"del_{hashlib.md5(doc_name.encode()).hexdigest()[:8]}",
                        help=f"Delete all chunks from {doc_name}",
                    ):
                        st.session_state["delete_trigger"] = doc_name

        # Handle delete outside of the loop to avoid rerun issues
        if st.session_state.get("delete_trigger"):
            doc_to_delete = st.session_state["delete_trigger"]
            st.session_state["delete_trigger"] = None
            with st.spinner(f"Deleting chunks from {doc_to_delete}..."):
                deleted = delete_document_chunks(emb, doc_to_delete)
                for key in ("vectordb", "chains", "chunk_count", "db_stats"):
                    st.session_state.pop(key, None)
            st.success(f"Deleted {deleted} chunks from {doc_to_delete}")
            st.rerun()

        if total_chunks == 0:
            st.info("No chunks in the vector database yet. Upload and index PDFs first.")

    # ---- HISTORY TAB ----
    with tab_history:
        st.markdown("### Chat History")
        history = st.session_state.get("history", [])
        st.metric("Total Messages", len(history))

        if history:
            if st.button("\U0001f5d1 Clear All History", use_container_width=True, type="secondary", key="clear_history"):
                st.session_state["history"] = []
                st.session_state["conversation"] = []
                save_chat_history([])
                st.success("History cleared!")
                st.rerun()

            if st.button("\U0001f195 New Topic", use_container_width=True, key="new_topic"):
                st.session_state["conversation"] = []
                st.success("Conversation memory reset! Start a fresh topic.")

            st.markdown("---")
            st.markdown("**Recent questions:**")
            for entry in reversed(history[-10:]):
                mode_icon = {"kid": "\U0001f9d2", "adult": "\U0001f454", "story": "\U0001f4d6"}.get(
                    entry.get("mode_key", ""), "\u2753"
                )
                st.caption(f"{mode_icon} {entry.get('question', '?')[:60]}")
        else:
            st.info("No chat history yet. Ask a question to get started!")


# ===================================================================
# Header
# ===================================================================

st.markdown(
    '<div class="main-header">'
    "<h1>\U0001f52e Quest Analytics RAG Assistant</h1>"
    "<p>Enhanced \u00b7 Conversational Memory \u00b7 PDF Management \u00b7 Storytelling Modes</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ===================================================================
# Initialize core components
# ===================================================================

if "embedding_model" not in st.session_state:
    with st.spinner("Loading embedding model (all-MiniLM-L6-v2)..."):
        st.session_state["embedding_model"] = get_embedding_model()
embedding_model = st.session_state["embedding_model"]

if "llm" not in st.session_state:
    with st.spinner("Initializing language model..."):
        llm, model_label = get_llm()
        st.session_state["llm"] = llm
        st.session_state["model_label"] = model_label
llm = st.session_state["llm"]
model_label = st.session_state["model_label"]

# --- Vector DB ---
if "vectordb" not in st.session_state:
    pdf_paths = list_all_pdfs()
    if pdf_paths:
        vdb, count = load_existing_vectordb(embedding_model)
        if vdb is not None:
            st.session_state["vectordb"] = vdb
            st.session_state["chunk_count"] = count
        else:
            with st.spinner("Building vector database from PDFs..."):
                docs, meta = load_pdf_documents(pdf_paths)
                if docs:
                    vdb, count = build_vectordb(docs, embedding_model)
                    st.session_state["vectordb"] = vdb
                    st.session_state["chunk_count"] = count

# --- QA Chains with MMR retrieval ---
if "chains" not in st.session_state and "vectordb" in st.session_state:
    retriever = st.session_state["vectordb"].as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7},
    )
    st.session_state["chains"] = get_storytelling_chains(llm, retriever)

# ===================================================================
# Status bar
# ===================================================================

status_cols = st.columns(4)
with status_cols[0]:
    st.metric("\U0001f4c4 PDFs Loaded", len(list_all_pdfs()))
with status_cols[1]:
    st.metric("\U0001f9e9 Chunks Indexed", st.session_state.get("chunk_count", 0))
with status_cols[2]:
    st.metric("\U0001f916 LLM", model_label.split("/")[-1][:18])
with status_cols[3]:
    conv_len = len(st.session_state.get("conversation", []))
    st.metric("\U0001f4ac Memory", f"{conv_len} turns")

st.markdown("---")

# ===================================================================
# Mode selector & question input
# ===================================================================

col_mode, col_input = st.columns([1, 3])

MODE_CONFIG = {
    "\U0001f9d2 Kid Mode": {"key": "kid", "desc": "Simple words & fun analogies (ages 8-12)"},
    "\U0001f454 Adult Mode": {"key": "adult", "desc": "Real-life examples (Netflix, banking...)"},
    "\U0001f4d6 Story Mode": {"key": "story", "desc": "Narrative format with characters & plot"},
}

with col_mode:
    st.markdown("### \U0001f3ad Choose a Mode")
    selected_mode_label = st.radio(
        "Explanation style",
        list(MODE_CONFIG.keys()),
        index=1,
        label_visibility="collapsed",
    )
    mode_key = MODE_CONFIG[selected_mode_label]["key"]
    st.caption(MODE_CONFIG[selected_mode_label]["desc"])

with col_input:
    st.markdown("### \U0001f4ac Ask a Question")
    question = st.text_input(
        "Type your question about the loaded documents...",
        placeholder="e.g. What is GDPR and why is it important?",
        label_visibility="collapsed",
    )
    ask_clicked = st.button("\U0001f680 Get Answer", type="primary", use_container_width=True)

# ===================================================================
# Process question
# ===================================================================

if ask_clicked and question.strip():
    if "chains" not in st.session_state:
        st.error("No documents indexed yet. Upload PDFs and rebuild the vector database.")
    else:
        chains = st.session_state["chains"]
        chain = chains[mode_key]
        chat_history = st.session_state.get("conversation", [])

        with st.spinner("\U0001f50d Searching documents & generating answer..."):
            progress = st.progress(0, text="Retrieving relevant chunks...")
            try:
                progress.progress(30, text="Running through language model...")
                result = chain.invoke({"query": question}, chat_history=chat_history)
                progress.progress(80, text="Formatting answer...")

                answer = result["result"].strip()
                source_docs = result["source_documents"]
                sources = []
                for doc in source_docs:
                    src = doc.metadata.get("source_file", "Unknown")
                    page = doc.metadata.get("page", "?")
                    sources.append({"file": src, "page": page})

                progress.progress(100, text="Done!")
                time.sleep(0.3)
                progress.empty()

                # Update conversation memory
                st.session_state["conversation"].append((question, answer))

                # Store in history
                entry = {
                    "question": question,
                    "mode": selected_mode_label,
                    "mode_key": mode_key,
                    "answer": answer,
                    "sources": sources,
                    "timestamp": datetime.now().isoformat(),
                }
                st.session_state["history"].append(entry)
                save_chat_history(st.session_state["history"])

            except Exception as exc:
                progress.empty()
                st.error(f"Error generating answer: {exc}")

elif ask_clicked:
    st.warning("Please type a question first.")

# ===================================================================
# Display conversation as chat bubbles
# ===================================================================

if st.session_state["history"]:
    st.markdown("---")

    hist_col1, hist_col2 = st.columns([3, 1])
    with hist_col1:
        st.markdown("## \U0001f4dd Conversation")
    with hist_col2:
        st.caption(f"{len(st.session_state['history'])} messages")

    for i, entry in enumerate(reversed(st.session_state["history"])):
        idx = len(st.session_state["history"]) - 1 - i

        # User message bubble
        mode_icon = {"kid": "\U0001f9d2", "adult": "\U0001f454", "story": "\U0001f4d6"}.get(
            entry.get("mode_key", ""), ""
        )
        ts = entry.get("timestamp", "")
        ts_display = ""
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                ts_display = f" <span style='color:#666;font-size:.75rem;'>{dt.strftime('%H:%M')}</span>"
            except Exception:
                pass

        st.markdown(
            f'<div class="chat-user">'
            f'<strong>\U0001f464 You</strong> {mode_icon}{ts_display}<br>'
            f'{entry["question"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Bot message bubble
        st.markdown(
            f'<div class="chat-bot">'
            f'<strong>\U0001f916 Assistant</strong> ({entry.get("mode", "")}) <br>'
            f'{entry["answer"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Source citations as badges
        if entry.get("sources"):
            seen = set()
            source_html = ""
            for s in entry["sources"]:
                badge = f'{s["file"]} p{s["page"]}'
                if badge not in seen:
                    seen.add(badge)
                    source_html += f'<span class="source-badge">{badge}</span>'
            st.markdown(f"**Sources:** {source_html}", unsafe_allow_html=True)

        # Switch-mode buttons
        btn_cols = st.columns(3)
        for col, (label, cfg) in zip(btn_cols, MODE_CONFIG.items()):
            with col:
                if cfg["key"] != entry.get("mode_key"):
                    if st.button(
                        label,
                        key=f"switch_{idx}_{cfg['key']}",
                        use_container_width=True,
                    ):
                        if "chains" in st.session_state:
                            with st.spinner(f"Re-generating in {label}..."):
                                res = st.session_state["chains"][cfg["key"]].invoke(
                                    {"query": entry["question"]}
                                )
                                new_answer = res["result"].strip()
                                new_sources = [
                                    {
                                        "file": d.metadata.get("source_file", "?"),
                                        "page": d.metadata.get("page", "?"),
                                    }
                                    for d in res["source_documents"]
                                ]
                                st.session_state["history"].append({
                                    "question": entry["question"],
                                    "mode": label,
                                    "mode_key": cfg["key"],
                                    "answer": new_answer,
                                    "sources": new_sources,
                                    "timestamp": datetime.now().isoformat(),
                                })
                                save_chat_history(st.session_state["history"])
                                st.rerun()

        st.markdown("<hr style='border-color:rgba(255,255,255,.06);'>", unsafe_allow_html=True)

# ===================================================================
# Empty state
# ===================================================================

if not st.session_state["history"] and not ask_clicked:
    st.markdown(
        """
        <div style="text-align:center; padding:3rem 0; color:#888;">
            <h2 style="color:#e94560;">Welcome! \U0001f44b</h2>
            <p style="font-size:1.1rem;">
                Upload PDF documents in the sidebar, then ask a question above.<br>
                Choose between <strong>Kid</strong>, <strong>Adult</strong>, or
                <strong>Story</strong> mode to change how answers are explained.
            </p>
            <p style="font-size:.95rem; color:#aaa; margin-top:.8rem;">
                <strong>New in Enhanced version:</strong><br>
                \U0001f4e4 Upload &amp; manage PDFs from the sidebar<br>
                \U0001f9e0 Conversational memory for follow-up questions<br>
                \U0001f4ac Chat bubbles with timestamps<br>
                \U0001f5d1 Delete documents from vector database<br>
                \U0001f4be Persistent chat history (saved to disk)
            </p>
            <p style="font-size:.9rem; color:#666; margin-top:1rem;">
                Powered by LangChain &bull; ChromaDB &bull; all-MiniLM-L6-v2 &bull;
                Hugging Face FLAN-T5
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
