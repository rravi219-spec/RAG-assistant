"""
Quest Analytics - RAG Storytelling Assistant (Streamlit Web Interface)
=====================================================================
A professional web UI for the RAG pipeline with storytelling modes.

Run with:  streamlit run app.py
"""

import os
import sys
import subprocess
import time
import shutil
import tempfile
from pathlib import Path

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
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

os.makedirs(PDFS_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# Support both .env (local) and st.secrets (Streamlit Cloud)
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
if not HF_API_KEY:
    HF_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", "")
    if HF_API_KEY:
        os.environ["HUGGINGFACE_API_KEY"] = HF_API_KEY

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Quest Analytics RAG Assistant",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a clean, modern look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Overall dark theme refinement */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%);
    }

    /* Header bar */
    .main-header {
        background: linear-gradient(90deg, #e94560 0%, #c23152 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: .5px;
    }
    .main-header p {
        color: rgba(255,255,255,.85);
        margin: .3rem 0 0 0;
        font-size: 1rem;
    }

    /* Source badge */
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

    /* Mode card styling */
    .mode-card {
        background: rgba(255,255,255,.04);
        border: 1px solid rgba(255,255,255,.1);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: .5rem;
    }

    /* Sidebar tweaks */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16213e 0%, #1a1a2e 100%);
    }

    /* Hide default Streamlit footer */
    footer {visibility: hidden;}

    /* Answer area */
    .answer-box {
        background: rgba(255,255,255,.03);
        border-left: 4px solid #e94560;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: .8rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===================================================================
# Heavy-lifting functions wrapped in Streamlit caching
# ===================================================================

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Return a LangChain-compatible embedding model (cached once)."""
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    from langchain_core.embeddings import Embeddings

    chroma_ef = DefaultEmbeddingFunction()

    class OnnxEmbeddings(Embeddings):
        """LangChain wrapper around ChromaDB ONNX embeddings (all-MiniLM-L6-v2)."""

        def __init__(self, ef):
            self._ef = ef

        def embed_documents(self, texts):
            return [emb.tolist() for emb in self._ef(texts)]

        def embed_query(self, text):
            return self._ef([text])[0].tolist()

    return OnnxEmbeddings(chroma_ef)


@st.cache_resource(show_spinner=False)
def get_llm():
    """Return the LLM instance (HuggingFace or fallback), cached once."""
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

    # Fallback: extractive summary LLM
    from langchain_core.language_models.llms import LLM
    from typing import Any, Optional

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


def build_vectordb(documents, embedding_model):
    """Build (or rebuild) the ChromaDB vector store from documents."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR,
        collection_name="quest_analytics_papers",
    )
    return vectordb, len(chunks)


def load_existing_vectordb(embedding_model):
    """Load an already-persisted ChromaDB (no rebuild)."""
    from langchain_community.vectorstores import Chroma

    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model,
        collection_name="quest_analytics_papers",
    )
    count = vectordb._collection.count()
    if count == 0:
        return None, 0
    return vectordb, count


def get_storytelling_chains(llm, retriever):
    """Build the three storytelling QA chains."""
    from langchain_core.prompts import PromptTemplate
    from langchain_core.language_models.llms import LLM as BaseLLMClass
    from typing import Any, Optional

    # Prompt templates -------------------------------------------------------
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

    # StorytellingLLM wrapper ------------------------------------------------
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

    # Simple retrieval QA chain (no dependency on langchain.chains) ----------
    class SimpleRetrievalQA:
        """Drop-in replacement for RetrievalQA that only needs langchain_core."""

        def __init__(self, llm, retriever, prompt):
            self._llm = llm
            self._retriever = retriever
            self._prompt = prompt

        def invoke(self, inputs):
            query = inputs["query"]
            try:
                docs = self._retriever.invoke(query)
            except AttributeError:
                docs = self._retriever.get_relevant_documents(query)
            context = "\n\n".join(doc.page_content for doc in docs)
            full_prompt = self._prompt.format(context=context, question=query)
            answer = self._llm._call(full_prompt)
            return {"result": answer, "source_documents": docs}

    # Build chains -----------------------------------------------------------
    chains = {}
    for mode_key in ("kid", "adult", "story"):
        mode_llm = StorytellingLLM(base_llm=llm, mode=mode_key)
        chains[mode_key] = SimpleRetrievalQA(
            llm=mode_llm,
            retriever=retriever,
            prompt=prompts[mode_key],
        )
    return chains


# ===================================================================
# PDF helpers
# ===================================================================

def list_loaded_pdfs():
    """Return list of PDF filenames currently in pdfs/ or root."""
    pdfs = list(Path(PDFS_DIR).glob("*.pdf"))
    root_pdfs = list(Path(BASE_DIR).glob("*.pdf"))
    all_paths = {str(p): p for p in pdfs + root_pdfs}
    return list(all_paths.values())


def load_pdf_documents(pdf_paths):
    """Load LangChain Documents from a list of PDF file paths."""
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
# Incremental PDF loading helpers
# ===================================================================

def get_all_pdfs_in_folder():
    """Get set of all PDF filenames from pdfs/ and project root."""
    pdf_paths = list_loaded_pdfs()
    return {p.name for p in pdf_paths}


def get_processed_pdfs(vectorstore):
    """Get set of PDF filenames already indexed in the vector database."""
    if vectorstore is None:
        return set()
    try:
        collection = vectorstore._collection
        all_metadata = collection.get(include=["metadatas"])
        processed = set()
        for metadata in all_metadata.get("metadatas", []):
            if metadata and "source_file" in metadata:
                processed.add(metadata["source_file"])
        return processed
    except Exception:
        return set()


def get_vector_db_stats(vectorstore):
    """Return dict with total_chunks, num_documents, and document list."""
    if vectorstore is None:
        return {"total_chunks": 0, "num_documents": 0, "documents": []}
    try:
        collection = vectorstore._collection
        total = collection.count()
        all_metadata = collection.get(include=["metadatas"])
        doc_chunks = {}
        for metadata in all_metadata.get("metadatas", []):
            if metadata and "source_file" in metadata:
                name = metadata["source_file"]
                doc_chunks[name] = doc_chunks.get(name, 0) + 1
        docs = [{"name": k, "chunks": v} for k, v in sorted(doc_chunks.items())]
        return {
            "total_chunks": total,
            "num_documents": len(doc_chunks),
            "documents": docs,
        }
    except Exception:
        return {"total_chunks": 0, "num_documents": 0, "documents": []}


def process_new_pdfs_only(vectorstore, embedding_model):
    """Process only NEW PDFs not yet in the database. Returns (vectorstore, bool)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    all_pdfs = get_all_pdfs_in_folder()
    processed_pdfs = get_processed_pdfs(vectorstore)
    new_pdfs = all_pdfs - processed_pdfs

    if not new_pdfs:
        st.sidebar.info(f"✅ All {len(all_pdfs)} PDF(s) already processed!")
        return vectorstore, False

    st.sidebar.info(f"🆕 Found {len(new_pdfs)} new PDF(s) to process...")

    # Resolve full paths for new PDFs
    new_paths = []
    for pdf_path in list_loaded_pdfs():
        if pdf_path.name in new_pdfs:
            new_paths.append(pdf_path)

    if not new_paths:
        return vectorstore, False

    docs, meta = load_pdf_documents(new_paths)
    if not docs:
        st.sidebar.warning("⚠️ No content extracted from new PDFs.")
        return vectorstore, False

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    total_new = 0
    for pdf_name in new_pdfs:
        pdf_chunks = [c for c in chunks if c.metadata.get("source_file") == pdf_name]
        if pdf_chunks:
            try:
                vectorstore.add_documents(pdf_chunks)
                total_new += len(pdf_chunks)
                st.sidebar.success(f"✅ {pdf_name}: {len(pdf_chunks)} chunks added")
            except Exception as e:
                st.sidebar.error(f"❌ {pdf_name}: {e}")

    if total_new > 0:
        st.sidebar.success(f"🎉 Total: {total_new} new chunks added to database!")

    # Update cached chunk count
    st.session_state["chunk_count"] = vectorstore._collection.count()

    return vectorstore, total_new > 0


# ===================================================================
# Sidebar: Document Library & Upload
# ===================================================================

with st.sidebar:
    st.markdown("## 📚 Document Library")
    st.caption("Upload PDFs or use existing documents")

    # PDF upload widget (drag & drop)
    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to analyze",
    )

    if uploaded_files:
        saved = []
        for uf in uploaded_files:
            dest = os.path.join(PDFS_DIR, uf.name)
            if not os.path.exists(dest):
                with open(dest, "wb") as f:
                    f.write(uf.getbuffer())
                saved.append(uf.name)
        if saved:
            st.success(f"Saved {len(saved)} new PDF(s)")
            # Clear cached vectordb so it rebuilds with new docs
            if "vectordb" in st.session_state:
                del st.session_state["vectordb"]
            if "chains" in st.session_state:
                del st.session_state["chains"]

    st.markdown("---")
    st.markdown("### Loaded Documents")

    available_pdfs = list_loaded_pdfs()
    if available_pdfs:
        for pdf_path in available_pdfs:
            size_kb = pdf_path.stat().st_size / 1024
            st.markdown(
                f"📄 **{pdf_path.name}** &nbsp; "
                f"<span style='color:#888;font-size:.8rem;'>({size_kb:.0f} KB)</span>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No PDFs found. Upload some above!")

    st.markdown("---")

    # --- Database Statistics ---
    st.markdown("### 📊 Database Stats")
    if "vectordb" in st.session_state:
        stats = get_vector_db_stats(st.session_state["vectordb"])
        stat_cols = st.columns(2)
        with stat_cols[0]:
            st.metric("Total Chunks", stats["total_chunks"])
        with stat_cols[1]:
            st.metric("Documents", stats["num_documents"])
        if stats["documents"]:
            st.markdown("**Indexed documents:**")
            for doc in stats["documents"]:
                st.markdown(
                    f"&nbsp;&nbsp;📄 {doc['name']} — `{doc['chunks']}` chunks"
                )
    else:
        st.caption("No database loaded yet.")

    st.markdown("---")

    # Rebuild button
    if st.button("🔄 Rebuild Vector Database", use_container_width=True):
        for key in ("vectordb", "chains", "chunk_count"):
            st.session_state.pop(key, None)
        # Clear the persisted chroma_db so it rebuilds from scratch
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
            os.makedirs(CHROMA_DIR, exist_ok=True)
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#666;font-size:.75rem;'>"
        "Quest Analytics RAG Assistant<br>Phase 1 + Phase 2</div>",
        unsafe_allow_html=True,
    )


# ===================================================================
# Header
# ===================================================================

st.markdown(
    '<div class="main-header">'
    "<h1>🔮 Quest Analytics RAG Assistant</h1>"
    "<p>AI-Powered Document Analysis with Storytelling Modes</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ===================================================================
# Initialize core components (cached)
# ===================================================================

# --- Embedding model (always needed) ---
if "embedding_model" not in st.session_state:
    with st.spinner("Loading embedding model (all-MiniLM-L6-v2)..."):
        st.session_state["embedding_model"] = get_embedding_model()

embedding_model = st.session_state["embedding_model"]

# --- LLM ---
if "llm" not in st.session_state:
    with st.spinner("Initializing language model..."):
        llm, model_label = get_llm()
        st.session_state["llm"] = llm
        st.session_state["model_label"] = model_label

llm = st.session_state["llm"]
model_label = st.session_state["model_label"]

# --- Vector DB (with incremental PDF loading) ---
if "vectordb" not in st.session_state:
    pdf_paths = list_loaded_pdfs()
    if pdf_paths:
        # Try loading existing DB first
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

# Check for new PDFs every time (even if DB already loaded)
if "vectordb" in st.session_state and list_loaded_pdfs():
    vdb, new_added = process_new_pdfs_only(
        st.session_state["vectordb"], embedding_model
    )
    st.session_state["vectordb"] = vdb
    if new_added:
        # Rebuild QA chains with updated retriever
        st.session_state.pop("chains", None)

# --- QA Chains ---
if "chains" not in st.session_state and "vectordb" in st.session_state:
    retriever = st.session_state["vectordb"].as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )
    st.session_state["chains"] = get_storytelling_chains(llm, retriever)

# ===================================================================
# Status bar
# ===================================================================

status_cols = st.columns(4)
with status_cols[0]:
    pdf_count = len(list_loaded_pdfs())
    st.metric("📄 PDFs Loaded", pdf_count)
with status_cols[1]:
    chunk_count = st.session_state.get("chunk_count", 0)
    st.metric("🧩 Chunks Indexed", chunk_count)
with status_cols[2]:
    st.metric("🤖 LLM", model_label.split("/")[-1][:18])
with status_cols[3]:
    st.metric("📐 Embed Dim", "384")

st.markdown("---")

# ===================================================================
# Conversation history
# ===================================================================

if "history" not in st.session_state:
    st.session_state["history"] = []  # list of dicts

# ===================================================================
# Mode selector & question input
# ===================================================================

col_mode, col_input = st.columns([1, 3])

MODE_CONFIG = {
    "🧒 Kid Mode": {"key": "kid", "desc": "Simple words & fun analogies (ages 8-12)"},
    "👔 Adult Mode": {"key": "adult", "desc": "Real-life examples (Netflix, banking...)"},
    "📖 Story Mode": {"key": "story", "desc": "Narrative format with characters & plot"},
}

with col_mode:
    st.markdown("### 🎭 Choose a Mode")
    selected_mode_label = st.radio(
        "Explanation style",
        list(MODE_CONFIG.keys()),
        index=1,
        label_visibility="collapsed",
    )
    mode_key = MODE_CONFIG[selected_mode_label]["key"]
    st.caption(MODE_CONFIG[selected_mode_label]["desc"])

with col_input:
    st.markdown("### 💬 Ask a Question")
    question = st.text_input(
        "Type your question about the loaded documents...",
        placeholder="e.g. What is GDPR and why is it important?",
        label_visibility="collapsed",
    )
    ask_clicked = st.button("🚀 Get Answer", type="primary", use_container_width=True)

# ===================================================================
# Process question
# ===================================================================

if ask_clicked and question.strip():
    if "chains" not in st.session_state:
        st.error("No documents indexed yet. Please upload PDFs and rebuild the vector database.")
    else:
        chains = st.session_state["chains"]
        chain = chains[mode_key]

        with st.spinner("🔍 Searching documents & generating answer..."):
            progress = st.progress(0, text="Retrieving relevant chunks...")
            try:
                progress.progress(30, text="Running through language model...")
                result = chain.invoke({"query": question})
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

                # Store in history
                st.session_state["history"].append({
                    "question": question,
                    "mode": selected_mode_label,
                    "mode_key": mode_key,
                    "answer": answer,
                    "sources": sources,
                })

            except Exception as exc:
                progress.empty()
                st.error(f"Error generating answer: {exc}")

elif ask_clicked:
    st.warning("Please type a question first.")

# ===================================================================
# Display conversation history (newest first)
# ===================================================================

if st.session_state["history"]:
    st.markdown("---")
    st.markdown("## 📝 Conversation History")

    for i, entry in enumerate(reversed(st.session_state["history"])):
        idx = len(st.session_state["history"]) - 1 - i
        with st.container():
            # Question header
            st.markdown(
                f"**{entry['mode']}** &nbsp;|&nbsp; *Q: {entry['question']}*"
            )

            # Answer
            st.markdown(
                f'<div class="answer-box">{entry["answer"]}</div>',
                unsafe_allow_html=True,
            )

            # Sources
            if entry["sources"]:
                source_html = " ".join(
                    f'<span class="source-badge">{s["file"]} p{s["page"]}</span>'
                    for s in entry["sources"]
                )
                st.markdown(f"**Sources:** {source_html}", unsafe_allow_html=True)

            # Switch-mode buttons
            st.caption("Switch explanation style for this question:")
            btn_cols = st.columns(3)
            for col, (label, cfg) in zip(btn_cols, MODE_CONFIG.items()):
                with col:
                    if cfg["key"] != entry["mode_key"]:
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
                                    })
                                    st.rerun()

            st.markdown("---")

# ===================================================================
# Empty state
# ===================================================================

if not st.session_state["history"] and not ask_clicked:
    st.markdown(
        """
        <div style="text-align:center; padding:3rem 0; color:#888;">
            <h2 style="color:#e94560;">Welcome! 👋</h2>
            <p style="font-size:1.1rem;">
                Upload PDF documents in the sidebar, then ask a question above.<br>
                Choose between <strong>Kid</strong>, <strong>Adult</strong>, or
                <strong>Story</strong> mode to change how answers are explained.
            </p>
            <p style="font-size:.9rem; color:#666; margin-top:1rem;">
                Powered by LangChain &bull; ChromaDB &bull; all-MiniLM-L6-v2 &bull;
                Hugging Face FLAN-T5
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
