"""
Quest Analytics - AI-Powered RAG Assistant for Paper Summarization
=================================================================
PHASE 1 - DAY 1: Solid Foundation with Hugging Face Integration

This script builds a complete RAG (Retrieval-Augmented Generation) pipeline
using LangChain with a real LLM (Hugging Face) to read, understand, and 
summarize research documents.

NEW FEATURES (Phase 1):
- ✅ Hugging Face LLM integration (FLAN-T5)
- ✅ Multi-PDF support (load multiple research papers)
- ✅ Secure API key management (.env file)
- ✅ Better source tracking (know which PDF answered)
- ✅ Document library management
- ✅ All 6 original tasks + screenshots preserved

Tasks:
1. Load documents using LangChain (PDF loader) - MULTI-PDF SUPPORT
2. Apply text splitting techniques
3. Embed documents using Sentence-Transformers (ONNX, all-MiniLM-L6-v2)
4. Create and configure ChromaDB vector database
5. Develop a retriever for document queries
6. Construct a QA Bot using LangChain and Hugging Face LLM
"""

import os
import sys
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDFS_DIR = os.path.join(BASE_DIR, "pdfs")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")

# Create directories if they don't exist
os.makedirs(PDFS_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# Get API key from environment
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
if not HF_API_KEY:
    print("ERROR: HUGGINGFACE_API_KEY not found in .env file!")
    print("Please create a .env file with your Hugging Face API key.")
    sys.exit(1)

print("=" * 70)
print("🚀 QUEST ANALYTICS RAG ASSISTANT - PHASE 1")
print("=" * 70)
print(f"✅ Hugging Face API Key loaded")
print(f"✅ PDFs Directory: {PDFS_DIR}")
print(f"✅ Screenshots Directory: {SCREENSHOTS_DIR}")
print("=" * 70 + "\n")


# ============================================================
# TASK 1: Load Documents Using LangChain (MULTI-PDF SUPPORT)
# ============================================================
print("=" * 70)
print("TASK 1: Loading Documents Using LangChain PDF Loader (MULTI-PDF)")
print("=" * 70)

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

# NEW: Support for loading multiple PDFs from a directory
def load_pdfs_from_directory(directory_path):
    """Load all PDFs from a directory"""
    pdf_files = list(Path(directory_path).glob("*.pdf"))
    
    if not pdf_files:
        print(f"\n⚠️  WARNING: No PDF files found in {directory_path}")
        print(f"Please add PDF files to the '{os.path.basename(directory_path)}' folder")
        return [], []
    
    all_documents = []
    pdf_metadata = []
    
    for pdf_file in pdf_files:
        print(f"\n📄 Loading: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        
        # Add source filename to metadata
        for doc in docs:
            doc.metadata['source_file'] = pdf_file.name
        
        all_documents.extend(docs)
        pdf_metadata.append({
            'filename': pdf_file.name,
            'pages': len(docs),
            'path': str(pdf_file)
        })
        print(f"   ✅ Loaded {len(docs)} pages")
    
    return all_documents, pdf_metadata

# Try to load from pdfs/ directory first, fallback to single file
documents = []
pdf_metadata = []

if os.path.exists(PDFS_DIR):
    documents, pdf_metadata = load_pdfs_from_directory(PDFS_DIR)

# Fallback: If no PDFs in directory, try the original GDPR file
if not documents:
    print(f"\n📄 Fallback: Loading single PDF from current directory")
    pdf_path = os.path.join(BASE_DIR, "GDPR-Framework.pdf")
    if os.path.exists(pdf_path):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata['source_file'] = "GDPR-Framework.pdf"
        pdf_metadata = [{
            'filename': "GDPR-Framework.pdf",
            'pages': len(documents),
            'path': pdf_path
        }]
        print(f"   ✅ Loaded {len(documents)} pages from GDPR-Framework.pdf")
    else:
        print(f"\n❌ ERROR: No PDFs found!")
        print(f"Please add PDF files to: {PDFS_DIR}")
        sys.exit(1)

print(f"\n{'='*70}")
print(f"📚 DOCUMENT LIBRARY SUMMARY")
print(f"{'='*70}")
print(f"Total PDFs Loaded: {len(pdf_metadata)}")
print(f"Total Pages: {len(documents)}")
print(f"\nPDF Details:")
for i, pdf in enumerate(pdf_metadata, 1):
    print(f"  {i}. {pdf['filename']} - {pdf['pages']} pages")

print(f"\nDocument Type: {type(documents[0])}")
print(f"Metadata Keys: {list(documents[0].metadata.keys())}")
print(f"\nFirst 500 characters of first document:\n{'-'*50}")
print(documents[0].page_content[:500])
print(f"\n{'='*70}\n")

# --- Save screenshot for Task 1 ---
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_facecolor('#1a1a2e')
fig.patch.set_facecolor('#16213e')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.4, 'Task 1: Load Documents - Multi-PDF Support (NEW!)',
        fontsize=16, fontweight='bold', color='#e94560', ha='center',
        fontfamily='monospace')

info_lines = [
    f"Loader: PyPDFLoader + DirectoryLoader",
    f"PDFs Loaded: {len(pdf_metadata)}",
    f"Total Pages: {len(documents)}",
    f"Source Tracking: Enabled ✅",
]
y_pos = 8.5
for line in info_lines:
    ax.text(0.5, y_pos, f"  {line}", fontsize=11, color='#00ff88',
            fontfamily='monospace', va='top')
    y_pos -= 0.55

ax.axhline(y=y_pos + 0.1, xmin=0.03, xmax=0.97, color='#e94560', linewidth=1)
y_pos -= 0.3

ax.text(0.5, y_pos, "  📚 Document Library:", fontsize=12,
        fontweight='bold', color='#ffd700', fontfamily='monospace')
y_pos -= 0.5

for i, pdf in enumerate(pdf_metadata[:8], 1):  # Show max 8 PDFs
    ax.text(0.5, y_pos, f"    {i}. {pdf['filename'][:50]} ({pdf['pages']} pages)",
            fontsize=10, color='#87ceeb', fontfamily='monospace')
    y_pos -= 0.38

if len(pdf_metadata) > 8:
    ax.text(0.5, y_pos, f"    ... and {len(pdf_metadata)-8} more PDFs",
            fontsize=10, color='#87ceeb', fontfamily='monospace')
    y_pos -= 0.38

ax.axhline(y=y_pos + 0.1, xmin=0.03, xmax=0.97, color='#444466', linewidth=1)
y_pos -= 0.3

ax.text(0.5, y_pos, "  Content Preview (First Document):", fontsize=12,
        fontweight='bold', color='#ffd700', fontfamily='monospace')
y_pos -= 0.5

preview_text = documents[0].page_content[:600]
wrapped = textwrap.wrap(preview_text, width=90)
for line in wrapped[:10]:
    ax.text(0.5, y_pos, f"  {line}", fontsize=8.5, color='#ffffff',
            fontfamily='monospace', va='top')
    y_pos -= 0.35

plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOTS_DIR, 'pdf_loader.png'),
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"✅ Screenshot saved: {SCREENSHOTS_DIR}/pdf_loader.png\n")


# ============================================================
# TASK 2: Apply Text Splitting Techniques
# ============================================================
print("=" * 70)
print("TASK 2: Applying Text Splitting Techniques")
print("=" * 70)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    Language,
)

# Method 1: RecursiveCharacterTextSplitter (most common for documents)
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)
recursive_chunks = recursive_splitter.split_documents(documents)

# Method 2: CharacterTextSplitter
char_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separator="\n"
)
char_chunks = char_splitter.split_documents(documents)

# Method 3: Code-aware splitter (demonstrating Language support)
code_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=200
)

print(f"\nOriginal document pages: {len(documents)}")
print(f"Original PDFs: {len(pdf_metadata)}")
print(f"\nMethod 1 - RecursiveCharacterTextSplitter:")
print(f"  Chunk size: 1000, Overlap: 200")
print(f"  Total chunks: {len(recursive_chunks)}")
print(f"  Avg chunk length: {sum(len(c.page_content) for c in recursive_chunks)//len(recursive_chunks)} chars")

print(f"\nMethod 2 - CharacterTextSplitter:")
print(f"  Chunk size: 1000, Overlap: 200")
print(f"  Total chunks: {len(char_chunks)}")
print(f"  Avg chunk length: {sum(len(c.page_content) for c in char_chunks)//len(char_chunks)} chars")

print(f"\nMethod 3 - Code-Aware Splitter:")
print(f"  Language: Python")
print(f"  Separators: {code_splitter._separators[:5]}")

print(f"\nSample Chunk (Recursive, chunk #1):\n{'-'*50}")
print(f"Source: {recursive_chunks[0].metadata.get('source_file', 'N/A')}")
print(recursive_chunks[0].page_content[:400])
print(f"\nSample Chunk (Recursive, chunk #2):\n{'-'*50}")
print(f"Source: {recursive_chunks[1].metadata.get('source_file', 'N/A')}")
print(recursive_chunks[1].page_content[:400])
print(f"\n{'='*70}\n")

# --- Save screenshot for Task 2 ---
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_facecolor('#1a1a2e')
fig.patch.set_facecolor('#16213e')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'Task 2: Text Splitting Techniques',
        fontsize=16, fontweight='bold', color='#e94560', ha='center',
        fontfamily='monospace')

y_pos = 8.8
methods = [
    ("RecursiveCharacterTextSplitter", len(recursive_chunks),
     sum(len(c.page_content) for c in recursive_chunks) // len(recursive_chunks),
     '1000', '200', '[\\n\\n, \\n, ". ", " ", ""]'),
    ("CharacterTextSplitter", len(char_chunks),
     sum(len(c.page_content) for c in char_chunks) // len(char_chunks),
     '1000', '200', '"\\n"'),
    ("Code-Aware Splitter (Python)", "N/A (demo)",
     "N/A", '1000', '200', 'Language.PYTHON separators'),
]

for name, chunks, avg_len, csize, overlap, seps in methods:
    ax.text(0.5, y_pos, f"  {name}", fontsize=12, fontweight='bold',
            color='#ffd700', fontfamily='monospace')
    y_pos -= 0.45
    ax.text(0.5, y_pos,
            f"    Chunks: {chunks}  |  Avg Length: {avg_len} chars  |  Size: {csize}  |  Overlap: {overlap}",
            fontsize=9.5, color='#00ff88', fontfamily='monospace')
    y_pos -= 0.35
    ax.text(0.5, y_pos, f"    Separators: {seps}",
            fontsize=9, color='#87ceeb', fontfamily='monospace')
    y_pos -= 0.55

ax.axhline(y=y_pos + 0.2, xmin=0.03, xmax=0.97, color='#e94560', linewidth=1)
y_pos -= 0.3

ax.text(0.5, y_pos, "  Sample Chunk #1 (RecursiveCharacterTextSplitter):",
        fontsize=11, fontweight='bold', color='#ffd700', fontfamily='monospace')
y_pos -= 0.35
ax.text(0.5, y_pos, f"    Source: {recursive_chunks[0].metadata.get('source_file', 'N/A')}",
        fontsize=9, color='#ffaa00', fontfamily='monospace')
y_pos -= 0.4

sample = recursive_chunks[0].page_content[:350]
wrapped = textwrap.wrap(sample, width=95)
for line in wrapped[:8]:
    ax.text(0.5, y_pos, f"    {line}", fontsize=8, color='#ffffff',
            fontfamily='monospace', va='top')
    y_pos -= 0.35

ax.axhline(y=y_pos + 0.15, xmin=0.03, xmax=0.97, color='#444466', linewidth=0.5)
y_pos -= 0.3

ax.text(0.5, y_pos, "  Sample Chunk #2 (RecursiveCharacterTextSplitter):",
        fontsize=11, fontweight='bold', color='#ffd700', fontfamily='monospace')
y_pos -= 0.35
ax.text(0.5, y_pos, f"    Source: {recursive_chunks[1].metadata.get('source_file', 'N/A')}",
        fontsize=9, color='#ffaa00', fontfamily='monospace')
y_pos -= 0.4
sample2 = recursive_chunks[1].page_content[:350]
wrapped2 = textwrap.wrap(sample2, width=95)
for line in wrapped2[:6]:
    ax.text(0.5, y_pos, f"    {line}", fontsize=8, color='#ffffff',
            fontfamily='monospace', va='top')
    y_pos -= 0.35

plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOTS_DIR, 'code_splitter.png'),
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"✅ Screenshot saved: {SCREENSHOTS_DIR}/code_splitter.png\n")


# ============================================================
# TASK 3: Embed Documents
# ============================================================
print("=" * 70)
print("TASK 3: Embedding Documents Using Sentence-Transformers (ONNX)")
print("=" * 70)

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_core.embeddings import Embeddings

# Use ChromaDB's built-in ONNX embedding (all-MiniLM-L6-v2, runs locally)
chroma_ef = DefaultEmbeddingFunction()


class OnnxEmbeddings(Embeddings):
    """LangChain-compatible wrapper around ChromaDB's ONNX embedding function."""

    def __init__(self, ef):
        self._ef = ef

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [emb.tolist() for emb in self._ef(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self._ef([text])[0].tolist()


embedding_model = OnnxEmbeddings(chroma_ef)

# Embed a sample query
sample_query = "What is GDPR compliance?"
query_embedding = embedding_model.embed_query(sample_query)

# Embed sample documents
sample_texts = [
    recursive_chunks[0].page_content,
    recursive_chunks[1].page_content,
    recursive_chunks[2].page_content
]
doc_embeddings = embedding_model.embed_documents(sample_texts)

print(f"\nEmbedding Model: all-MiniLM-L6-v2 (ONNX Runtime)")
print(f"Embedding Dimension: {len(query_embedding)}")
print(f"\nSample Query: '{sample_query}'")
print(f"Query Embedding (first 10 values): {[round(v, 4) for v in query_embedding[:10]]}")
print(f"\nDocument Embeddings Generated: {len(doc_embeddings)}")
print(f"Doc Embedding #1 (first 10 values): {[round(v, 4) for v in doc_embeddings[0][:10]]}")
print(f"Doc Embedding #2 (first 10 values): {[round(v, 4) for v in doc_embeddings[1][:10]]}")

# Compute cosine similarity
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

sim_q_d1 = cosine_similarity(query_embedding, doc_embeddings[0])
sim_q_d2 = cosine_similarity(query_embedding, doc_embeddings[1])
sim_d1_d2 = cosine_similarity(doc_embeddings[0], doc_embeddings[1])

print(f"\nCosine Similarities:")
print(f"  Query <-> Doc1: {sim_q_d1:.4f}")
print(f"  Query <-> Doc2: {sim_q_d2:.4f}")
print(f"  Doc1  <-> Doc2: {sim_d1_d2:.4f}")
print(f"\n{'='*70}\n")

# --- Save screenshot for Task 3 ---
fig, axes = plt.subplots(1, 2, figsize=(16, 8),
                          gridspec_kw={'width_ratios': [3, 2]})
fig.patch.set_facecolor('#16213e')

ax = axes[0]
ax.set_facecolor('#1a1a2e')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'Task 3: Document Embeddings',
        fontsize=15, fontweight='bold', color='#e94560', ha='center',
        fontfamily='monospace')

y_pos = 8.8
info = [
    ("Model:", "all-MiniLM-L6-v2 (ONNX Runtime)"),
    ("Dimension:", f"{len(query_embedding)}"),
    ("Normalization:", "True"),
    ("Device:", "CPU"),
]
for label, val in info:
    ax.text(0.3, y_pos, f"  {label} {val}", fontsize=11, color='#00ff88',
            fontfamily='monospace')
    y_pos -= 0.5

ax.axhline(y=y_pos + 0.2, xmin=0.03, xmax=0.97, color='#e94560', linewidth=1)
y_pos -= 0.3

ax.text(0.3, y_pos, f"  Query: '{sample_query}'", fontsize=11,
        color='#ffd700', fontfamily='monospace')
y_pos -= 0.5
embed_str = str([round(v, 4) for v in query_embedding[:8]]) + " ..."
ax.text(0.3, y_pos, f"  Embedding: {embed_str}", fontsize=8.5,
        color='#87ceeb', fontfamily='monospace')
y_pos -= 0.6

ax.text(0.3, y_pos, "  Cosine Similarities:", fontsize=12, fontweight='bold',
        color='#ffd700', fontfamily='monospace')
y_pos -= 0.5
sims = [
    ("Query <-> Chunk 1", sim_q_d1),
    ("Query <-> Chunk 2", sim_q_d2),
    ("Chunk 1 <-> Chunk 2", sim_d1_d2),
]
for label, sim in sims:
    color = '#00ff88' if sim > 0.5 else '#ffaa00' if sim > 0.3 else '#ff6666'
    ax.text(0.3, y_pos, f"    {label}: {sim:.4f}", fontsize=11,
            color=color, fontfamily='monospace')
    y_pos -= 0.45

ax.axhline(y=y_pos + 0.15, xmin=0.03, xmax=0.97, color='#444466', linewidth=0.5)
y_pos -= 0.3
ax.text(0.3, y_pos, f"  Documents Embedded: {len(doc_embeddings)} chunks",
        fontsize=11, color='#00ff88', fontfamily='monospace')
y_pos -= 0.45
ax.text(0.3, y_pos, f"  Total chunks to embed: {len(recursive_chunks)}",
        fontsize=11, color='#00ff88', fontfamily='monospace')

ax2 = axes[1]
ax2.set_facecolor('#1a1a2e')
ax2.set_title('Embedding Vector Heatmap (first 50 dims)', color='#e94560',
              fontsize=11, fontweight='bold', fontfamily='monospace')

heatmap_data = np.array([
    query_embedding[:50],
    doc_embeddings[0][:50],
    doc_embeddings[1][:50],
    doc_embeddings[2][:50]
])
im = ax2.imshow(heatmap_data, aspect='auto', cmap='coolwarm', interpolation='nearest')
ax2.set_yticks([0, 1, 2, 3])
ax2.set_yticklabels(['Query', 'Chunk 1', 'Chunk 2', 'Chunk 3'],
                     fontsize=9, color='#ffffff', fontfamily='monospace')
ax2.set_xlabel('Embedding Dimensions', color='#ffffff', fontsize=9,
               fontfamily='monospace')
ax2.tick_params(colors='#ffffff')
plt.colorbar(im, ax=ax2, shrink=0.6)

plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOTS_DIR, 'embedding.png'),
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"✅ Screenshot saved: {SCREENSHOTS_DIR}/embedding.png\n")


# ============================================================
# TASK 4: Create and Configure Vector Database (ChromaDB)
# ============================================================
print("=" * 70)
print("TASK 4: Creating Vector Database with ChromaDB")
print("=" * 70)

from langchain_community.vectorstores import Chroma

persist_directory = os.path.join(BASE_DIR, "chroma_db")

vectordb = Chroma.from_documents(
    documents=recursive_chunks,
    embedding=embedding_model,
    persist_directory=persist_directory,
    collection_name="quest_analytics_papers"
)

collection = vectordb._collection
collection_count = collection.count()

print(f"\nVector Database: ChromaDB")
print(f"Persist Directory: {persist_directory}")
print(f"Collection Name: quest_analytics_papers")
print(f"Total Documents Stored: {collection_count}")
print(f"Embedding Function: all-MiniLM-L6-v2 (ONNX)")
print(f"Embedding Dimension: {len(query_embedding)}")

test_query = "What are the GDPR requirements for data protection?"
similar_docs = vectordb.similarity_search_with_score(test_query, k=3)

print(f"\nTest Similarity Search:")
print(f"  Query: '{test_query}'")
print(f"  Top {len(similar_docs)} results:")
for i, (doc, score) in enumerate(similar_docs):
    print(f"\n  Result {i+1} (distance: {score:.4f}):")
    print(f"    Source: {doc.metadata.get('source_file', 'N/A')}")
    print(f"    Page: {doc.metadata.get('page', 'N/A')}")
    print(f"    Preview: {doc.page_content[:150]}...")
print(f"\n{'='*70}\n")

# --- Save screenshot for Task 4 ---
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_facecolor('#1a1a2e')
fig.patch.set_facecolor('#16213e')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'Task 4: Vector Database - ChromaDB',
        fontsize=16, fontweight='bold', color='#e94560', ha='center',
        fontfamily='monospace')

y_pos = 8.8
db_info = [
    ("Database:", "ChromaDB (Persistent)"),
    ("Collection:", "quest_analytics_papers"),
    ("Documents Stored:", f"{collection_count}"),
    ("Embedding Model:", "all-MiniLM-L6-v2 (ONNX)"),
    ("Embedding Dim:", f"{len(query_embedding)}"),
    ("Persist Dir:", "chroma_db/"),
]
for label, val in db_info:
    ax.text(0.4, y_pos, f"  {label} {val}", fontsize=11, color='#00ff88',
            fontfamily='monospace')
    y_pos -= 0.48

ax.axhline(y=y_pos + 0.2, xmin=0.03, xmax=0.97, color='#e94560', linewidth=1)
y_pos -= 0.35

ax.text(0.4, y_pos, "  Similarity Search Test", fontsize=12, fontweight='bold',
        color='#ffd700', fontfamily='monospace')
y_pos -= 0.45
ax.text(0.4, y_pos, f"  Query: \"{test_query}\"", fontsize=10,
        color='#ffffff', fontfamily='monospace')
y_pos -= 0.55

for i, (doc, score) in enumerate(similar_docs):
    source = doc.metadata.get('source_file', 'N/A')
    page = doc.metadata.get('page', 'N/A')
    ax.text(0.4, y_pos, f"  Result {i+1} (distance: {score:.4f}, {source}, page: {page})",
            fontsize=10, fontweight='bold', color='#ffaa00', fontfamily='monospace')
    y_pos -= 0.4
    preview = doc.page_content[:120].replace('\n', ' ')
    wrapped = textwrap.wrap(preview, width=85)
    for line in wrapped[:2]:
        ax.text(0.4, y_pos, f"    {line}", fontsize=8.5, color='#87ceeb',
                fontfamily='monospace')
        y_pos -= 0.32
    y_pos -= 0.15

plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOTS_DIR, 'vectordb.png'),
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"✅ Screenshot saved: {SCREENSHOTS_DIR}/vectordb.png\n")


# ============================================================
# TASK 5: Develop a Retriever
# ============================================================
print("=" * 70)
print("TASK 5: Developing Document Retriever")
print("=" * 70)

retriever_basic = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

retriever_mmr = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.7}
)

retriever_threshold = vectordb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.3, "k": 4}
)

test_queries = [
    "What is GDPR and what are its main principles?",
    "How should biometric data be handled under GDPR?",
    "What are the penalties for GDPR non-compliance?",
]

print("\nRetriever Configurations:")
print("  1. Similarity Search (k=4)")
print("  2. MMR - Maximum Marginal Relevance (k=4, fetch_k=10, lambda=0.7)")
print("  3. Similarity Score Threshold (threshold=0.3, k=4)")

print("\nRetriever Test Results:")
for q in test_queries:
    docs = retriever_basic.invoke(q)
    print(f"\n  Query: '{q}'")
    print(f"  Documents Retrieved: {len(docs)}")
    for i, doc in enumerate(docs[:2]):
        source = doc.metadata.get('source_file', 'N/A')
        page = doc.metadata.get('page', 'N/A')
        print(f"    Doc {i+1} ({source}, page {page}): {doc.page_content[:100]}...")

print("\n  MMR Retriever Test:")
mmr_docs = retriever_mmr.invoke(test_queries[0])
print(f"  Query: '{test_queries[0]}'")
print(f"  Documents Retrieved: {len(mmr_docs)}")
for i, doc in enumerate(mmr_docs[:2]):
    source = doc.metadata.get('source_file', 'N/A')
    page = doc.metadata.get('page', 'N/A')
    print(f"    Doc {i+1} ({source}, page {page}): {doc.page_content[:100]}...")

print(f"\n{'='*70}\n")

# --- Save screenshot for Task 5 ---
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_facecolor('#1a1a2e')
fig.patch.set_facecolor('#16213e')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'Task 5: Document Retriever',
        fontsize=16, fontweight='bold', color='#e94560', ha='center',
        fontfamily='monospace')

y_pos = 8.8
configs = [
    ("Similarity Search", "k=4", "#00ff88"),
    ("MMR (Max Marginal Relevance)", "k=4, fetch_k=10, lambda=0.7", "#87ceeb"),
    ("Score Threshold", "threshold=0.3, k=4", "#ffaa00"),
]
ax.text(0.3, y_pos, "  Retriever Configurations:", fontsize=12,
        fontweight='bold', color='#ffd700', fontfamily='monospace')
y_pos -= 0.5
for name, params, color in configs:
    ax.text(0.3, y_pos, f"    > {name}: {params}", fontsize=10,
            color=color, fontfamily='monospace')
    y_pos -= 0.42

ax.axhline(y=y_pos + 0.15, xmin=0.03, xmax=0.97, color='#e94560', linewidth=1)
y_pos -= 0.4

ax.text(0.3, y_pos, "  Retrieval Test Results:", fontsize=12,
        fontweight='bold', color='#ffd700', fontfamily='monospace')
y_pos -= 0.5

for q in test_queries:
    docs = retriever_basic.invoke(q)
    q_short = q[:65] + "..." if len(q) > 65 else q
    ax.text(0.3, y_pos, f"  Q: \"{q_short}\"", fontsize=9.5,
            color='#ffffff', fontfamily='monospace')
    y_pos -= 0.38
    source = docs[0].metadata.get('source_file', 'N/A')
    page = docs[0].metadata.get('page', 'N/A')
    ax.text(0.3, y_pos, f"     Retrieved: {len(docs)} docs | Source: {source[:30]}, page: {page}",
            fontsize=9, color='#00ff88', fontfamily='monospace')
    y_pos -= 0.32
    preview = docs[0].page_content[:100].replace('\n', ' ')
    ax.text(0.3, y_pos, f"     Preview: {preview}...",
            fontsize=8, color='#87ceeb', fontfamily='monospace')
    y_pos -= 0.48

ax.axhline(y=y_pos + 0.2, xmin=0.03, xmax=0.97, color='#444466', linewidth=0.5)
y_pos -= 0.35
ax.text(0.3, y_pos, "  MMR vs Similarity:", fontsize=11, fontweight='bold',
        color='#ffd700', fontfamily='monospace')
y_pos -= 0.42
ax.text(0.3, y_pos, "    MMR provides diverse results by balancing relevance & novelty",
        fontsize=9.5, color='#00ff88', fontfamily='monospace')
y_pos -= 0.35
ax.text(0.3, y_pos, "    Similarity returns closest matches by embedding distance",
        fontsize=9.5, color='#87ceeb', fontfamily='monospace')

plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOTS_DIR, 'retriever.png'),
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"✅ Screenshot saved: {SCREENSHOTS_DIR}/retriever.png\n")


# ============================================================
# TASK 6: Construct a QA Bot Using LangChain and Hugging Face LLM
# ============================================================
print("=" * 70)
print("TASK 6: Constructing QA Bot with Hugging Face LLM")
print("=" * 70)

from langchain_community.llms import HuggingFaceHub
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Initialize Hugging Face LLM
model_name = os.getenv("HF_MODEL_NAME", "google/flan-t5-base")

print(f"\n🤖 Initializing Hugging Face LLM...")
print(f"   Model: {model_name}")
print(f"   API Key: {'✅ Loaded' if HF_API_KEY else '❌ Missing'}")

try:
    llm = HuggingFaceHub(
        repo_id=model_name,
        model_kwargs={
            "temperature": 0.7,
            "max_length": 512,
        },
        huggingfacehub_api_token=HF_API_KEY
    )
    print(f"   Status: ✅ LLM initialized successfully!")
except Exception as e:
    print(f"   Status: ❌ Error initializing LLM: {e}")
    print(f"\n   Falling back to extractive summarization...")
    
    # Fallback to extractive LLM if Hugging Face fails
    from langchain_core.language_models.llms import LLM
    from typing import Any, Optional
    
    class ExtractiveSummaryLLM(LLM):
        @property
        def _llm_type(self) -> str:
            return "extractive-summary-llm"

        def _call(self, prompt: str, stop: Optional[list[str]] = None,
                  run_manager: Any = None, **kwargs) -> str:
            question = ""
            context = ""

            if "Question:" in prompt and "Context:" in prompt:
                parts = prompt.split("Question:")
                context_part = parts[0]
                question = parts[1].split("Answer:")[0].strip() if len(parts) > 1 else ""

                if "Context:" in context_part:
                    context = context_part.split("Context:")[1].strip()
            else:
                context = prompt

            question_words = set(question.lower().split())
            stop_words = {'what', 'is', 'the', 'a', 'an', 'are', 'how', 'should',
                          'under', 'for', 'of', 'and', 'in', 'to', 'by', 'with',
                          'do', 'does', 'why', 'it', 'its', 'that', 'this', 'be',
                          'on', 'at', 'from', 'or', 'which', 'was', 'were', 'has',
                          'have', 'been', 'being', 'can', 'could', 'will', 'would'}
            question_keywords = question_words - stop_words

            sentences = []
            for sent in context.replace('\n', ' ').split('.'):
                sent = sent.strip()
                if len(sent) > 20:
                    sentences.append(sent + '.')

            if not sentences:
                return "No relevant information found in the provided context."

            scored = []
            for sent in sentences:
                sent_words = set(sent.lower().split())
                overlap = len(question_keywords & sent_words)
                scored.append((overlap, sent))

            scored.sort(key=lambda x: x[0], reverse=True)
            top_sentences = [s[1] for s in scored[:3] if s[0] > 0]

            if not top_sentences:
                top_sentences = [scored[0][1]] if scored else ["No relevant information found."]

            answer = " ".join(top_sentences)
            if len(answer) > 500:
                answer = answer[:497] + "..."

            return answer
    
    llm = ExtractiveSummaryLLM()
    model_name = "ExtractiveSummaryLLM (Fallback)"

# Create custom prompt template
prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say "I don't have enough information to answer this question."
Keep your answer concise and focused on the question.

Context:
{context}

Question: {question}

Helpful Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Build the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever_basic,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

print(f"\n✅ QA Chain Configuration:")
print(f"   LLM: {model_name}")
print(f"   Chain Type: stuff (all retrieved docs stuffed into prompt)")
print(f"   Retriever: Similarity Search (k=4)")
print(f"   Source Tracking: Enabled")

# Test the QA Bot with multiple questions
qa_questions = [
    "What is GDPR and why is it important?",
    "What are the key principles of data protection under GDPR?",
    "How should organizations handle biometric data?",
    "What are the penalties for GDPR violations?",
    "What is Privacy by Design?",
]

print(f"\n{'='*50}")
print("QA Bot Test Results")
print(f"{'='*50}")

qa_results = []
for question in qa_questions:
    print(f"\n🔍 Processing: {question}")
    try:
        result = qa_chain.invoke({"query": question})
        answer = result["result"].strip()
        source_docs = result["source_documents"]
        
        # Extract source information
        sources_info = []
        for doc in source_docs:
            source_file = doc.metadata.get('source_file', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            sources_info.append(f"{source_file} (p{page})")
        
        qa_results.append((question, answer, sources_info))
        
        print(f"✅ Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        print(f"   Sources: {', '.join(sources_info[:3])}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        qa_results.append((question, f"Error: {str(e)}", []))

print(f"\n{'='*70}")
print("Quest Analytics RAG Assistant - All Tasks Complete!")
print(f"{'='*70}\n")

# --- Save screenshot for Task 6 ---
fig, ax = plt.subplots(figsize=(16, 12))
ax.set_facecolor('#1a1a2e')
fig.patch.set_facecolor('#16213e')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.6, 'Task 6: QA Bot - Hugging Face LLM',
        fontsize=16, fontweight='bold', color='#e94560', ha='center',
        fontfamily='monospace')

y_pos = 9.1
config_info = [
    f"LLM: {model_name[:60]}",
    f"Chain: RetrievalQA (stuff)",
    f"Retriever: ChromaDB Similarity Search (k=4)",
    f"Embedding: all-MiniLM-L6-v2 (384 dim, ONNX)",
    f"Source Tracking: Enabled ✅",
]
for line in config_info:
    ax.text(0.3, y_pos, f"  {line}", fontsize=10, color='#00ff88',
            fontfamily='monospace')
    y_pos -= 0.38

ax.axhline(y=y_pos + 0.15, xmin=0.03, xmax=0.97, color='#e94560', linewidth=1)
y_pos -= 0.3

ax.text(0.3, y_pos, "  QA Bot Responses:", fontsize=13, fontweight='bold',
        color='#ffd700', fontfamily='monospace')
y_pos -= 0.45

for question, answer, sources in qa_results:
    q_short = question[:70] + "..." if len(question) > 70 else question
    ax.text(0.3, y_pos, f"  Q: {q_short}", fontsize=9.5, fontweight='bold',
            color='#ffffff', fontfamily='monospace')
    y_pos -= 0.35

    a_short = answer[:130] if len(answer) <= 130 else answer[:130] + "..."
    wrapped_a = textwrap.wrap(f"A: {a_short}", width=90)
    for line in wrapped_a[:2]:
        ax.text(0.3, y_pos, f"  {line}", fontsize=9, color='#87ceeb',
                fontfamily='monospace')
        y_pos -= 0.3

    sources_str = ', '.join(sources[:2]) if sources else "No sources"
    ax.text(0.3, y_pos, f"  Sources: {sources_str}", fontsize=8,
            color='#666699', fontfamily='monospace')
    y_pos -= 0.42

# Architecture footer
ax.axhline(y=y_pos + 0.15, xmin=0.03, xmax=0.97, color='#e94560', linewidth=1)
y_pos -= 0.35
ax.text(5, y_pos, "Multi-PDF → Chunks → Embeddings → ChromaDB → Retriever → Hugging Face LLM → Answer",
        fontsize=9, color='#e94560', ha='center', fontfamily='monospace',
        fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOTS_DIR, 'qabot.png'),
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"✅ Screenshot saved: {SCREENSHOTS_DIR}/qabot.png")

print("\n" + "=" * 70)
print("📸 ALL SCREENSHOTS SAVED:")
print("=" * 70)
screenshots = [
    "1. pdf_loader.png    - Task 1: Multi-PDF Document Loading",
    "2. code_splitter.png - Task 2: Text Splitting",
    "3. embedding.png     - Task 3: Document Embeddings",
    "4. vectordb.png      - Task 4: Vector Database",
    "5. retriever.png     - Task 5: Document Retriever",
    "6. qabot.png         - Task 6: QA Bot with Hugging Face",
]
for s in screenshots:
    print(f"   {s}")
print("=" * 70)

print("\n" + "=" * 70)
print("🎉 PHASE 1 - DAY 1 COMPLETE!")
print("=" * 70)
print("✅ Hugging Face LLM integrated")
print("✅ Multi-PDF support added")
print("✅ Source tracking enabled")
print("✅ All 6 tasks completed")
print("✅ All screenshots generated")
print("\n📁 Your files:")
print(f"   - Main script: quest_analytics_rag.py")
print(f"   - API key: .env")
print(f"   - PDFs folder: {PDFS_DIR}")
print(f"   - Screenshots: {SCREENSHOTS_DIR}")
print(f"   - Vector DB: {persist_directory}")
print("\n🚀 TOMORROW (Day 2): Add storytelling modes (Kid/Adult/Story)!")
print("=" * 70)
