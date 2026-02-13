# Quest Analytics RAG Assistant - Phase 1 Setup Guide

## 🚀 Welcome, Broski!

You've just upgraded your RAG system with:
- ✅ Hugging Face LLM integration (FLAN-T5)
- ✅ Multi-PDF support
- ✅ Secure API key management
- ✅ Better source tracking
- ✅ All 6 original tasks preserved!

---

## 📋 Prerequisites

Make sure you have Python 3.8+ installed.

---

## 🔧 Installation Steps

### Step 1: Install Required Packages

Run this command to install all dependencies:

```bash
pip install langchain langchain-community langchain-core chromadb sentence-transformers python-dotenv huggingface-hub matplotlib numpy
```

OR use this shorter version:

```bash
pip install langchain langchain-community langchain-core chromadb sentence-transformers python-dotenv huggingface-hub matplotlib numpy --break-system-packages
```

Note: The `--break-system-packages` flag might be needed on some systems.

---

### Step 2: Set Up Your PDFs Folder

**Option A: Use the pdfs/ folder (Recommended)**

1. Create a folder called `pdfs` in your project directory
2. Move your `GDPR-Framework.pdf` into the `pdfs/` folder
3. Add any other research PDFs you want to process

```
your_project/
├── quest_analytics_rag.py
├── .env
└── pdfs/
    ├── GDPR-Framework.pdf
    ├── research_paper_2.pdf
    └── research_paper_3.pdf
```

**Option B: Keep GDPR-Framework.pdf in current directory**

The script will automatically detect it if it's in the same folder as the script.

---

### Step 3: Verify Your .env File

Make sure your `.env` file contains:

```
HUGGINGFACE_API_KEY=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
HF_MODEL_NAME=google/flan-t5-base
```

This is already created for you! ✅

---

## ▶️ Running the Script

Simply run:

```bash
python quest_analytics_rag.py
```

OR:

```bash
python3 quest_analytics_rag.py
```

---

## 📸 What to Expect

The script will:

1. ✅ Load all PDFs from the `pdfs/` folder (or fallback to GDPR-Framework.pdf)
2. ✅ Split documents into chunks
3. ✅ Create embeddings
4. ✅ Build ChromaDB vector database
5. ✅ Set up retriever
6. ✅ Initialize Hugging Face LLM (FLAN-T5)
7. ✅ Test QA Bot with sample questions
8. ✅ Generate 6 screenshots in `screenshots/` folder

**Total runtime: ~2-5 minutes** (depending on number of PDFs)

---

## 📁 Output Files

After running, you'll have:

```
your_project/
├── quest_analytics_rag.py    (Main script)
├── .env                       (API key - keep secret!)
├── pdfs/                      (Your research papers)
│   └── GDPR-Framework.pdf
├── screenshots/               (Generated visualizations)
│   ├── pdf_loader.png
│   ├── code_splitter.png
│   ├── embedding.png
│   ├── vectordb.png
│   ├── retriever.png
│   └── qabot.png
└── chroma_db/                 (Vector database storage)
```

---

## 🎯 Testing Your Setup

After running the script, check:

1. ✅ All 6 screenshots are in `screenshots/` folder
2. ✅ Terminal shows "PHASE 1 - DAY 1 COMPLETE!"
3. ✅ QA Bot answered 5 test questions
4. ✅ Source tracking shows which PDF/page answered each question

---

## 🐛 Troubleshooting

### Issue: "HUGGINGFACE_API_KEY not found"
**Solution:** Make sure `.env` file is in the same directory as the script

### Issue: "No PDF files found"
**Solution:** 
- Create `pdfs/` folder
- Move GDPR-Framework.pdf (or other PDFs) into it
- OR keep GDPR-Framework.pdf in the main directory

### Issue: "ModuleNotFoundError"
**Solution:** Install missing package:
```bash
pip install [package_name] --break-system-packages
```

### Issue: Hugging Face API timeout
**Solution:** 
- The script has a fallback extractive LLM
- It will automatically use fallback if Hugging Face fails
- Check your internet connection
- Verify API key is correct

---

## 🎨 What Changed from Original Code?

### NEW Features:
1. ✅ **Multi-PDF Support** - Load multiple research papers at once
2. ✅ **Hugging Face LLM** - Real AI model instead of extractive summarization
3. ✅ **Source Tracking** - Know which PDF answered each question
4. ✅ **Organized Folders** - Clean file structure (pdfs/, screenshots/)
5. ✅ **Secure API Keys** - Stored in .env file, not in code
6. ✅ **Error Handling** - Fallback mechanisms if things fail

### Preserved:
- ✅ All 6 original tasks
- ✅ All 6 screenshots
- ✅ Same ChromaDB setup
- ✅ Same embedding model (ONNX)
- ✅ Same code structure

---

## 🚀 Next Steps (Tomorrow - Day 2)

Phase 2 will add:
- 🧒 **Kid Mode** - Explain research like you're 5
- 👨‍💼 **Adult Mode** - Real-life examples and applications
- 📖 **Story Mode** - Turn research into engaging narratives
- 🎭 **Mode Selector** - Toggle between explanation styles

---

## 💡 Tips

1. **Start small:** Test with 1-2 PDFs first
2. **Check screenshots:** They show you what's happening at each step
3. **Read terminal output:** It's very detailed and helpful
4. **Try different questions:** Modify the `qa_questions` list in the code

---

## 🎉 Congrats, Broski!

You now have a **production-grade RAG system** with:
- Real LLM integration
- Multi-document support
- Source attribution
- Professional screenshots

**Tomorrow we make it MAGICAL with storytelling modes!** ✨

---

## 📞 Need Help?

If you run into issues:
1. Check the terminal output for error messages
2. Verify all files are in the right locations
3. Make sure all packages are installed
4. Check that your API key is valid

**Let's build something LEGENDARY!** 🔥
