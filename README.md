# Modular RAG Pipeline

This project implements a modular Retrieval-Augmented Generation (RAG) pipeline using Python. It supports loading data from both PDF and TXT files, chunking, embedding, vector search with FAISS, and answer generation with a language model via DSPy.

## Features
- Ingests text from PDF and TXT files (easily extensible to other formats)
- Chunks text for granular retrieval
- Embeds text using HuggingFace Transformers
- Fast similarity search with FAISS
- Modular design for easy extension
- Uses DSPy for LLM orchestration

## Setup

1. **Clone the repository and navigate to the project directory.**

2. **Create the conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate rag-pipeline
   ```

3. **(Optional) If you have a GPU, replace `cpuonly` with the appropriate `cudatoolkit` version in `environment.yml` and reinstall.**

4. **Add your HuggingFace token:**
   - Edit `main.py` and replace `'YOUR_HF_TOKEN'` with your actual token.

5. **Add your data:**
   - Place your `.pdf` and/or `.txt` files in the project directory.
   - Update the `sources` list in `main.py` to include your files.

## Usage

Run the main script:
```bash
python main.py
```

## Project Structure

- `data_loader.py` — Load text from PDF and TXT files
- `chunker.py` — Chunk text with overlap
- `embedder.py` — Load embedding model and generate embeddings
- `faiss_index.py` — Build and use a FAISS index
- `retriever.py` — Retrieve top-k similar chunks for a query
- `rag_module.py` — DSPy RAG module definition
- `main.py` — Entry point, wiring all modules together
- `environment.yml` — Conda environment specification

## Extending
To add new data sources, extend `data_loader.py` with new loader functions.

## Notes
- This pipeline is for prototyping and educational purposes. For production, consider improvements such as better chunking, metadata tracking, and evaluation.
- Requires a HuggingFace account and token for LLM access.

---

Feel free to open issues or contribute improvements! 

---

## **How to Use Persistent RAG Across Sessions**

### **Session 1: Build Once (Slow)**
```python
# First time setup - this takes time
rag = PersistentRAGManager()
sources = ["your_document1.pdf", "your_document2.txt"]
rag.build_pipeline(sources, "my_corpus_name", hf_token="your_token")
rag.setup_dspy(hf_token="your_token")

# Test it
result = rag.ask("What is this about?")
print(result['answer'])
```

**What happens:**
- Loads your documents
- Chunks them
- Generates embeddings
- Builds FAISS index
- **Saves everything to `rag_cache/` directory**

---

### **Session 2, 3, 4...: Use Anytime (Fast)**
```python
# In any new session - this is fast!
rag = PersistentRAGManager()
rag.build_pipeline([], "my_corpus_name", hf_token="your_token")  # Loads from cache
rag.setup_dspy(hf_token="your_token")

# Ask unlimited questions
result1 = rag.ask("Question 1?")
result2 = rag.ask("Question 2?")
# ... as many as you want
```

**What happens:**
- Loads embeddings from disk
- Loads FAISS index from disk
- Loads documents from disk
- **No re-processing needed!**

---

## **Key Benefits**

1. **Build Once, Use Forever**: Process your documents once, ask questions anytime
2. **Fast Startup**: Subsequent sessions load in seconds, not minutes
3. **Multiple Corpora**: You can have different RAG systems for different topics
4. **Persistent Storage**: Your embeddings and index are saved between sessions
5. **Memory Efficient**: Only loads what you need

---

## **Workflow Example**

```bash
# Session 1: Build your RAG
python session_example.py
# Edit the file to uncomment session_1_build_pipeline()

# Session 2: Use your RAG (new terminal)
python session_example.py  
# Edit the file to uncomment session_2_use_pipeline()

# Session 3: Interactive questions (new terminal)
python session_example.py
# Edit the file to uncomment session_3_quick_questions()
```

---

## **What Gets Saved**

The system saves these files in `rag_cache/`:
- `my_corpus_name_documents.pkl` - Your chunked documents
- `my_corpus_name_embeddings.npy` - Document embeddings
- `my_corpus_name_index.faiss` - FAISS search index
- `my_corpus_name_metadata.pkl` - Source info and metadata

---

## **Multiple Corpora**

You can have different RAG systems for different topics:

```python
# Philosophy RAG
rag1 = PersistentRAGManager()
rag1.build_pipeline(philosophy_files, "philosophy", hf_token=token)

# Science RAG  
rag2 = PersistentRAGManager()
rag2.build_pipeline(science_files, "science", hf_token=token)

# Use either one anytime
rag1.ask("What is existentialism?")
rag2.ask("What is quantum physics?")
```

This gives you a **persistent, reusable RAG system** that you can build once and use across multiple sessions! 

---

## **How to Use Your RAG Pipeline**

### **Option 1: Interactive Mode (Updated main.py)**
```bash
python main.py
```
- Loads your documents once
- Builds the index once  
- Then you can ask unlimited questions interactively
- Type `quit` to exit

### **Option 2: Programmatic Mode (rag_manager.py)**
```python
from rag_manager import RAGManager

# Setup once
sources = ["your_document1.pdf", "your_document2.txt"]
rag = RAGManager(sources, hf_token="your_token")

# Ask multiple questions
result1 = rag.ask("What is the main topic?")
result2 = rag.ask("Can you explain concept X?")
result3 = rag.ask("What are the key findings?")

print(result1['answer'])
print(result2['answer'])
print(result3['answer'])
```

### **Option 3: Batch Questions**
```python
questions = [
    "What is the main argument?",
    "What methodology was used?",
    "What are the conclusions?",
    "How does this relate to previous work?"
]

for question in questions:
    result = rag.ask(question)
    print(f"Q: {question}")
    print(f"A: {result['answer']}\n")
```

---

## **Key Benefits of This Approach**

1. **One-time Setup**: Load documents and build index only once
2. **Fast Queries**: Subsequent questions are much faster
3. **Reusable**: Same pipeline for multiple questions
4. **Flexible**: Can ask follow-up questions, related questions, etc.
5. **Memory Efficient**: Index stays in memory

---

## **Workflow for Your Use Case**

1. **Prepare your documents** (PDFs, TXTs about your topic)
2. **Update the sources list** in either script
3. **Add your HuggingFace token**
4. **Run the pipeline** (interactive or programmatic)
5. **Ask unlimited questions** about your topic!

The pipeline will remember your documents and can answer any question about the content you've loaded. 