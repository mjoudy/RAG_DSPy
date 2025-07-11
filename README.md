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