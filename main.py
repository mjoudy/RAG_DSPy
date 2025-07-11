import dspy
from data_loader import load_corpus
from chunker import chunk_text
from embedder import load_embedder, get_embeddings
from faiss_index import build_faiss_index
from retriever import retrieval_model
from rag_module import RAG

# 1. Load data from multiple sources (PDF, TXT, etc.)
sources = ["1.pdf", "2.pdf", "3.pdf"]  # Add .txt or other files as needed
corpus = load_corpus(sources)
documents = chunk_text(corpus)

# 2. Embedding
model_name = 'sentence-transformers/bert-base-nli-mean-tokens'
tokenizer, model = load_embedder(model_name)
doc_embeddings = get_embeddings(documents, tokenizer, model)

# 3. FAISS index
index = build_faiss_index(doc_embeddings)

# 4. Retrieval model for DSPy

def dspy_retrieval_model(query, k=5):
    return retrieval_model(query, k, tokenizer, model, index, documents, get_embeddings)

# 5. DSPy setup
from huggingface_hub import login
hf_token = 'YOUR_HF_TOKEN'  # Replace with your actual token
login(token=hf_token)
llm = dspy.HFModel(model='google/gemma-2b')
dspy.settings.configure(lm=llm, rm=dspy_retrieval_model)

# 6. RAG pipeline
rag = RAG(num_passages=3)

# Example usage (replace with your own logic as needed)
example_query = "What is Martin Heidegger best known for in philosophy?"
response = rag(example_query)
print(response.answer) 