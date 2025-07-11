import dspy
from data_loader import load_corpus
from chunker import chunk_text
from embedder import load_embedder, get_embeddings
from faiss_index import build_faiss_index
from retriever import retrieval_model
from rag_module import RAG

def setup_rag_pipeline(sources, model_name='sentence-transformers/bert-base-nli-mean-tokens', hf_token='YOUR_HF_TOKEN'):
    """Setup the RAG pipeline once and return the configured components."""
    
    print("Loading data...")
    corpus = load_corpus(sources)
    documents = chunk_text(corpus)
    print(f"Loaded {len(documents)} chunks from {len(sources)} sources")
    
    print("Loading embedding model...")
    tokenizer, model = load_embedder(model_name)
    doc_embeddings = get_embeddings(documents, tokenizer, model)
    
    print("Building FAISS index...")
    index = build_faiss_index(doc_embeddings)
    
    def dspy_retrieval_model(query, k=5):
        return retrieval_model(query, k, tokenizer, model, index, documents, get_embeddings)
    
    print("Setting up DSPy...")
    from huggingface_hub import login
    login(token=hf_token)
    llm = dspy.HFModel(model='google/gemma-2b')
    dspy.settings.configure(lm=llm, rm=dspy_retrieval_model)
    
    rag = RAG(num_passages=3)
    print("RAG pipeline ready!")
    
    return rag

def interactive_rag():
    """Interactive RAG session."""
    
    # Configuration
    sources = ["1.pdf", "2.pdf", "3.pdf"]  # Update with your files
    hf_token = 'YOUR_HF_TOKEN'  # Replace with your actual token
    
    # Setup pipeline once
    rag = setup_rag_pipeline(sources, hf_token=hf_token)
    
    print("\n" + "="*50)
    print("RAG Pipeline Ready! Ask questions about your documents.")
    print("Type 'quit' to exit.")
    print("="*50 + "\n")
    
    while True:
        question = input("Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
            
        try:
            print("Thinking...")
            response = rag(question)
            print(f"\nAnswer: {response.answer}")
            print(f"Retrieved {len(response.context)} passages")
            print("-" * 50)
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    interactive_rag() 