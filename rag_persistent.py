import dspy
import pickle
import os
import numpy as np
from data_loader import load_corpus
from chunker import chunk_text
from embedder import load_embedder, get_embeddings
from faiss_index import build_faiss_index
from retriever import retrieval_model
from rag_module import RAG

class PersistentRAGManager:
    """Persistent RAG manager that can save/load pipeline state."""
    
    def __init__(self, cache_dir="rag_cache"):
        """
        Initialize persistent RAG manager.
        
        Args:
            cache_dir (str): Directory to store cached embeddings and index
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Components that will be loaded/saved
        self.documents = None
        self.tokenizer = None
        self.model = None
        self.index = None
        self.rag = None
        self.sources = None
        self.model_name = None
        
    def _get_cache_paths(self, corpus_name):
        """Get file paths for cached components."""
        base_path = os.path.join(self.cache_dir, corpus_name)
        return {
            'documents': f"{base_path}_documents.pkl",
            'embeddings': f"{base_path}_embeddings.npy",
            'index': f"{base_path}_index.faiss",
            'metadata': f"{base_path}_metadata.pkl"
        }
    
    def _save_pipeline(self, corpus_name, documents, embeddings, index, sources, model_name):
        """Save pipeline components to disk."""
        paths = self._get_cache_paths(corpus_name)
        
        # Save documents
        with open(paths['documents'], 'wb') as f:
            pickle.dump(documents, f)
        
        # Save embeddings
        np.save(paths['embeddings'], embeddings)
        
        # Save FAISS index
        import faiss
        faiss.write_index(index, paths['index'])
        
        # Save metadata
        metadata = {
            'sources': sources,
            'model_name': model_name,
            'num_documents': len(documents),
            'embedding_dim': embeddings.shape[1]
        }
        with open(paths['metadata'], 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Pipeline saved to {self.cache_dir}/{corpus_name}_*")
    
    def _load_pipeline(self, corpus_name):
        """Load pipeline components from disk."""
        paths = self._get_cache_paths(corpus_name)
        
        # Check if all files exist
        for path in paths.values():
            if not os.path.exists(path):
                return False
        
        try:
            # Load documents
            with open(paths['documents'], 'rb') as f:
                self.documents = pickle.load(f)
            
            # Load embeddings
            embeddings = np.load(paths['embeddings'])
            
            # Load FAISS index
            import faiss
            self.index = faiss.read_index(paths['index'])
            
            # Load metadata
            with open(paths['metadata'], 'rb') as f:
                metadata = pickle.load(f)
            
            self.sources = metadata['sources']
            self.model_name = metadata['model_name']
            
            print(f"Pipeline loaded from cache: {len(self.documents)} documents")
            return True
            
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            return False
    
    def build_pipeline(self, sources, corpus_name, model_name='sentence-transformers/bert-base-nli-mean-tokens', 
                      hf_token='YOUR_HF_TOKEN', force_rebuild=False):
        """
        Build or load RAG pipeline.
        
        Args:
            sources (list): List of file paths
            corpus_name (str): Name for this corpus (used for caching)
            model_name (str): Embedding model name
            hf_token (str): HuggingFace token
            force_rebuild (bool): Force rebuild even if cache exists
        """
        # Try to load from cache first
        if not force_rebuild and self._load_pipeline(corpus_name):
            print("Using cached pipeline")
        else:
            print("Building new pipeline...")
            self._build_new_pipeline(sources, corpus_name, model_name, hf_token)
    
    def _build_new_pipeline(self, sources, corpus_name, model_name, hf_token):
        """Build a new pipeline from scratch."""
        print("Loading data...")
        corpus = load_corpus(sources)
        self.documents = chunk_text(corpus)
        print(f"Loaded {len(self.documents)} chunks from {len(sources)} sources")
        
        print("Loading embedding model...")
        self.tokenizer, self.model = load_embedder(model_name)
        embeddings = get_embeddings(self.documents, self.tokenizer, self.model)
        
        print("Building FAISS index...")
        self.index = build_faiss_index(embeddings)
        
        # Save to cache
        self._save_pipeline(corpus_name, self.documents, embeddings, self.index, sources, model_name)
        
        self.sources = sources
        self.model_name = model_name
    
    def setup_dspy(self, hf_token='YOUR_HF_TOKEN'):
        """Setup DSPy components (needs to be done in each session)."""
        if self.index is None:
            raise ValueError("Pipeline not loaded. Call build_pipeline() first.")
        
        def dspy_retrieval_model(query, k=5):
            return retrieval_model(query, k, self.tokenizer, self.model, self.index, self.documents, get_embeddings)
        
        print("Setting up DSPy...")
        from huggingface_hub import login
        login(token=hf_token)
        llm = dspy.HFModel(model='google/gemma-2b')
        dspy.settings.configure(lm=llm, rm=dspy_retrieval_model)
        
        self.rag = RAG(num_passages=3)
        print("DSPy ready!")
    
    def ask(self, question):
        """Ask a question."""
        if self.rag is None:
            raise ValueError("DSPy not setup. Call setup_dspy() first.")
        
        try:
            response = self.rag(question)
            return {
                'answer': response.answer,
                'context': response.context,
                'num_passages': len(response.context)
            }
        except Exception as e:
            return {
                'answer': f"Error: {str(e)}",
                'context': [],
                'num_passages': 0
            }
    
    def get_info(self):
        """Get information about the loaded corpus."""
        if self.documents is None:
            return {"status": "No pipeline loaded"}
        
        return {
            'sources': self.sources,
            'num_documents': len(self.documents),
            'embedding_model': self.model_name
        }
    
    def list_cached_corpora(self):
        """List all cached corpora."""
        if not os.path.exists(self.cache_dir):
            return []
        
        corpora = set()
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('_metadata.pkl'):
                corpus_name = filename.replace('_metadata.pkl', '')
                corpora.add(corpus_name)
        
        return list(corpora)

# Example usage
if __name__ == "__main__":
    # Initialize persistent RAG manager
    rag = PersistentRAGManager()
    
    # List existing cached corpora
    print("Cached corpora:", rag.list_cached_corpora())
    
    # Build or load pipeline
    sources = ["1.pdf", "2.pdf", "3.pdf"]  # Your documents
    corpus_name = "heidegger_philosophy"  # Name for this corpus
    hf_token = 'YOUR_HF_TOKEN'  # Your HuggingFace token
    
    # This will load from cache if available, or build new if not
    rag.build_pipeline(sources, corpus_name, hf_token=hf_token)
    
    # Setup DSPy (needed in each session)
    rag.setup_dspy(hf_token=hf_token)
    
    # Ask questions
    questions = [
        "What is Martin Heidegger best known for?",
        "What are the main concepts in Being and Time?",
        "How does Heidegger define Dasein?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        result = rag.ask(question)
        print(f"Answer: {result['answer']}")
        print(f"Retrieved {result['num_passages']} passages")
        print("-" * 50) 