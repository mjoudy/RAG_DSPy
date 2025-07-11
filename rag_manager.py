import dspy
from data_loader import load_corpus
from chunker import chunk_text
from embedder import load_embedder, get_embeddings
from faiss_index import build_faiss_index
from retriever import retrieval_model
from rag_module import RAG

class RAGManager:
    """Manages a RAG pipeline for a specific corpus."""
    
    def __init__(self, sources, model_name='sentence-transformers/bert-base-nli-mean-tokens', hf_token='YOUR_HF_TOKEN'):
        """
        Initialize RAG pipeline for given sources.
        
        Args:
            sources (list): List of file paths (PDF, TXT, etc.)
            model_name (str): HuggingFace model name for embeddings
            hf_token (str): HuggingFace token for LLM access
        """
        self.sources = sources
        self.model_name = model_name
        self.hf_token = hf_token
        
        # Initialize components
        self.documents = None
        self.tokenizer = None
        self.model = None
        self.index = None
        self.rag = None
        
        # Setup pipeline
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Setup the complete RAG pipeline."""
        print("Loading data...")
        corpus = load_corpus(self.sources)
        self.documents = chunk_text(corpus)
        print(f"Loaded {len(self.documents)} chunks from {len(self.sources)} sources")
        
        print("Loading embedding model...")
        self.tokenizer, self.model = load_embedder(self.model_name)
        doc_embeddings = get_embeddings(self.documents, self.tokenizer, self.model)
        
        print("Building FAISS index...")
        self.index = build_faiss_index(doc_embeddings)
        
        def dspy_retrieval_model(query, k=5):
            return retrieval_model(query, k, self.tokenizer, self.model, self.index, self.documents, get_embeddings)
        
        print("Setting up DSPy...")
        from huggingface_hub import login
        login(token=self.hf_token)
        llm = dspy.HFModel(model='google/gemma-2b')
        dspy.settings.configure(lm=llm, rm=dspy_retrieval_model)
        
        self.rag = RAG(num_passages=3)
        print("RAG pipeline ready!")
    
    def ask(self, question):
        """
        Ask a question and get an answer.
        
        Args:
            question (str): The question to ask
            
        Returns:
            dict: Contains 'answer' and 'context' (retrieved passages)
        """
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
        return {
            'sources': self.sources,
            'num_documents': len(self.documents),
            'embedding_model': self.model_name
        }

# Example usage
if __name__ == "__main__":
    # Initialize RAG for your documents
    sources = ["1.pdf", "2.pdf", "3.pdf"]  # Update with your files
    hf_token = 'YOUR_HF_TOKEN'  # Replace with your actual token
    
    rag_manager = RAGManager(sources, hf_token=hf_token)
    
    # Ask multiple questions
    questions = [
        "What is Martin Heidegger best known for?",
        "What are the main concepts in Being and Time?",
        "How does Heidegger define Dasein?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        result = rag_manager.ask(question)
        print(f"Answer: {result['answer']}")
        print(f"Retrieved {result['num_passages']} passages")
        print("-" * 50) 