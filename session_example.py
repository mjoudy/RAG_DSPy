from rag_persistent import PersistentRAGManager

def session_1_build_pipeline():
    """First session: Build the pipeline and save it."""
    print("=== SESSION 1: Building Pipeline ===")
    
    rag = PersistentRAGManager()
    
    # Your documents
    sources = ["1.pdf", "2.pdf", "3.pdf"]
    corpus_name = "my_philosophy_corpus"
    hf_token = 'YOUR_HF_TOKEN'  # Replace with your token
    
    # Build pipeline (this will take time the first time)
    rag.build_pipeline(sources, corpus_name, hf_token=hf_token)
    
    # Setup DSPy
    rag.setup_dspy(hf_token=hf_token)
    
    # Test a few questions
    print("\nTesting the pipeline:")
    result = rag.ask("What is the main topic of these documents?")
    print(f"Answer: {result['answer']}")
    
    print("\nPipeline built and saved! You can now close this session.")
    print("The pipeline is cached and ready for future sessions.")

def session_2_use_pipeline():
    """Second session: Load and use the saved pipeline."""
    print("=== SESSION 2: Using Saved Pipeline ===")
    
    rag = PersistentRAGManager()
    
    # List available corpora
    print("Available cached corpora:", rag.list_cached_corpora())
    
    # Load the pipeline (this will be fast!)
    corpus_name = "my_philosophy_corpus"
    hf_token = 'YOUR_HF_TOKEN'  # Replace with your token
    
    # Load from cache (no need to specify sources again)
    rag.build_pipeline([], corpus_name, hf_token=hf_token)  # Sources ignored when loading from cache
    
    # Setup DSPy
    rag.setup_dspy(hf_token=hf_token)
    
    # Ask questions
    questions = [
        "What are the key concepts discussed?",
        "Can you summarize the main arguments?",
        "What methodology was used in this research?",
        "What are the conclusions?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        result = rag.ask(question)
        print(f"Answer: {result['answer']}")
        print("-" * 50)

def session_3_quick_questions():
    """Third session: Just ask quick questions."""
    print("=== SESSION 3: Quick Questions ===")
    
    rag = PersistentRAGManager()
    
    # Load and setup (very fast now)
    corpus_name = "my_philosophy_corpus"
    hf_token = 'YOUR_HF_TOKEN'
    
    rag.build_pipeline([], corpus_name, hf_token=hf_token)
    rag.setup_dspy(hf_token=hf_token)
    
    # Quick questions
    while True:
        question = input("\nAsk a question (or 'quit' to exit): ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if question:
            result = rag.ask(question)
            print(f"Answer: {result['answer']}")

if __name__ == "__main__":
    # Uncomment the session you want to run:
    
    # session_1_build_pipeline()  # Run this first to build the pipeline
    # session_2_use_pipeline()    # Run this in a new session to use the pipeline
    # session_3_quick_questions() # Run this for interactive questions
    
    print("Choose which session to run by uncommenting the appropriate line.")
    print("1. First run session_1_build_pipeline() to create the cache")
    print("2. Then run session_2_use_pipeline() or session_3_quick_questions() in new sessions") 