def retrieval_model(query, k, tokenizer, model, index, documents, get_embeddings):
    query_embedding = get_embeddings([query], tokenizer, model)
    _, indices = index.search(query_embedding, k)
    retrieved_passages = [type('Passage', (object,), {'long_text': documents[idx]})() for idx in indices[0]]
    return retrieved_passages 