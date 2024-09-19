from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

def generate_text(model, db, prompt: str, constraints: dict):
    # Initialize the HuggingFace Hub LLM
    hf_llm = HuggingFaceHub(
        repo_id=model,  # Model ID from HuggingFace
        task="text-generation",  # Task type (text generation)
        model_kwargs={
            "max_new_tokens": 512,    # Limit tokens in output
            "top_k": 30,              # Top-k sampling
            "temperature": 0.3,       # Controls randomness (lower = less random)
            "repetition_penalty": 1.2, # Penalizes repetition
        },
    )

    # Create a retriever with the specified constraints
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 10,                  # Retrieve top 2 similar documents
            "filter": constraints     # Apply constraints for filtering
        }
    )

    # Manually retrieve documents before running the chain
    retrieved_docs = retriever.get_relevant_documents(prompt)
    
    # Print the retrieved documents for inspection
    print("Retrieved Documents:")
    for idx, doc in enumerate(retrieved_docs):
        print(f"Document {idx + 1}:")
        print(doc.page_content)
        print(doc.metadata)
        print("\n" + "-"*50 + "\n")

    # Modify the prompt by including document metadata in the text
    # Concatenate metadata with the document content
    modified_prompt = prompt + "\n\n Only answer based on the following documents:\n"
    for doc in retrieved_docs:
        modified_prompt += f"{doc.page_content} metadata: {doc.metadata}\n\n"

    # Generate the response using the ConversationalRetrievalChain
    chain = RetrievalQA.from_chain_type( 
        llm=hf_llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=False
    )

    # Use invoke instead of run
    response = chain.run(modified_prompt)


    return response
