from rag.retriever import retrieve_documents
from llm.chain import generate_answer


question = input("Ask a question about the Husky A300: ")


# Retrieve relevant documents from FAISS
results = retrieve_documents(question)
for i, result in enumerate(results):
    print(f"\n--- Retrieved Chunk {i + 1} ---")
    print(result.page_content)
    print(f"Source: {result.metadata['source']}")
    print(f"Page: {result.metadata['page_label']}")


# Combine retrieved chunks into one context
context = "\n\n".join(
    result.page_content for result in results
)


# Generate answer using Gemini
answer = generate_answer(question, context)


# Display answer
print("\n--- AI Answer ---")
print(answer)


# Display sources
print("\n--- Sources ---")

for result in results:
    print(
        f"{result.metadata['source']}, "
        f"Page {result.metadata['page_label']}"
    )