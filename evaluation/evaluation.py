import json
from app.rag.retriever import retrieve_documents, vector_store
from app.llm.chain import generate_answer


with open("evaluation/questions.json", "r") as file:
    test_cases = json.load(file)
# Count how many retrieval tests pass
correct = 0

for test_case in test_cases:

    question = test_case["question"]
    expected_answer = test_case["expected_answer"]

    print("\nQuestion:", question)
    print("Expected answer:", expected_answer)


    results = retrieve_documents(question)

    context = "\n\n".join(
    result.page_content for result in results
)

    generated_answer = generate_answer(
    question,
    context
)
    print("Generated answer:", generated_answer)

    retrieved_text = " ".join(
    result.page_content for result in results
)

    normalized_retrieved_text = " ".join(
    retrieved_text.lower().split()
)

    normalized_expected_answer = " ".join(
    expected_answer.lower().split()
)

    answer_found = normalized_expected_answer in normalized_retrieved_text

      # Increase score if retrieval was successful
    if answer_found:
        correct += 1

    print("Expected answer found in retrieved chunks:", answer_found)

# Calculate retrieval accuracy
accuracy = correct / len(test_cases) * 100

print(f"\nRetrieval accuracy: {accuracy:.1f}%")


