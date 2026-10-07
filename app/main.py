from app.llm.tool_chain import generate_tool_response


question = input("Ask a question: ")

answer = generate_tool_response(question)

print("\n--- AI Answer ---")
print(answer)