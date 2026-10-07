from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0
)

print("Calling Gemini...")

response = llm.invoke("Say hello")

print("Gemini responded:")
print(response.text)