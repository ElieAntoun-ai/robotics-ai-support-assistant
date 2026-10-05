from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Load environment variables
load_dotenv()


# Create Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)


# Create prompt template
prompt = ChatPromptTemplate.from_template("""
You are a robotics technical support assistant.

Answer the question using only the provided context.
If the answer cannot be found in the context, say "I don't know."

Context:
{context}

Question:
{question}
""")


# Convert LLM output into a normal Python string
output_parser = StrOutputParser()


# LCEL chain
chain = prompt | llm | output_parser


def generate_answer(question, context):

    answer = chain.invoke({
        "question": question,
        "context": context
    })

    return answer