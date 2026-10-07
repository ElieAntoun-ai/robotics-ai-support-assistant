from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ToolMessage, SystemMessage
from app.tools.robot_tools import get_robot_status
from app.tools.documentation_tools import search_documentation


# Load environment variables
load_dotenv()


# Create Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0
)


llm_with_tools = llm.bind_tools(
    [
        get_robot_status,
        search_documentation
    ]
)


tools = {
    "get_robot_status": get_robot_status,
    "search_documentation": search_documentation
}


# Execute the tool requested by Gemini
def execute_tool(tool_call):

    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    selected_tool = tools[tool_name]

    result = selected_tool.invoke(tool_args)

    return result


def generate_tool_response(question):

    messages = [
    SystemMessage(
     content="""
You are a robotics technical support assistant.

Use the available tools to answer the user's question.

When you use the search_documentation tool:
- Answer using the retrieved documentation.
- Always include the source file and page number at the end of the answer.
- Do not invent a source or page number.

Do not invent information that is not provided by the tools.
"""
    ),
    ("user", question)
]

    # First Gemini call
    response = llm_with_tools.invoke(messages)

    # Store Gemini's response
    messages.append(response)

    # Continue as long as Gemini requests tools
    while response.tool_calls:

        # Execute every tool requested by Gemini
        for tool_call in response.tool_calls:

            tool_result = execute_tool(tool_call)

            # Convert the tool result into a ToolMessage
            tool_message = ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            )

            # Store the tool result in the conversation history
            messages.append(tool_message)

        # Send the updated conversation history back to Gemini
        response = llm_with_tools.invoke(messages)

        # Store Gemini's new response
        messages.append(response)

    # When Gemini no longer requests a tool,
    # return its final natural-language answer
    return response.text

