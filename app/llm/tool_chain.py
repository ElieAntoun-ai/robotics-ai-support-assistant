from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ToolMessage
from app.tools.robot_tools import get_robot_status
from app.tools.documentation_tools import search_documentation


# Load environment variables
load_dotenv()


# Create Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
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

    # Store the full conversation history
    messages = [
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
    return response.content

