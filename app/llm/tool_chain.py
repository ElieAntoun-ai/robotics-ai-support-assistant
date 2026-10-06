from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ToolMessage
from app.tools.robot_tools import get_robot_status


# Load environment variables
load_dotenv()


# Create Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)


# Give Gemini access to the robot tools
llm_with_tools = llm.bind_tools(
    [get_robot_status]
)


# Map tool names to the actual Python tools
tools = {
    "get_robot_status": get_robot_status
}


# Execute the tool requested by Gemini
def execute_tool(tool_call):

    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    selected_tool = tools[tool_name]

    result = selected_tool.invoke(tool_args)

    return result


# Ask Gemini a question and handle tool calls
def generate_tool_response(question):

    # First Gemini call:
    # Gemini decides whether a tool is needed
    response = llm_with_tools.invoke(question)

    if response.tool_calls:

        # Get the tool Gemini requested
        tool_call = response.tool_calls[0]

        # Execute the requested tool
        tool_result = execute_tool(tool_call)

        # Convert the result into a ToolMessage
        tool_message = ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call["id"]
        )

        # Send the tool result back to Gemini
        final_response = llm_with_tools.invoke(
            [response, tool_message]
        )

        return final_response.content

    # If Gemini did not request a tool,
    # return its normal response
    return response.content