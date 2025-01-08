from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel
from typing_extensions import TypedDict

from env import set_keys
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

set_keys()

DEFAULT_THREAD_ID = "default"


class State(TypedDict):
    messages: Annotated[list, add_messages]
    ask_human: bool


class RequestAssistance(BaseModel):
    """Escalate the conversation to an expert. Use this if you are unable to assist directly or if the user requires support beyond your permissions.

    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """

    request: str


tavily_search = TavilySearchResults(max_results=2)
tools = [tavily_search]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620").bind_tools(
    tools + [RequestAssistance]
)
memory = MemorySaver()

graph_builder = StateGraph(State)


def chatbot(state: State):
    resp = llm.invoke(state["messages"])
    ask_human = False

    if resp.tool_calls and resp.tool_calls[0]["name"] == RequestAssistance.__name__:
        ask_human = True

    return {"messages": [resp], "ask_human": ask_human}


def human_node(state: State):
    new_msgs = []
    last = state["messages"][-1]

    if not isinstance(last, ToolMessage):
        new_msgs.append(
            ToolMessage(
                content="No response from human",
                tool_call_id=last.tool_calls[0]["id"],
            )
        )

    return {"messages": new_msgs, "ask_human": False}


graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("human", human_node)


def get_next(state: State):
    if state["ask_human"]:
        return "human"

    return tools_condition(state)


graph_builder.set_entry_point("chatbot")
graph_builder.add_conditional_edges("chatbot", get_next)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("human", "chatbot")


graph = graph_builder.compile(checkpointer=memory, interrupt_before=["human"])


def stream_graph_updates(user_input: str, thread: str):
    for value in graph.stream(
        {"messages": [("user", user_input)]},
        {"configurable": {"thread_id": thread}},
        stream_mode="values",
    ):
        value["messages"][-1].pretty_print()


def run_chatbot():
    while True:
        user_input = input("User: ")
        thread = input("Thread (enter for default): ") or DEFAULT_THREAD_ID
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input, thread)


if __name__ == "__main__":
    run_chatbot()
