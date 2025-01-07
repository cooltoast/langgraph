from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict

from env import set_keys
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

set_keys()


class State(TypedDict):
    messages: Annotated[list, add_messages]


tavily_search = TavilySearchResults(max_results=2)
tools = [tavily_search]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620").bind_tools(tools)

graph_builder = StateGraph(State)


graph_builder.add_node(
    "chatbot", lambda state: {"messages": [llm.invoke(state["messages"])]}
)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.set_entry_point("chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant: ", value["messages"][-1].content)


def run_chatbot():
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)


if __name__ == "__main__":
    run_chatbot()
