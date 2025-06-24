import os
from typing import Annotated

from langchain.chat_models import init_chat_model
# from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langgraph.constants import START
from langgraph.graph import add_messages, StateGraph
from IPython.display import Image, display
from typing_extensions import TypedDict

os.environ["OPENAI_BASE_URL"] = "https://api.gptsapi.net/v1"
os.environ["OPENAI_API_KEY"] = "sk-oR4bed2c6c453572fa5a5fc1ee4822dc065976d3071BIQLR"
llm = init_chat_model("openai:gpt-4o")
# llm = init_chat_model("deepseek:deepseek-r1")
# llm = ChatOpenAI(model='gpt-4')
class State(TypedDict):
    # messages 的类型是 "list"。注解中的 `add_messages` 函数
    # 定义了如何更新这个状态键
    # (在这种情况下，它是将消息追加到列表中，而不是覆盖它们)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# 第一个参数是唯一的节点名称
# 第二个参数是每次使用该节点时
# 将被调用的函数或对象
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile()

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break