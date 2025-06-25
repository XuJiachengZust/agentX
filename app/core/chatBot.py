import os
from typing import Annotated, List, TypedDict

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import StateGraph, add_messages


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

class ChatApp:
    def __init__(self):
        # 初始化对话历史
        self.history = []  # 存储所有消息的历史记录

        # 设置环境变量
        os.environ["TAVILY_API_KEY"] = "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"
        os.environ["OPENAI_BASE_URL"] = "https://api.gptsapi.net/v1"
        os.environ["OPENAI_API_KEY"] = "sk-oR4bed2c6c453572fa5a5fc1ee4822dc065976d3071BIQLR"

        # 初始化模型
        self.llm = ChatOpenAI(
            model='gpt-4o',
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=0.5  # 略微提高温度使回复更有创意
        )

        # 初始化工具
        search = TavilySearchResults(max_results=3)  # 增加搜索结果数量
        self.tools = [search]

        # 构建状态图
        self.graph = self._create_graph()

    def _create_graph(self):
        """创建带有记忆功能的状态图"""
        graph_builder = StateGraph(State)

        # 添加节点
        graph_builder.add_node("chatbot", self._chatbot)
        graph_builder.add_node("tools", self._invoke_tool)

        # 添加条件边（修复判断逻辑）
        graph_builder.add_conditional_edges(
            "chatbot",
            lambda state: "tools" if (
                    state["messages"] and
                    isinstance(state["messages"][-1], AIMessage) and
                    state["messages"][-1].tool_calls
            ) else "end",
            {"tools": "tools", "end": END}
        )

        # 添加边
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_edge(START, "chatbot")

        # 设置入口点并编译图
        graph_builder.set_entry_point("chatbot")
        return graph_builder.compile()

    def _chatbot(self, state: State):
        """带上下文记忆的聊天节点"""
        # 绑定工具到LLM
        bound = self.llm.bind_tools(self.tools)
        # 调用模型（使用整个历史记录）
        response = bound.invoke(state["messages"])
        return {"messages": [response]}

    def _invoke_tool(self, state: State):
        """处理工具调用的自定义节点"""
        last_message = state["messages"][-1]

        # 提取工具调用信息
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"messages": [last_message]}

        # 执行所有工具调用
        tool_messages = []
        for tool_call in last_message.tool_calls:
            # 查找匹配的工具
            tool = next((t for t in self.tools if t.name == tool_call["name"]), None)
            if not tool:
                continue

            # 执行工具调用
            try:
                output = tool.invoke(tool_call["args"])
                tool_messages.append(
                    ToolMessage(
                        content=str(output),
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"]
                    )
                )
            except Exception as e:
                tool_messages.append(
                    ToolMessage(
                        content=f"Error executing tool: {str(e)}",
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"]
                    )
                )

        return {"messages": tool_messages}

    def process_user_input(self, user_input: str):
        """处理用户输入并返回助手响应"""
        # 添加用户消息到历史
        self.history.append(HumanMessage(content=user_input))

        # 构建当前状态（包含所有历史）
        current_state = {"messages": self.history.copy()}

        # 存储最终响应
        final_response = None

        # 处理图更新
        for event in self.graph.stream(current_state):
            for key, update in event.items():
                if key != "__end__" and "messages" in update:
                    # 更新历史记录
                    self.history.extend(update["messages"])

                    # 捕获最终响应
                    if update["messages"] and isinstance(update["messages"][-1], AIMessage):
                        final_response = update["messages"][-1].content

        return final_response

    def run(self):
        """运行聊天应用"""
        print("对话开始 (输入 'quit', 'exit' 或 'q' 结束)")
        print("此聊天具有上下文记忆功能，会记住之前的对话")

        try:
            while True:
                user_input = input("\nUser: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("对话结束")
                    break

                # 处理用户输入并获取响应
                response = self.process_user_input(user_input)

                if response:
                    print(f"\nAssistant: {response}")
                else:
                    print("\nAssistant: (思考中...)")

        except KeyboardInterrupt:
            print("\n对话结束")
        except Exception as e:
            print(f"\n发生错误: {str(e)}")


if __name__ == "__main__":
    app = ChatApp()
    app.run()