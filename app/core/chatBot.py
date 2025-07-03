import os
from typing import Annotated, List, TypedDict, Literal
from pydantic import SecretStr
from langchain_core.runnables.config import RunnableConfig
from langgraph.types import Command

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import StateGraph, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    llm_output: str | None  # 添加 llm_output 字段


class ChatApp:
    def __init__(self):
        # 设置环境变量
        os.environ["TAVILY_API_KEY"] = "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"
        os.environ["OPENAI_BASE_URL"] = "https://api.gptsapi.net/v1"
        os.environ["OPENAI_API_KEY"] = "sk-oR4bed2c6c453572fa5a5fc1ee4822dc065976d3071BIQLR"

        # 初始化记忆存储
        self.memory = MemorySaver()
        self.config: RunnableConfig = {
            "configurable": {
                "thread_id": "default_thread"
            },
            "run_name": "default_thread",
            "callbacks": None
        }

        # 初始化模型
        api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            model='gpt-4o',
            api_key=None if api_key is None else SecretStr(api_key),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=0.7
        )

        # 初始化工具
        self.tools = [TavilySearchResults(max_results=3)]

        # 构建状态图
        self.graph = self._create_graph()

    def _create_graph(self):
        """创建带有记忆功能的状态图"""
        graph_builder = StateGraph(State)

        # 添加节点
        graph_builder.add_node("chatbot", self._chatbot)
        graph_builder.add_node("tools", self._invoke_tool)
        graph_builder.add_node("human_assistance", self.human_approval)

        # 添加条件边
        graph_builder.add_conditional_edges(
            "chatbot",
            lambda state: "human_assistance" if (
                state["messages"] and
                isinstance(state["messages"][-1], AIMessage) and
                state["messages"][-1].tool_calls
            ) else "end",
            {"human_assistance": "human_assistance", "end": END}
        )

        # 从人工审批到工具调用或聊天节点的边
        graph_builder.add_conditional_edges(
            "human_assistance",
            lambda state: "tools" if state.get("approved", False) else "chatbot",
            {"tools": "tools", "chatbot": "chatbot"}
        )
        
        # 工具调用完成后返回聊天节点
        graph_builder.add_edge("tools", "chatbot")

        # 设置入口点
        graph_builder.add_edge(START, "chatbot")
        graph_builder.set_entry_point("chatbot")
        
        return graph_builder.compile(checkpointer=self.memory)

    def _chatbot(self, state: State):
        """带上下文记忆的聊天节点"""
        # 绑定工具到LLM
        bound = self.llm.bind_tools(self.tools)
        # 调用模型（使用整个历史记录）
        response = bound.invoke(state["messages"])
        return {"messages": [response], "llm_output": response.content}

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

    def human_approval(self, state: State) -> dict:
        """处理人工审批"""
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"messages": state["messages"], "approved": False}

        # 显示工具调用信息供审批
        tool_calls_info = "\n".join(
            f"工具: {call['name']}\n参数: {call['args']}"
            for call in last_message.tool_calls
        )
        
        # 使用 interrupt 获取用户审批
        is_approved = interrupt(
            {
                "question": "是否允许以下工具调用？",
                "tool_calls": tool_calls_info,
                "assistant_message": ""
            }
        )

        if is_approved:
            print("\n工具调用已批准，等待结果...")
        else:
            print("\n工具调用已拒绝")
            # 为每个被拒绝的工具调用添加拒绝响应
            tool_messages = [
                ToolMessage(
                    content="用户拒绝了此工具调用",
                    tool_call_id=call["id"],
                    name=call["name"]
                )
                for call in last_message.tool_calls
            ]
            return {"messages": state["messages"] + tool_messages, "approved": False}

        return {"messages": state["messages"], "approved": is_approved}

    def process_user_input(self, user_input: str):
        """处理用户输入并返回助手响应
        
        此方法处理用户输入，管理对话流程，包括工具调用的审批和执行。
        整个处理流程包括：接收用户输入 -> 调用模型 -> 处理工具调用（如果有）-> 返回响应
        
        Args:
            user_input (str): 用户输入的文本消息
            
        Returns:
            str: 助手的响应消息或工具调用的结果
        """
        # 将用户输入转换为消息对象，用于与模型交互
        user_message = HumanMessage(content=user_input)
        final_response = None

        try:
            # 初始化事件列表，用于跟踪对话过程中的所有事件
            events = []
            
            # 使用图结构处理对话流程
            # stream方法会逐步返回处理过程中的事件
            for event in self.graph.stream(
                    {"messages": [user_message]},  # 输入当前用户消息
                    self.config,  # 配置信息，包含会话ID等
                    stream_mode="values"  # 设置流模式为值模式
            ):
                events.append(event)  # 记录事件
                print(f"Debug - Event: {event}")  # 输出调试信息

                # 处理需要人工审批的工具调用
                if "__interrupt__" in event:
                    interrupt_data = event["__interrupt__"][0].value
                    print("\n需要审批工具调用:")
                    print(f"工具调用: {interrupt_data['tool_calls']}")
                    
                    # 获取用户审批输入
                    while True:
                        approval = input("\n是否允许? (y/n): ").lower().strip()
                        if approval in ['y', 'yes', 'n', 'no']:
                            break
                        print("请输入 y 或 n")
                    
                    # 处理审批结果
                    is_approved = approval in ['y', 'yes']
                    if not is_approved:
                        # 如果工具调用被拒绝，重置会话状态并返回提示信息
                        self.config = {
                            "configurable": {
                                "thread_id": f"thread_{hash(user_input)}"  # 生成新的会话ID
                            },
                            "run_name": f"thread_{hash(user_input)}",
                            "callbacks": None
                        }
                        return "抱歉，由于工具调用被拒绝，我无法提供相关信息。请问您有其他问题吗？"

                    # 如果工具调用被批准，继续执行
                    print("\n工具调用已批准，等待结果...")
                    # 使用 Command(resume=True) 继续执行被中断的流程
                    for next_event in self.graph.stream(
                        Command(resume=True),
                        self.config,
                        stream_mode="values"
                    ):
                        events.append(next_event)
                        print(f"Debug - Next Event: {next_event}")
                        # 检查工具调用的结果
                        if "messages" in next_event:
                            messages = next_event["messages"]
                            if messages and isinstance(messages[-1], ToolMessage):
                                return f"工具调用结果:\n{messages[-1].content}"
                    continue

                # 处理常规消息更新
                if "messages" in event:
                    messages = event["messages"]
                    if messages:
                        last_message = messages[-1]
                        # 处理工具调用的响应
                        if isinstance(last_message, ToolMessage):
                            return f"工具调用结果:\n{last_message.content}"
                        # 处理AI助手的响应
                        elif isinstance(last_message, AIMessage):
                            final_response = last_message.content

            # 返回最终响应或默认消息
            return final_response or "(无响应)"

        except Exception as e:
            # 异常处理：记录错误并重置状态
            print(f"\n调试 - 错误: {str(e)}")
            # 发生错误时重置会话状态
            self.config = {
                "configurable": {
                    "thread_id": f"thread_{hash(str(e))}"  # 使用错误信息生成新的会话ID
                },
                "run_name": f"thread_{hash(str(e))}",
                "callbacks": None
            }
            return f"发生错误: {str(e)}"

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