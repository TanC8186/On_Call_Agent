"""RAG Agent 服务 - LangGraph 多轮对话 + 流式输出

支持 DeepSeek / DashScope / OpenAI 等多种 LLM 后端切换。
- DashScope: LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1  LLM_MODEL=qwen-max
"""

from typing import Annotated, Any, AsyncGenerator, Dict, Sequence

from langchain.agents import create_agent
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from loguru import logger
from typing_extensions import TypedDict

from app.config import config
from app.tools import get_current_time, retrieve_knowledge
from app.agent.mcp_client import get_mcp_client_with_retry


class AgentState(TypedDict):
    """Agent 状态"""
    messages: Annotated[Sequence[BaseMessage], add_messages]


def trim_messages_middleware(state: AgentState) -> dict[str, Any] | None:
    """修剪消息历史，只保留最近的几条消息以适应上下文窗口"""
    messages = state["messages"]
    if len(messages) <= 7:
        return None
    first_msg = messages[0]
    recent_messages = messages[-6:] if len(messages) % 2 == 0 else messages[-7:]
    new_messages = [first_msg] + list(recent_messages)
    logger.debug(f"修剪消息历史: {len(messages)} -> {len(new_messages)} 条")
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


class RagAgentService:
    """RAG Agent 服务 - 使用 LangGraph + ChatOpenAI (兼容 DeepSeek 等)"""

    def __init__(self, streaming: bool = True):
        self.model_name = config.llm_model
        self.streaming = streaming
        self.system_prompt = self._build_system_prompt()

        # 使用 OpenAI 兼容模式，通过 .env 配置切换提供商
        self.model = ChatOpenAI(
            model=self.model_name,
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            temperature=0.7,
            streaming=streaming,
        )

        # 定义基础工具
        self.tools = [retrieve_knowledge, get_current_time]

        # MCP 客户端（延迟初始化）
        self.mcp_tools: list = []

        # 创建内存检查点
        self.checkpointer = MemorySaver()

        # Agent 初始化（异步完成）
        self.agent = None
        self._agent_initialized = False

        logger.info(
            f"RAG Agent 初始化: model={self.model_name}, "
            f"base_url={config.llm_base_url}, streaming={streaming}"
        )

    async def _initialize_agent(self):
        """异步初始化 Agent（MCP 不可用时仅使用本地工具）"""
        if self._agent_initialized:
            return

        # 尝试加载 MCP 工具（MCP 服务不可用时自动跳过）
        try:
            mcp_client = await get_mcp_client_with_retry()
            mcp_tools = await mcp_client.get_tools()
            logger.info(f"成功加载 {len(mcp_tools)} 个 MCP 工具")
            self.mcp_tools = mcp_tools
        except Exception as e:
            logger.warning(f"MCP 工具加载失败（服务可能未启动），仅使用本地工具: {e}")
            self.mcp_tools = []

        all_tools = self.tools + self.mcp_tools
        self.agent = create_agent(
            self.model,
            tools=all_tools,
            checkpointer=self.checkpointer,
        )
        self._agent_initialized = True

        if all_tools:
            tool_names = [tool.name if hasattr(tool, "name") else str(tool) for tool in all_tools]
            logger.info(f"可用工具列表: {', '.join(tool_names)}")

    def _build_system_prompt(self) -> str:
        from textwrap import dedent
        return dedent("""
            你是一个专业的AI助手，能够使用多种工具来帮助用户解决问题。

            工作原则:
            1. 理解用户需求，选择合适的工具来完成任务
            2. 当需要获取实时信息或专业知识时，主动使用相关工具
            3. 基于工具返回的结果提供准确、专业的回答
            4. 如果工具无法提供足够信息，请诚实地告知用户

            回答要求:
            - 保持友好、专业的语气
            - 回答简洁明了，重点突出
            - 基于事实，不编造信息
            - 如有不确定的地方，明确说明

            请根据用户的问题，灵活使用可用工具，提供高质量的帮助。
        """).strip()

    async def query(self, question: str, session_id: str) -> str:
        """非流式处理用户问题"""
        try:
            await self._initialize_agent()
            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（非流式）: {question}")

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=question)
            ]
            agent_input = {"messages": messages}
            config_dict = {"configurable": {"thread_id": session_id}}

            result = await self.agent.ainvoke(input=agent_input, config=config_dict)

            messages_result = result.get("messages", [])
            if messages_result:
                last_message = messages_result[-1]
                answer = last_message.content if hasattr(last_message, 'content') else str(last_message)
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    tool_names = [tc.get("name", "unknown") for tc in last_message.tool_calls]
                    logger.info(f"[会话 {session_id}] Agent 调用了工具: {tool_names}")
                logger.info(f"[会话 {session_id}] RAG Agent 查询完成")
                return answer

            return ""
        except Exception as e:
            logger.error(f"[会话 {session_id}] RAG Agent 查询失败: {e}")
            raise

    async def query_stream(
        self, question: str, session_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """流式处理用户问题"""
        try:
            await self._initialize_agent()
            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（流式）: {question}")

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=question)
            ]
            agent_input = {"messages": messages}
            config_dict = {"configurable": {"thread_id": session_id}}

            async for token, metadata in self.agent.astream(
                input=agent_input, config=config_dict, stream_mode="messages",
            ):
                node_name = metadata.get('langgraph_node', 'unknown') if isinstance(metadata, dict) else 'unknown'
                message_type = type(token).__name__

                if message_type in ("AIMessage", "AIMessageChunk"):
                    content_blocks = getattr(token, 'content_blocks', None)
                    if content_blocks and isinstance(content_blocks, list):
                        for block in content_blocks:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                text_content = block.get('text', '')
                                if text_content:
                                    yield {
                                        "type": "content",
                                        "data": text_content,
                                        "node": node_name
                                    }

            logger.info(f"[会话 {session_id}] RAG Agent 查询完成（流式）")
            yield {"type": "complete"}
        except Exception as e:
            logger.error(f"[会话 {session_id}] RAG Agent 查询失败（流式）: {e}")
            yield {"type": "error", "data": str(e)}
            raise

    def get_session_history(self, session_id: str) -> list:
        """获取会话历史"""
        try:
            config_dict = {"configurable": {"thread_id": session_id}}
            checkpoint_tuple = self.checkpointer.get(config_dict)
            if not checkpoint_tuple:
                return []

            if hasattr(checkpoint_tuple, 'checkpoint'):
                checkpoint_data = checkpoint_tuple.checkpoint
            else:
                checkpoint_data = checkpoint_tuple[0] if checkpoint_tuple else {}

            messages = checkpoint_data.get("channel_values", {}).get("messages", [])
            history = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    continue
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content if hasattr(msg, 'content') else str(msg)
                timestamp = getattr(msg, 'timestamp', None)
                if not timestamp:
                    from datetime import datetime
                    timestamp = datetime.now().isoformat()
                history.append({"role": role, "content": content, "timestamp": timestamp})

            logger.info(f"获取会话历史: {session_id}, 消息数量: {len(history)}")
            return history
        except Exception as e:
            logger.error(f"获取会话历史失败: {session_id}, 错误: {e}")
            return []

    def clear_session(self, session_id: str) -> bool:
        """清空会话历史"""
        try:
            self.checkpointer.delete_thread(session_id)
            logger.info(f"已清除会话历史: {session_id}")
            return True
        except Exception as e:
            logger.error(f"清空会话历史失败: {session_id}, 错误: {e}")
            return False

    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("清理 RAG Agent 服务资源...")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")


# 全局单例 - 启用流式输出
rag_agent_service = RagAgentService(streaming=True)
