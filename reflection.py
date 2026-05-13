"""
记忆反思与摘要模块
定时将碎片化记忆整理为结构化知识
"""

import time
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional

from core.logging_manager import get_logger

logger = get_logger("vector_memory.reflection", "cyan")


class ReflectionManager:
    """记忆反思管理器"""

    REFLECTION_PROMPT = """请分析以下对话记录，提取关于用户的关键信息。

对话记录：
{memories}

请按以下格式输出摘要（仅输出有实际内容的部分，跳过空的部分）：

## 用户画像
- 身份信息：（如姓名、职业、年龄等）
- 偏好习惯：（如喜欢/讨厌的事物）
- 性格特点：（如说话风格、兴趣爱好）

## 重要事件
（记录对话中提到的重要事件或决定）

## 关系进展
（与 AI 助手的互动特点、信任度变化等）

## 待记住事项
（用户明确要求记住的事情、约定等）

注意：
1. 只提取有价值的信息，忽略闲聊
2. 使用简洁的语言
3. 如果某部分没有相关内容，直接跳过该部分"""

    def __init__(
        self,
        embedding,
        vector_store,
        llm_api,
        delete_raw: bool = False,
        min_memories: int = 5,
    ):
        """
        初始化反思管理器

        Args:
            embedding: Embedding 生成器
            vector_store: 向量存储
            llm_api: LLM 客户端
            delete_raw: 反思后是否删除原始记忆
            min_memories: 触发反思的最小记忆数
        """
        self.embedding = embedding
        self.vector_store = vector_store
        self.llm_api = llm_api
        self.delete_raw = delete_raw
        self.min_memories = min_memories

    async def run_daily_reflection(self) -> Dict[str, Any]:
        """
        执行每日记忆反思

        Returns:
            反思结果统计
        """
        logger.info("开始每日记忆反思...")

        stats = {
            "total_memories": 0,
            "sessions_processed": 0,
            "summaries_created": 0,
            "memories_deleted": 0,
            "errors": [],
        }

        try:
            # 获取过去 24 小时的记忆
            cutoff_time = int(time.time()) - 86400
            memories = await self._get_recent_memories(cutoff_time)

            stats["total_memories"] = len(memories)

            if len(memories) < self.min_memories:
                logger.info(
                    f"记忆数量不足 ({len(memories)} < {self.min_memories})，跳过反思"
                )
                return stats

            # 按会话分组
            session_groups = self._group_by_session(memories)

            # 处理每个会话
            for session_id, session_memories in session_groups.items():
                try:
                    result = await self._process_session(session_id, session_memories)
                    if result:
                        stats["sessions_processed"] += 1
                        stats["summaries_created"] += 1
                        if self.delete_raw:
                            stats["memories_deleted"] += len(session_memories)
                except Exception as e:
                    error_msg = f"处理会话 {session_id} 失败: {e}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)

            logger.info(f"每日反思完成: {stats}")
            return stats

        except Exception as e:
            error_msg = f"每日反思失败: {e}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
            return stats

    async def _get_recent_memories(self, cutoff_time: int) -> List[Dict]:
        """获取指定时间后的所有记忆"""
        try:
            count = await self.vector_store.count()
            results = await self.vector_store.get_all_memories(limit=max(count, 1))
        except Exception as e:
            logger.error(f"获取最近记忆失败: {e}")
            return []

        # 按时间过滤，排除已有的摘要
        recent = []
        for r in results:
            metadata = r.get("metadata", {})
            timestamp = metadata.get("timestamp", 0)
            mem_type = metadata.get("type", "raw")

            if timestamp >= cutoff_time and mem_type != "summary":
                recent.append(r)

        return recent

    def _group_by_session(self, memories: List[Dict]) -> Dict[str, List[Dict]]:
        """按会话 ID 分组"""
        groups: Dict[str, List[Dict]] = {}
        for mem in memories:
            session_id = mem.get("metadata", {}).get("session_id", "unknown")
            if session_id not in groups:
                groups[session_id] = []
            groups[session_id].append(mem)
        return groups

    def _get_summary_llm_client(self):
        """优先获取更轻量的摘要 LLM，失败时回退到默认 LLM。"""
        provider_mgr = getattr(self.llm_api, "provider_mgr", None)
        if not provider_mgr:
            return None

        for getter_name in ("get_default_fast_llm", "get_default_llm"):
            getter = getattr(provider_mgr, getter_name, None)
            if not callable(getter):
                continue
            try:
                client = getter()
            except Exception as e:
                logger.warning(f"获取 {getter_name} 失败: {e}")
                continue
            if client:
                return client

        return None

    def _build_fallback_summary(self, memories: List[Dict], title: str) -> str:
        """在 LLM 不可用时，生成一个确定性的摘要。"""
        users = []
        snippets = []

        for mem in memories[:8]:
            metadata = mem.get("metadata", {})
            user = metadata.get("user_nickname", "用户")
            text = (mem.get("text", "") or "").strip()
            if user and user not in users:
                users.append(user)
            if text:
                if len(text) > 120:
                    text = text[:120] + "..."
                snippets.append(f"- {user}: {text}")

        lines = [title, f"- 记忆条数: {len(memories)}"]
        if users:
            lines.append(f"- 相关用户: {'、'.join(users[:5])}")

        if snippets:
            lines.append("")
            lines.append("主要记忆片段:")
            lines.extend(snippets)

        return "\n".join(lines)

    async def _generate_summary_text(self, memories: List[Dict], days: int, label: str) -> str:
        """优先使用 LLM 生成摘要，失败时回退到确定性摘要。"""
        memory_texts = []
        for mem in memories[:50]:
            user = mem.get("metadata", {}).get("user_nickname", "用户")
            text = mem.get("text", "")
            memory_texts.append(f"- {user}: {text}")

        summary_prompt = f"""请总结以下 {len(memory_texts)} 条对话记忆（{label}），提取关键信息：

{chr(10).join(memory_texts)}

请按以下格式总结：
1. 主要话题
2. 重要事件或信息
3. 用户偏好或习惯（如有）
"""

        llm_client = self._get_summary_llm_client()
        if llm_client:
            try:
                from core.provider import LLMRequest

                request = LLMRequest(messages=[{"role": "user", "content": summary_prompt}])
                response = await llm_client.chat(request)
                summary_text = (response.text_response or "").strip()
                if len(summary_text) >= 20:
                    return summary_text
                logger.warning("LLM 生成的摘要过短，改用兜底摘要")
            except Exception:
                logger.error(f"LLM 生成摘要失败，使用兜底摘要:\n{traceback.format_exc()}")

        return self._build_fallback_summary(memories, f"【{label}】")

    async def _process_session(
        self, session_id: str, memories: List[Dict]
    ) -> Optional[str]:
        """
        处理单个会话的记忆反思

        Returns:
            生成的摘要记忆 ID
        """
        if len(memories) < 3:  # 太少不值得摘要
            return None

        memories.sort(key=lambda x: x.get("metadata", {}).get("timestamp", 0))

        memory_ids = []
        for mem in memories[:50]:  # 限制数量
            memory_ids.append(mem.get("id"))

        summary_text = await self._generate_summary_text(
            memories=memories,
            days=1,
            label=f"会话 {session_id} 记忆反思",
        )

        if not summary_text:
            logger.warning(f"会话 {session_id} 未能生成摘要")
            return None

        summary_embedding = await self.embedding.generate(summary_text)

        today = datetime.now().strftime("%Y-%m-%d")
        summary_metadata = {
            "type": "summary",
            "importance": 1.0,
            "source_count": len(memories),
            "time_range": today,
            "session_id": session_id,
            "timestamp": int(time.time()),
            "user_id": "system",
            "user_nickname": "记忆摘要",
            "platform": "system",
            "adapter": "reflection",
        }

        summary_id = await self.vector_store.add(
            text=f"[{today} 记忆摘要]\n{summary_text}",
            embedding=summary_embedding,
            metadata=summary_metadata,
        )

        logger.info(f"会话 {session_id} 生成摘要: {summary_id}")

        if self.delete_raw and memory_ids:
            for mem_id in memory_ids:
                if mem_id:
                    await self.vector_store.delete(mem_id)
            logger.info(f"删除 {len(memory_ids)} 条原始记忆")

        return summary_id

    async def manual_reflection(self, session_id: str = None, days: int = 7) -> str:
        """
        手动触发记忆反思

        Args:
            session_id: 指定会话（可选）
            days: 反思天数范围

        Returns:
            反思结果描述
        """
        cutoff_time = int(time.time()) - (days * 86400)
        memories = await self._get_recent_memories(cutoff_time)

        if session_id:
            memories = [
                m
                for m in memories
                if m.get("metadata", {}).get("session_id") == session_id
            ]

        if len(memories) < self.min_memories:
            return f"记忆数量不足 ({len(memories)} 条)，无法生成有效摘要"

        session_groups = self._group_by_session(memories)
        results = []

        for sid, mems in session_groups.items():
            summary_id = await self._process_session(sid, mems)
            if summary_id:
                results.append(f"会话 {sid}: 生成摘要 {summary_id}")

        if not results:
            return "未能生成任何摘要"

        return "反思完成:\n" + "\n".join(results)

    async def summarize_recent_memories(
        self, days: int = 7, session_id: str = None
    ) -> str:
        """总结最近一段时间的记忆。"""
        cutoff_time = int(time.time()) - (days * 86400)
        memories = await self._get_recent_memories(cutoff_time)

        if session_id:
            memories = [
                m
                for m in memories
                if m.get("metadata", {}).get("session_id") == session_id
            ]

        if not memories:
            return f"最近 {days} 天没有记忆"

        return await self._generate_summary_text(
            memories=memories,
            days=days,
            label=f"最近 {days} 天记忆总结",
        )

    async def get_summaries(
        self, session_id: str = None, limit: int = 10
    ) -> List[Dict]:
        """
        获取已生成的摘要

        Args:
            session_id: 指定会话（可选）
            limit: 最大返回数量

        Returns:
            摘要列表
        """
        # 使用摘要相关的查询
        query_embedding = await self.embedding.generate("记忆摘要总结")

        filter_dict = {"type": "summary"}
        if session_id:
            filter_dict["session_id"] = session_id

        results = await self.vector_store.search(
            embedding=query_embedding, top_k=limit, filter_dict=filter_dict
        )

        # 只保留摘要类型
        summaries = [
            r for r in results if r.get("metadata", {}).get("type") == "summary"
        ]

        return summaries
