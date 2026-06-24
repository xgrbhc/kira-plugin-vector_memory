"""
上下文注入模块
在消息处理前自动检索并注入相关记忆
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from core.logging_manager import get_logger

logger = get_logger("vector_memory.injector", "cyan")


@dataclass
class InjectionResult:
    """注入结果"""

    injected: bool
    memories: List[Dict[str, Any]]
    reason: str


class ContextInjector:
    """上下文注入器"""

    INJECTION_TEMPLATE = """[系统提示 - 相关历史记忆]
以下是与当前对话可能相关的历史记忆，请自然地参考这些信息回复，但不要直接说"我查到了记忆"或"根据历史记录"：

{memories}
[历史记忆结束]"""

    def __init__(
        self,
        embedding,
        vector_store,
        threshold: float = 0.75,
        top_k: int = 2,
        cooldown: int = 60,
        rerank_fn: Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
        rerank_enabled: bool = True,
    ):
        """
        初始化注入器

        Args:
            embedding: Embedding 生成器
            vector_store: 向量存储
            threshold: 相似度阈值
            top_k: 最大注入数量
            cooldown: 冷却时间（秒）
        """
        self.embedding = embedding
        self.vector_store = vector_store
        self.threshold = threshold
        self.top_k = top_k
        self.cooldown = cooldown
        self.rerank_fn = rerank_fn
        self.rerank_enabled = rerank_enabled

        # 冷却记录 {session_id: last_injection_time}
        self._cooldown_map: Dict[str, float] = {}

    async def try_inject(
        self, query: str, session_id: str, filter_dict: Optional[Dict] = None
    ) -> InjectionResult:
        """
        尝试注入相关记忆

        Args:
            query: 查询文本（用户消息）
            session_id: 会话 ID
            filter_dict: 过滤条件

        Returns:
            InjectionResult
        """
        # 检查冷却
        now = time.time()
        last_time = self._cooldown_map.get(session_id, 0)
        if now - last_time < self.cooldown:
            remaining = int(self.cooldown - (now - last_time))
            return InjectionResult(
                injected=False, memories=[], reason=f"冷却中 ({remaining}s)"
            )

        # 查询长度检查
        if len(query.strip()) < 5:
            return InjectionResult(injected=False, memories=[], reason="查询过短")

        try:
            # 生成查询向量
            query_embedding = await self.embedding.generate(query)

            # 搜索相似记忆
            results = await self.vector_store.search(
                embedding=query_embedding,
                top_k=self.top_k * 2,  # 多查一些，后面过滤
                filter_dict=filter_dict,
            )

            # 按相似度过滤
            relevant_memories = [
                r for r in results if r.get("similarity", 0) >= self.threshold
            ]
            if self.rerank_enabled and self.rerank_fn:
                relevant_memories = self.rerank_fn(relevant_memories)
            relevant_memories = relevant_memories[: self.top_k]

            if not relevant_memories:
                return InjectionResult(
                    injected=False, memories=[], reason="未找到高相似度记忆"
                )

            # 更新冷却
            self._cooldown_map[session_id] = now

            logger.debug(f"注入 {len(relevant_memories)} 条记忆到会话 {session_id}")

            return InjectionResult(
                injected=True,
                memories=relevant_memories,
                reason=f"找到 {len(relevant_memories)} 条相关记忆",
            )

        except Exception as e:
            logger.error(f"注入检索失败: {e}")
            return InjectionResult(
                injected=False, memories=[], reason=f"检索错误: {str(e)}"
            )

    def format_injection(self, memories: List[Dict[str, Any]]) -> str:
        """
        格式化注入内容

        Args:
            memories: 记忆列表

        Returns:
            格式化后的字符串
        """
        if not memories:
            return ""

        memory_texts = []
        for mem in memories:
            text = mem.get("text", "")
            metadata = mem.get("metadata", {})
            timestamp = metadata.get("timestamp", 0)

            # 格式化时间
            if timestamp:
                time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
            else:
                time_str = "未知时间"

            user = metadata.get("user_nickname", "用户")

            # 截断过长文本
            if len(text) > 150:
                text = text[:150] + "..."

            memory_texts.append(f"[{time_str}] {user}: {text}")

        return self.INJECTION_TEMPLATE.format(memories="\n".join(memory_texts))

    def clear_cooldown(self, session_id: str = None):
        """清除冷却记录"""
        if session_id:
            self._cooldown_map.pop(session_id, None)
        else:
            self._cooldown_map.clear()

    def get_cooldown_remaining(self, session_id: str) -> int:
        """获取剩余冷却时间"""
        last_time = self._cooldown_map.get(session_id, 0)
        remaining = self.cooldown - (time.time() - last_time)
        return max(0, int(remaining))
