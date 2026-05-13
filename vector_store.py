"""
向量存储抽象层
当前实现 ChromaDB 后端
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional

from core.logging_manager import get_logger

logger = get_logger("vector_memory.store", "cyan")


class VectorStore(ABC):
    """向量存储抽象基类"""

    @abstractmethod
    async def add(
        self, text: str, embedding: List[float], metadata: Dict[str, Any]
    ) -> str:
        """
        添加记忆

        Args:
            text: 文本内容
            embedding: 向量
            metadata: 元数据（session_id, user_id, timestamp 等）

        Returns:
            记忆 ID
        """
        pass

    @abstractmethod
    async def search(
        self,
        embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """
        搜索相似记忆

        Args:
            embedding: 查询向量
            top_k: 返回数量
            filter_dict: 过滤条件

        Returns:
            搜索结果列表
        """
        pass

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """删除记忆"""
        pass

    @abstractmethod
    async def count(self) -> int:
        """获取记忆总数"""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """清空所有记忆"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """关闭存储"""
        pass


class ChromaVectorStore(VectorStore):
    """ChromaDB 向量存储实现"""

    def __init__(self, persist_dir: str, collection_name: str = "kira_memories"):
        """
        初始化 ChromaDB 存储

        Args:
            persist_dir: 持久化目录
            collection_name: 集合名称
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("使用向量记忆库需要安装 chromadb: pip install chromadb")

        # 初始化 ChromaDB 客户端
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir), settings=Settings(anonymized_telemetry=False)
        )

        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"description": "KiraAI 语义记忆库"}
        )

        logger.info(
            f"ChromaDB 初始化完成，集合: {collection_name}，"
            f"现有记忆: {self.collection.count()} 条"
        )

    async def add(
        self, text: str, embedding: List[float], metadata: Dict[str, Any]
    ) -> str:
        """添加记忆"""
        memory_id = f"mem_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

        # 确保 metadata 中的值是 ChromaDB 支持的类型
        clean_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean_metadata[key] = value
            else:
                clean_metadata[key] = str(value)

        await asyncio.to_thread(
            self.collection.add,
            ids=[memory_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[clean_metadata],
        )

        logger.debug(f"添加记忆: {memory_id}, 长度: {len(text)}")
        return memory_id

    async def search(
        self,
        embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """搜索相似记忆"""
        # 构建 where 条件
        where = None
        if filter_dict:
            if len(filter_dict) == 1:
                key, value = list(filter_dict.items())[0]
                where = {key: {"$eq": value}}
            else:
                where = {"$and": [{k: {"$eq": v}} for k, v in filter_dict.items()]}

        current_count = await self.count()
        results = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[embedding],
            n_results=min(top_k, current_count or 1),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # 格式化结果
        output = []
        if results and results.get("ids") and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                output.append(
                    {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        # 将距离转换为相似度（ChromaDB 默认使用 L2 距离）
                        "similarity": 1 / (1 + results["distances"][0][i]),
                    }
                )

        return output

    async def delete(self, memory_id: str) -> bool:
        """删除记忆"""
        try:
            await asyncio.to_thread(self.collection.delete, ids=[memory_id])
            logger.debug(f"删除记忆: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"删除记忆失败: {e}")
            return False

    async def count(self) -> int:
        """获取记忆总数"""
        return await asyncio.to_thread(self.collection.count)

    async def clear(self) -> None:
        """清空所有记忆"""
        # ChromaDB 不支持直接清空，需要删除后重建
        await asyncio.to_thread(self.client.delete_collection, self.collection_name)
        self.collection = await asyncio.to_thread(
            self.client.get_or_create_collection,
            name=self.collection_name,
            metadata={"description": "KiraAI 语义记忆库"},
        )
        logger.info("已清空所有记忆")

    async def get_by_session(self, session_id: str, limit: int = 100) -> List[Dict]:
        """获取指定会话的所有记忆"""
        results = await asyncio.to_thread(
            self.collection.get,
            where={"session_id": {"$eq": session_id}},
            limit=limit,
            include=["documents", "metadatas"],
        )

        output = []
        if results and results.get("ids"):
            for i in range(len(results["ids"])):
                output.append(
                    {
                        "id": results["ids"][i],
                        "text": results["documents"][i],
                        "metadata": results["metadatas"][i],
                    }
                )

        return output

    async def cleanup_old(self, max_count: int) -> int:
        """清理超出限制的旧记忆（简单 FIFO）"""
        current_count = await self.count()
        if current_count <= max_count:
            return 0

        # 获取所有记忆并按时间排序
        all_memories = await asyncio.to_thread(
            self.collection.get,
            include=["metadatas"],
        )

        if not all_memories or not all_memories.get("ids"):
            return 0

        # 按时间戳排序
        memories_with_time = []
        for i, memory_id in enumerate(all_memories["ids"]):
            timestamp = all_memories["metadatas"][i].get("timestamp", 0)
            memories_with_time.append((memory_id, timestamp))

        memories_with_time.sort(key=lambda x: x[1])

        # 删除最旧的记忆
        delete_count = current_count - max_count
        ids_to_delete = [m[0] for m in memories_with_time[:delete_count]]

        if ids_to_delete:
            await asyncio.to_thread(self.collection.delete, ids=ids_to_delete)
            logger.info(f"清理了 {len(ids_to_delete)} 条旧记忆")

        return len(ids_to_delete)

    async def cleanup_smart(
        self,
        max_count: int,
        time_weight: float = 0.3,
        importance_weight: float = 0.7,
    ) -> int:
        """
        智能清理：综合时间和重要性

        Args:
            max_count: 最大保留数量
            time_weight: 时间权重
            importance_weight: 重要性权重

        Returns:
            删除的记忆数量
        """
        current_count = await self.count()
        if current_count <= max_count:
            return 0

        # 获取所有记忆
        all_memories = await asyncio.to_thread(
            self.collection.get,
            include=["metadatas"],
        )

        if not all_memories or not all_memories.get("ids"):
            return 0

        now = int(time.time())

        # 计算综合评分
        scored_memories = []
        for i, memory_id in enumerate(all_memories["ids"]):
            metadata = all_memories["metadatas"][i]
            timestamp = metadata.get("timestamp", 0)
            importance = float(metadata.get("importance", 0.3))
            mem_type = metadata.get("type", "raw")

            # 摘要类型永不删除
            if mem_type == "summary":
                score = float("inf")
            else:
                # 时间分数：越久远越低
                age_days = (now - timestamp) / 86400
                time_score = max(0, 1 - (age_days / 365))

                # 综合分数
                score = time_score * time_weight + importance * importance_weight

            scored_memories.append((memory_id, score, mem_type))

        # 按分数升序排序（低分优先删除）
        scored_memories.sort(key=lambda x: x[1])

        # 删除超出限制的低分记忆
        delete_count = current_count - max_count
        ids_to_delete = []

        for mem_id, score, mem_type in scored_memories:
            if len(ids_to_delete) >= delete_count:
                break
            # 不删除摘要
            if mem_type != "summary":
                ids_to_delete.append(mem_id)

        if ids_to_delete:
            await asyncio.to_thread(self.collection.delete, ids=ids_to_delete)
            logger.info(f"智能清理了 {len(ids_to_delete)} 条记忆")

        return len(ids_to_delete)

    async def get_all_memories(
        self, limit: int = 1000, include_embeddings: bool = False
    ) -> List[Dict]:
        """
        获取所有记忆

        Args:
            limit: 最大返回数量
            include_embeddings: 是否包含向量

        Returns:
            记忆列表
        """
        include_fields = ["documents", "metadatas"]
        if include_embeddings:
            include_fields.append("embeddings")

        results = self.collection.get(limit=limit, include=include_fields)

        output = []
        if results and results.get("ids"):
            for i in range(len(results["ids"])):
                mem = {
                    "id": results["ids"][i],
                    "text": results["documents"][i] if results.get("documents") else "",
                    "metadata": (
                        results["metadatas"][i] if results.get("metadatas") else {}
                    ),
                }
                if include_embeddings and results.get("embeddings"):
                    mem["embedding"] = results["embeddings"][i]
                output.append(mem)

        return output

    async def close(self) -> None:
        """关闭存储"""
        # ChromaDB PersistentClient 会自动持久化
        pass
