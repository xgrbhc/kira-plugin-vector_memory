"""
向量存储抽象层，当前实现为 ChromaDB 后端。

这里把 Chroma 的同步 API 统一放到 `asyncio.to_thread()` 中执行，
避免统计、迁移、导出等批量操作阻塞 KiraAI 的事件循环。
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.logging_manager import get_logger

logger = get_logger("vector_memory.store", "cyan")


class VectorStore(ABC):
    """向量存储抽象基类。"""

    @abstractmethod
    async def add(
        self, text: str, embedding: List[float], metadata: Dict[str, Any]
    ) -> str:
        """添加一条记忆并返回 memory_id。"""
        pass

    @abstractmethod
    async def search(
        self,
        embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """搜索相似记忆。"""
        pass

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """删除一条记忆。"""
        pass

    @abstractmethod
    async def update_memory(
        self,
        memory_id: str,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """更新单条记忆；可按需更新正文、embedding 和 metadata。"""
        pass

    @abstractmethod
    async def count(self) -> int:
        """获取当前集合中的记忆总数。"""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """清空当前集合。"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """关闭存储。"""
        pass


class ChromaVectorStore(VectorStore):
    """ChromaDB 向量存储实现。"""

    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "kira_memories",
        collection_metadata: Optional[Dict[str, Any]] = None,
        create_if_missing: bool = True,
    ):
        """
        初始化 ChromaDB 存储。

        参数:
            persist_dir: ChromaDB 持久化目录。
            collection_name: Chroma collection 名称。
            collection_metadata: 创建新集合时使用的 metadata。
            create_if_missing: 为 False 时只打开已存在集合，避免误创建空集合。
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("使用向量记忆库需要安装 chromadb: pip install chromadb")

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        metadata = collection_metadata or {"description": "KiraAI 语义记忆库"}
        if create_if_missing:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata=metadata,
            )
        else:
            self.collection = self.client.get_collection(name=collection_name)

        logger.info(
            f"ChromaDB 初始化完成，集合: {collection_name}，"
            f"现有记忆: {self.collection.count()} 条"
        )

    @staticmethod
    def _clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Chroma metadata 只支持基础标量，这里统一清洗，避免写入失败。"""
        clean_metadata: Dict[str, Any] = {}
        for key, value in (metadata or {}).items():
            if value is None:
                clean_metadata[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                clean_metadata[key] = value
            else:
                clean_metadata[key] = str(value)
        return clean_metadata

    @classmethod
    def _normalize_where(cls, filter_dict: Optional[Dict[str, Any]]) -> Optional[Dict]:
        """
        把简单 dict 或 `$and`/`$or` 过滤表达式转换为 Chroma where。

        简单写法:
            {"scope": "global"}

        复杂写法:
            {"$or": [{"scope": "global"}, {"$and": [{"scope": "session"}, ...]}]}
        """
        if not filter_dict:
            return None

        if "$and" in filter_dict:
            children = [
                cls._normalize_where(item)
                for item in filter_dict.get("$and", [])
                if item
            ]
            children = [item for item in children if item]
            if not children:
                return None
            if len(children) == 1:
                return children[0]
            return {"$and": children}

        if "$or" in filter_dict:
            children = [
                cls._normalize_where(item)
                for item in filter_dict.get("$or", [])
                if item
            ]
            children = [item for item in children if item]
            if not children:
                return None
            if len(children) == 1:
                return children[0]
            return {"$or": children}

        field_conditions = []
        for key, value in filter_dict.items():
            if isinstance(value, dict) and any(k.startswith("$") for k in value):
                field_conditions.append({key: value})
            else:
                field_conditions.append({key: {"$eq": value}})

        if not field_conditions:
            return None
        if len(field_conditions) == 1:
            return field_conditions[0]
        return {"$and": field_conditions}

    async def add(
        self, text: str, embedding: List[float], metadata: Dict[str, Any]
    ) -> str:
        """添加记忆。"""
        memory_id = f"mem_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        clean_metadata = self._clean_metadata(metadata)

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
        """搜索相似记忆。"""
        where = self._normalize_where(filter_dict)
        current_count = await self.count()
        if current_count <= 0:
            return []

        results = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[embedding],
            n_results=min(top_k, current_count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results and results.get("ids") and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                output.append(
                    {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": distance,
                        # 旧集合默认 L2 距离，这里保持旧版行为，避免阈值含义突然变化。
                        "similarity": 1 / (1 + distance),
                    }
                )

        return output

    async def get_by_id(
        self, memory_id: str, include_embedding: bool = False
    ) -> Optional[Dict[str, Any]]:
        """按 ID 获取单条记忆。"""
        include_fields = ["documents", "metadatas"]
        if include_embedding:
            include_fields.append("embeddings")

        results = await asyncio.to_thread(
            self.collection.get,
            ids=[memory_id],
            include=include_fields,
        )
        if not results or not results.get("ids"):
            return None

        memory: Dict[str, Any] = {
            "id": results["ids"][0],
            "text": results["documents"][0] if results.get("documents") else "",
            "metadata": results["metadatas"][0] if results.get("metadatas") else {},
        }
        if include_embedding and results.get("embeddings") is not None:
            embedding = results["embeddings"][0]
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            memory["embedding"] = embedding
        return memory

    async def update_metadata(self, memory_id: str, metadata: Dict[str, Any]) -> bool:
        """只更新 metadata，不改 document 和 embedding。"""
        try:
            await asyncio.to_thread(
                self.collection.update,
                ids=[memory_id],
                metadatas=[self._clean_metadata(metadata)],
            )
            return True
        except Exception as e:
            logger.error(f"更新记忆 metadata 失败: {memory_id}, {e}")
            return False

    async def update_memory(
        self,
        memory_id: str,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """更新单条记忆；正文变更时由调用方传入重新生成的 embedding。"""
        try:
            kwargs: Dict[str, Any] = {"ids": [memory_id]}
            if text is not None:
                kwargs["documents"] = [text]
            if embedding is not None:
                kwargs["embeddings"] = [embedding]
            if metadata is not None:
                kwargs["metadatas"] = [self._clean_metadata(metadata)]
            if len(kwargs) == 1:
                return False

            await asyncio.to_thread(self.collection.update, **kwargs)
            logger.debug(f"更新记忆: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"更新记忆失败: {memory_id}, {e}")
            return False

    async def delete(self, memory_id: str) -> bool:
        """删除记忆。"""
        try:
            await asyncio.to_thread(self.collection.delete, ids=[memory_id])
            logger.debug(f"删除记忆: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"删除记忆失败: {e}")
            return False

    async def count(self) -> int:
        """获取记忆总数。"""
        return await asyncio.to_thread(self.collection.count)

    async def clear(self) -> None:
        """清空当前集合。"""
        await asyncio.to_thread(self.client.delete_collection, self.collection_name)
        self.collection = await asyncio.to_thread(
            self.client.get_or_create_collection,
            name=self.collection_name,
            metadata={"description": "KiraAI 语义记忆库"},
        )
        logger.info("已清空所有记忆")

    async def delete_collection(self) -> None:
        """删除当前打开的集合。调用方必须先完成确认和备份。"""
        await asyncio.to_thread(self.client.delete_collection, self.collection_name)
        self.collection = None
        logger.info(f"已删除 Chroma 集合: {self.collection_name}")

    async def get_by_session(self, session_id: str, limit: int = 100) -> List[Dict]:
        """获取指定会话的记忆。"""
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
        """简单 FIFO 清理。"""
        current_count = await self.count()
        if current_count <= max_count:
            return 0

        all_memories = await self.get_all_memories(limit=current_count)
        memories_with_time = [
            (
                item["id"],
                item.get("metadata", {}).get("timestamp", 0),
            )
            for item in all_memories
        ]
        memories_with_time.sort(key=lambda x: x[1])

        delete_count = current_count - max_count
        ids_to_delete = [item[0] for item in memories_with_time[:delete_count]]

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
        """综合时间和重要性清理超额记忆。"""
        current_count = await self.count()
        if current_count <= max_count:
            return 0

        all_memories = await self.get_all_memories(limit=current_count)
        now = int(time.time())

        scored_memories = []
        for memory in all_memories:
            metadata = memory.get("metadata", {})
            timestamp = metadata.get("timestamp", 0)
            importance = float(metadata.get("importance", 0.3))
            mem_type = metadata.get("type", "raw")

            if mem_type == "summary":
                score = float("inf")
            else:
                age_days = (now - timestamp) / 86400
                time_score = max(0, 1 - (age_days / 365))
                score = time_score * time_weight + importance * importance_weight

            scored_memories.append((memory["id"], score, mem_type))

        scored_memories.sort(key=lambda x: x[1])
        delete_count = current_count - max_count
        ids_to_delete = []

        for mem_id, _score, mem_type in scored_memories:
            if len(ids_to_delete) >= delete_count:
                break
            if mem_type != "summary":
                ids_to_delete.append(mem_id)

        if ids_to_delete:
            await asyncio.to_thread(self.collection.delete, ids=ids_to_delete)
            logger.info(f"智能清理了 {len(ids_to_delete)} 条记忆")

        return len(ids_to_delete)

    async def get_all_memories(
        self,
        limit: int = 1000,
        offset: int = 0,
        include_embeddings: bool = False,
    ) -> List[Dict]:
        """
        分页获取记忆。

        参数:
            limit: 本页最大返回数量。
            offset: 起始偏移。
            include_embeddings: 是否包含 embedding。
        """
        include_fields = ["documents", "metadatas"]
        if include_embeddings:
            include_fields.append("embeddings")

        results = await asyncio.to_thread(
            self.collection.get,
            limit=limit,
            offset=offset,
            include=include_fields,
        )

        output = []
        if results and results.get("ids"):
            embeddings = results.get("embeddings")
            for i in range(len(results["ids"])):
                mem = {
                    "id": results["ids"][i],
                    "text": results["documents"][i] if results.get("documents") else "",
                    "metadata": (
                        results["metadatas"][i] if results.get("metadatas") else {}
                    ),
                }
                if include_embeddings and embeddings is not None:
                    embedding = embeddings[i]
                    if hasattr(embedding, "tolist"):
                        embedding = embedding.tolist()
                    mem["embedding"] = embedding
                output.append(mem)

        return output

    async def close(self) -> None:
        """
        关闭存储。

        ChromaDB PersistentClient 没有稳定的公开 close API。这里不删除目录，
        只释放引用；Windows 下文件句柄可能会延迟释放。
        """
        self.collection = None
        self.client = None
