"""
Embedding 生成器
支持云端 Embedding 接口
"""

import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional

from core.logging_manager import get_logger

logger = get_logger("vector_memory.embeddings", "cyan")


class BaseEmbedding(ABC):
    """Embedding 基类"""

    @abstractmethod
    async def generate(self, text: str) -> List[float]:
        """生成文本的 Embedding 向量"""
        pass

    @abstractmethod
    async def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成 Embedding"""
        pass


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI Embedding 实现"""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = None

    async def _get_client(self):
        """懒加载 OpenAI 客户端"""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    async def generate(self, text: str) -> List[float]:
        """生成单个文本的 Embedding"""
        try:
            client = await self._get_client()
            response = await client.embeddings.create(model=self.model, input=text)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI Embedding 生成失败: {e}")
            raise

    async def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成 Embedding"""
        try:
            client = await self._get_client()
            response = await client.embeddings.create(model=self.model, input=texts)
            # 按索引排序返回
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]
        except Exception as e:
            logger.error(f"OpenAI 批量 Embedding 生成失败: {e}")
            raise


class ZhipuEmbedding(BaseEmbedding):
    """智谱 AI Embedding 实现"""

    def __init__(self, api_key: str, model: str = "embedding-3"):
        self.api_key = api_key
        self.model = model

    async def generate(self, text: str) -> List[float]:
        """生成单个文本的 Embedding"""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://open.bigmodel.cn/api/paas/v4/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": text},
            )
            result = response.json()
            return result["data"][0]["embedding"]

    async def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成（智谱暂不支持批量，逐个调用）"""
        results = []
        for text in texts:
            embedding = await self.generate(text)
            results.append(embedding)
        return results


class EmbeddingFactory:
    """Embedding 工厂"""

    @staticmethod
    def create(
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> BaseEmbedding:
        """
        创建 Embedding 实例

        Args:
            provider: 提供者类型 (openai/zhipu)
            model: 模型名称
            api_key: API Key（OpenAI/智谱需要）
            base_url: API 地址（OpenAI 可选）

        Returns:
            BaseEmbedding 实例
        """
        if provider == "openai":
            if not api_key:
                raise ValueError("OpenAI Embedding 需要 API Key")
            return OpenAIEmbedding(
                api_key=api_key,
                model=model,
                base_url=base_url or "https://api.openai.com/v1",
            )
        elif provider == "zhipu":
            if not api_key:
                raise ValueError("智谱 Embedding 需要 API Key")
            return ZhipuEmbedding(api_key=api_key, model=model)
        else:
            raise ValueError(f"不支持的 Embedding 提供者: {provider}")
