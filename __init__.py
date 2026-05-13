"""
向量记忆库插件 v1.1
提供基于语义相似度的长期记忆检索能力

新增功能：
- 记忆重要性评分
- 自动上下文注入
- 每日记忆反思
- 智能消息过滤
"""

from .main import VectorMemoryPlugin
from .embeddings import EmbeddingFactory, BaseEmbedding, OpenAIEmbedding
from .vector_store import ChromaVectorStore, VectorStore
from .importance import (
    ImportanceScorerFactory,
    BaseImportanceScorer,
    RuleBasedScorer,
    LLMBasedScorer,
    HybridScorer,
)
from .filter import MessageFilter
from .injector import ContextInjector, InjectionResult
from .reflection import ReflectionManager

__all__ = [
    # 主插件
    "VectorMemoryPlugin",
    # Embedding
    "EmbeddingFactory",
    "BaseEmbedding",
    "OpenAIEmbedding",
    # 向量存储
    "ChromaVectorStore",
    "VectorStore",
    # 重要性评分
    "ImportanceScorerFactory",
    "BaseImportanceScorer",
    "RuleBasedScorer",
    "LLMBasedScorer",
    "HybridScorer",
    # 消息过滤
    "MessageFilter",
    # 上下文注入
    "ContextInjector",
    "InjectionResult",
    # 记忆反思
    "ReflectionManager",
]

__version__ = "1.1.0"
