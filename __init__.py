"""
向量记忆库插件 v2.0.0
提供基于语义相似度的长期记忆检索能力

新增功能：
- 分层记忆作用域
- 数据契约与旧 metadata 迁移
- 自动上下文注入与每日记忆反思
- 诊断、导出、重建索引和安全集合切换工具
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

__version__ = "2.0.0"
