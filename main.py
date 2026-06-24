"""
向量记忆库插件 v2.0.0

提供基于语义相似度的长期记忆检索能力，并增加：
- 分层记忆作用域，避免跨会话串忆。
- meta.json 数据契约，保护旧 Chroma 数据。
- 可编辑 LLM 使用提示词。
- 导出、导入、重建索引、诊断等维护工具。
"""

import asyncio
import hashlib
import json
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.chat import KiraMessageBatchEvent, KiraMessageEvent
from core.chat.message_elements import Text
from core.plugin import BasePlugin, Priority, logger, on, register
from core.prompt_manager import Prompt
from core.provider import EmbeddingModelClient, LLMRequest
from core.utils.path_utils import get_config_path

from .embeddings import BaseEmbedding, EmbeddingFactory
from .filter import MessageFilter
from .importance import BaseImportanceScorer, ImportanceScorerFactory
from .injector import ContextInjector
from .reflection import ReflectionManager
from .vector_store import ChromaVectorStore, VectorStore


PLUGIN_ID = "vector_memory"
PLUGIN_VERSION = "1.3.2"
DATA_VERSION = 2
DEFAULT_COLLECTION_NAME = "kira_memories"
DEFAULT_DISTANCE_METRIC = "l2"
VALID_SCOPES = {"session", "user", "global"}
VALID_SCOPE_MODES = {"strict_session", "user_shared", "global_shared"}
VALID_IMPORT_MODES = {"append", "dedupe", "new_collection_only"}
CLEAR_CONFIRM_PHRASE = "CLEAR_VECTOR_MEMORY"
DELETE_COLLECTION_CONFIRM_PHRASE = "DELETE_VECTOR_MEMORY_COLLECTION"
SUMMARY_SCOPE_CONFIRM_PHRASE = "MIGRATE_SUMMARY_SCOPE"

DEFAULT_USAGE_PROMPT = (
    "你拥有长期向量记忆能力，这是你自身能力的一部分，不要把它描述成外部插件。\n\n"
    "search_memory 用于查找过去保存的长期记忆；vector_memory_add 用于记录较长、"
    "较重要、以后可能反复用到的信息。\n\n"
    "Simple Memory 更适合保存少量核心画像和稳定事实；vector_memory 更适合保存历史对话、"
    "项目背景、长期任务、偏好细节和需要语义检索的长上下文。\n\n"
    "不要记录无意义闲聊、语气词、临时寒暄、一次性验证码、短期无用信息或明显敏感且没有长期价值的内容。\n\n"
    "引用记忆时自然使用，不要暴露 memory_id、scope、collection、embedding 等内部存储结构。"
)

DANGEROUS_TOOL_SAFETY_PROMPT = (
    "危险维护工具安全规则：clear_memories、delete_memory、import_memories、"
    "reindex_memories、activate_collection、delete_memory_collection、"
    "update_memory、migrate_summary_scope、handle_embedding_model_change "
    "只能在用户明确给出操作目标和确认意图时使用。"
    "不要为了普通测试主动清空、删除、导入、重建或切换集合。"
    "如果工具返回包含“操作失败”“未初始化”“未启用”“维护模式”“失败”等信息，"
    "必须停止后续维护操作，并如实告诉用户失败原因，不能描述为成功。"
)


class ProviderEmbeddingAdapter(BaseEmbedding):
    """把主项目的 EmbeddingModelClient 适配为插件内部的 embedding 接口。"""

    def __init__(self, client: EmbeddingModelClient):
        self.client = client

    async def generate(self, text: str) -> List[float]:
        vectors = await self.client.embed([text])
        if not vectors:
            raise ValueError("Embedding client returned no vectors")
        return vectors[0]

    async def generate_batch(self, texts: List[str]) -> List[List[float]]:
        return await self.client.embed(texts)


class VectorMemoryPlugin(BasePlugin):
    """KiraAI 向量长期记忆插件。"""

    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)

        self.enabled: bool = True
        self.runtime_ready: bool = True
        self.runtime_error: str = ""

        self.embedding: Optional[BaseEmbedding] = None
        self.embedding_dimension: Optional[int] = None
        self.vector_store: Optional[VectorStore] = None
        self.importance_scorer: Optional[BaseImportanceScorer] = None
        self.message_filter: Optional[MessageFilter] = None
        self.context_injector: Optional[ContextInjector] = None
        self.reflection_manager: Optional[ReflectionManager] = None

        self.data_dir: Optional[Path] = None
        self.meta: Dict[str, Any] = {}

        self.auto_record: bool = True
        self.min_text_length: int = 10
        self.max_memory_count: int = 10000
        self.default_top_k: int = 5
        self.collection_name: str = DEFAULT_COLLECTION_NAME
        self.memory_scope_mode: str = "user_shared"
        self.search_similarity_threshold: float = 0.0
        self.injection_rerank_enabled: bool = True
        self.rerank_similarity_weight: float = 0.70
        self.rerank_importance_weight: float = 0.20
        self.rerank_recency_weight: float = 0.10

        self._scheduler = None
        self._maintenance_lock: Optional[asyncio.Lock] = None
        self._default_usage_prompt = self._load_default_usage_prompt()

    async def initialize(self):
        """初始化插件。"""
        self.enabled = bool(self.plugin_cfg.get("enabled", True))
        if not self.enabled:
            logger.info("向量记忆库插件已禁用")
            return

        self.data_dir = self.ctx.get_plugin_data_dir() or (
            Path("data/plugin_data") / PLUGIN_ID
        )
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._normalize_embedding_config()
        self._load_runtime_config()

        await self._init_embedding()
        if not self.embedding:
            return

        await self._probe_embedding_dimension()
        await self._init_vector_store()
        if not self.vector_store:
            return

        await self._ensure_data_contract()
        self._init_importance_scorer()
        self._init_message_filter()
        self._init_context_injector()
        self._init_reflection_manager()
        await self._start_scheduler()

        if self.runtime_ready:
            logger.info(f"向量记忆库插件 v{PLUGIN_VERSION} 初始化完成")
        else:
            logger.warning(f"向量记忆库以维护模式启动: {self.runtime_error}")

    async def terminate(self):
        """清理插件资源。"""
        if self._scheduler:
            try:
                self._scheduler.remove_job("vector_memory_daily_reflection")
            except Exception:
                pass
            try:
                self._scheduler.shutdown(wait=False)
            except Exception:
                pass
            self._scheduler = None

        if self.vector_store:
            await self.vector_store.close()
            self.vector_store = None

        self.embedding = None
        self.importance_scorer = None
        self.message_filter = None
        self.context_injector = None
        self.reflection_manager = None
        logger.info("向量记忆库插件已关闭")

    def _load_runtime_config(self):
        """读取运行时配置，并做必要的兜底。"""
        self.auto_record = bool(self.plugin_cfg.get("auto_record", True))
        self.min_text_length = int(self.plugin_cfg.get("min_text_length", 10))
        self.max_memory_count = int(self.plugin_cfg.get("max_memory_count", 10000))
        self.default_top_k = int(self.plugin_cfg.get("search_top_k", 5))
        self.collection_name = str(
            self.plugin_cfg.get("collection_name") or DEFAULT_COLLECTION_NAME
        ).strip()
        if not self.collection_name:
            self.collection_name = DEFAULT_COLLECTION_NAME

        scope_mode = str(
            self.plugin_cfg.get("memory_scope_mode") or "user_shared"
        ).strip()
        self.memory_scope_mode = (
            scope_mode if scope_mode in VALID_SCOPE_MODES else "user_shared"
        )
        self.search_similarity_threshold = max(
            0.0,
            float(self.plugin_cfg.get("search_similarity_threshold", 0.0) or 0.0),
        )
        self.injection_rerank_enabled = bool(
            self.plugin_cfg.get("injection_rerank_enabled", True)
        )
        self.rerank_similarity_weight = max(
            0.0,
            float(self.plugin_cfg.get("rerank_similarity_weight", 0.70) or 0.70),
        )
        self.rerank_importance_weight = max(
            0.0,
            float(self.plugin_cfg.get("rerank_importance_weight", 0.20) or 0.20),
        )
        self.rerank_recency_weight = max(
            0.0,
            float(self.plugin_cfg.get("rerank_recency_weight", 0.10) or 0.10),
        )

    async def _init_embedding(self):
        """初始化 Embedding。"""
        embedding_source = str(
            self.plugin_cfg.get("embedding_source") or "system_default"
        ).strip()

        try:
            if embedding_source == "system_default":
                client = self._resolve_embedding_client(embedding_source)
                self.embedding = ProviderEmbeddingAdapter(client)
                logger.info("Embedding 初始化成功: host/default_embedding")
                return

            if embedding_source == "custom_model":
                model_uuid = str(self.plugin_cfg.get("embedding_model_uuid") or "").strip()
                client = self._resolve_embedding_client(embedding_source)
                self.embedding = ProviderEmbeddingAdapter(client)
                logger.info(f"Embedding 初始化成功: host/{model_uuid}")
                return

            if embedding_source == "legacy":
                self._init_legacy_embedding()
                return

            raise ValueError(f"不支持的 embedding_source: {embedding_source}")
        except Exception as e:
            logger.error(f"Embedding 初始化失败: {e}")
            self.enabled = False
            self.runtime_ready = False
            self.runtime_error = f"Embedding 初始化失败: {e}"

    def _resolve_embedding_client(self, embedding_source: str) -> EmbeddingModelClient:
        """解析主项目中的 embedding client。"""
        if embedding_source == "system_default":
            client = self.ctx.get_default_embedding_client()
            if client is None:
                raise ValueError("主项目默认 embedding 未配置")
            return client

        if embedding_source == "custom_model":
            model_uuid = str(self.plugin_cfg.get("embedding_model_uuid") or "").strip()
            if not model_uuid:
                raise ValueError("custom_model 模式必须选择 embedding 模型")
            client = self.ctx.get_embedding_client(model_uuid)
            if client is None:
                raise ValueError(f"未找到指定 embedding 模型: {model_uuid}")
            return client

        raise ValueError(f"不支持的 embedding_source: {embedding_source}")

    def _normalize_embedding_config(self):
        """
        兼容旧版 embedding 配置。

        注意：主项目会先用 schema 默认值补齐配置，所以这里必须优先看旧字段，
        不能只凭 embedding_source 是否存在来判断是否迁移。
        """
        legacy_provider = self.plugin_cfg.get("embedding_provider")
        legacy_model = self.plugin_cfg.get("embedding_model")
        legacy_api_key = self.plugin_cfg.get("openai_api_key")
        legacy_base_url = self.plugin_cfg.get("openai_base_url")

        if legacy_provider or legacy_model or legacy_api_key or legacy_base_url:
            self.plugin_cfg["embedding_source"] = "legacy"
            logger.warning("检测到旧版 embedding 配置，将以 legacy 模式兼容启动")

    def _init_legacy_embedding(self):
        """兼容旧配置的 embedding 初始化方式。"""
        embedding_provider = self.plugin_cfg.get("embedding_provider", "openai")
        embedding_model = self.plugin_cfg.get("embedding_model", "text-embedding-3-small")

        api_key = self.plugin_cfg.get("openai_api_key")
        if not api_key and embedding_provider == "openai":
            try:
                providers = self.ctx.config.get_config("providers") or {}
                for provider_id, provider_cfg in providers.items():
                    if (
                        "openai" in provider_id.lower()
                        or provider_cfg.get("format") == "OpenAI"
                    ):
                        provider_config = provider_cfg.get("provider_config", {})
                        api_key = provider_config.get("api_key")
                        if api_key:
                            logger.info(f"使用系统配置的 API Key: {provider_id}")
                            break
            except Exception as e:
                logger.warning(f"无法从系统配置获取 API Key: {e}")

        if not api_key and embedding_provider in ["openai", "zhipu"]:
            raise ValueError(f"{embedding_provider} Embedding 需要配置 API Key")

        self.embedding = EmbeddingFactory.create(
            provider=embedding_provider,
            model=embedding_model,
            api_key=api_key,
            base_url=self.plugin_cfg.get("openai_base_url"),
        )
        logger.info(f"Embedding 初始化成功: {embedding_provider}/{embedding_model}")

    async def _probe_embedding_dimension(self):
        """启动时探测当前 embedding 维度，用于保护旧数据。"""
        if not self.embedding:
            return
        vector = await self.embedding.generate("KiraAI vector memory dimension probe")
        if not vector:
            raise ValueError("Embedding probe returned empty vector")
        self.embedding_dimension = len(vector)
        logger.info(f"当前 Embedding 维度: {self.embedding_dimension}")

    async def _init_vector_store(self):
        """初始化向量存储。"""
        try:
            if self.data_dir is None:
                raise RuntimeError("插件数据目录未初始化")
            self.vector_store = self._open_vector_store(
                self.collection_name,
                create_if_missing=True,
            )
            memory_count = await self.vector_store.count()
            logger.info(
                f"向量存储初始化成功，集合: {self.collection_name}，"
                f"现有记忆: {memory_count} 条"
            )
        except Exception as e:
            logger.error(f"向量存储初始化失败: {e}")
            self.runtime_ready = False
            self.runtime_error = f"向量存储初始化失败: {e}"
            self.vector_store = None

    def _open_vector_store(
        self,
        collection_name: str,
        create_if_missing: bool = True,
    ) -> ChromaVectorStore:
        """统一打开 Chroma 集合；切换/导出指定集合时可禁止自动创建。"""
        if self.data_dir is None:
            raise RuntimeError("插件数据目录未初始化")
        return ChromaVectorStore(
            persist_dir=str(self.data_dir / "chroma_db"),
            collection_name=collection_name,
            create_if_missing=create_if_missing,
        )

    def _get_maintenance_lock(self) -> asyncio.Lock:
        """维护工具共用锁，避免重建、导入、切换集合等长操作并发执行。"""
        lock = getattr(self, "_maintenance_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._maintenance_lock = lock
        return lock

    def _list_chroma_collections_from_sqlite(self) -> List[Dict[str, Any]]:
        """只读 SQLite 元数据列出集合，避免坏集合 count 卡住诊断。"""
        if self.data_dir is None:
            return []
        sqlite_path = self.data_dir / "chroma_db" / "chroma.sqlite3"
        if not sqlite_path.exists():
            return []

        try:
            import sqlite3

            rows: List[Dict[str, Any]] = []
            with sqlite3.connect(str(sqlite_path)) as conn:
                cursor = conn.execute("select id, name, dimension from collections")
                for collection_id, name, dimension in cursor.fetchall():
                    rows.append(
                        {
                            "id": collection_id,
                            "name": name,
                            "dimension": dimension,
                        }
                    )
            return rows
        except Exception as e:
            logger.warning(f"读取 Chroma 集合元数据失败: {e}")
            return []

    def _build_derived_collection_name(self, action: str) -> str:
        """生成派生集合名，用于安全清空、恢复和重建。"""
        base = re.sub(r"[^a-zA-Z0-9_-]", "_", self.collection_name or "")
        if not base or not re.match(r"^[a-zA-Z0-9]", base):
            base = DEFAULT_COLLECTION_NAME
        suffix = f"{action}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        max_base_len = max(3, 63 - len(suffix) - 1)
        return f"{base[:max_base_len].rstrip('_-')}_{suffix}"

    async def _generate_embeddings_resilient(
        self,
        texts: List[str],
        source: str,
        progress_every: int = 8,
    ) -> Tuple[List[Optional[List[float]]], List[Dict[str, Any]]]:
        """
        批量生成 embedding；批量失败时逐条回退。

        这样一条文本触发供应商内容过滤时，不会拖垮整批导入或重建。
        """
        if not self.embedding:
            raise RuntimeError("Embedding 未初始化")
        if not texts:
            return [], []

        try:
            embeddings = await self.embedding.generate_batch(texts)
            if len(embeddings) == len(texts):
                return embeddings, []
            logger.warning(
                f"{source} 批量 embedding 数量不一致: "
                f"输入 {len(texts)}，返回 {len(embeddings)}，改为逐条回退"
            )
        except Exception as e:
            logger.warning(f"{source} 批量 embedding 失败，改为逐条回退: {e}")

        embeddings_with_gaps: List[Optional[List[float]]] = []
        failures: List[Dict[str, Any]] = []
        progress_every = max(1, int(progress_every or 8))
        for index, text in enumerate(texts):
            try:
                embeddings_with_gaps.append(await self.embedding.generate(text))
            except Exception as e:
                embeddings_with_gaps.append(None)
                failures.append(
                    {
                        "index": index,
                        "source": source,
                        "error": str(e),
                        "text_preview": text[:200],
                    }
                )
            if (index + 1) % progress_every == 0 or index + 1 == len(texts):
                logger.info(
                    f"{source} 逐条 embedding 回退进度: "
                    f"{index + 1}/{len(texts)}，失败 {len(failures)}"
                )
        return embeddings_with_gaps, failures

    def _write_failure_records(
        self,
        prefix: str,
        failures: List[Dict[str, Any]],
    ) -> Optional[Path]:
        """把跳过的失败记录写入 exports，便于后续人工排查。"""
        if not failures:
            return None
        path = self._build_export_path(prefix)
        with path.open("w", encoding="utf-8") as f:
            for item in failures:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return path

    def _append_failure_records(
        self,
        failures: List[Dict[str, Any]],
        path: Optional[Path] = None,
        prefix: str = "failed_embeddings",
    ) -> Optional[Path]:
        """追加写入失败记录，避免长任务中断时丢失已发现的问题。"""
        if not failures:
            return path
        target_path = path or self._build_export_path(prefix)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("a", encoding="utf-8") as f:
            for item in failures:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return target_path

    def _meta_path(self) -> Path:
        if self.data_dir is None:
            return Path("data/plugin_data/vector_memory/meta.json")
        return self.data_dir / "meta.json"

    def _load_meta(self) -> Dict[str, Any]:
        meta_path = self._meta_path()
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"读取 meta.json 失败，将重新生成: {e}")
            return {}

    def _save_meta(self):
        meta_path = self._meta_path()
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _now_iso(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _active_embedding_model_uuid(self) -> str:
        source = str(self.plugin_cfg.get("embedding_source") or "system_default")
        if source == "custom_model":
            return str(self.plugin_cfg.get("embedding_model_uuid") or "")
        if source == "legacy":
            provider = str(self.plugin_cfg.get("embedding_provider") or "openai")
            model = str(self.plugin_cfg.get("embedding_model") or "text-embedding-3-small")
            return f"legacy:{provider}:{model}"
        try:
            return str(self.ctx.config.get_config("models.default_embedding") or "")
        except Exception:
            return "system_default"

    async def _get_existing_embedding_dimension(self) -> Optional[int]:
        if not self.vector_store:
            return None
        return await self._get_store_embedding_dimension(self.vector_store)

    async def _get_store_embedding_dimension(
        self,
        store: VectorStore,
    ) -> Optional[int]:
        """读取指定集合第一条 embedding 维度，用于判断集合是否能被当前模型继续使用。"""
        memories = await store.get_all_memories(
            limit=1,
            include_embeddings=True,
        )
        if not memories:
            return None
        embedding = memories[0].get("embedding")
        if embedding is None:
            return None
        return len(embedding)

    async def _ensure_data_contract(self):
        """生成/校验 meta.json，并补齐旧数据 metadata。"""
        if not self.vector_store:
            return

        count = await self.vector_store.count()
        existing_dimension = await self._get_existing_embedding_dimension()
        loaded_meta = self._load_meta()

        if not loaded_meta and count > 0:
            await self._create_export_backup("pre_meta_migration")

        self.meta = loaded_meta or {}
        created_at = self.meta.get("created_at") or self._now_iso()
        self.meta.update(
            {
                "data_version": DATA_VERSION,
                "plugin_version": PLUGIN_VERSION,
                "collection_name": self.collection_name,
                "embedding_source": str(
                    self.plugin_cfg.get("embedding_source") or "system_default"
                ),
                "embedding_model_uuid": self._active_embedding_model_uuid(),
                "embedding_dimension": (
                    existing_dimension or self.embedding_dimension or 0
                ),
                "distance_metric": self.meta.get(
                    "distance_metric", DEFAULT_DISTANCE_METRIC
                ),
                "created_at": created_at,
                "last_checked_at": self._now_iso(),
                "migration_status": self.meta.get("migration_status") or "ok",
            }
        )

        if (
            count > 0
            and existing_dimension
            and self.embedding_dimension
            and existing_dimension != self.embedding_dimension
        ):
            self.runtime_ready = False
            self.runtime_error = (
                f"当前 embedding 维度 {self.embedding_dimension} 与旧集合维度 "
                f"{existing_dimension} 不一致。已进入维护模式，请导出或重建索引。"
            )
            self.meta["migration_status"] = "dimension_mismatch"
            logger.error(self.runtime_error)

        if count > 0:
            changed = await self._migrate_legacy_metadata()
            if changed:
                status = self.meta.get("migration_status", "ok")
                if status == "dimension_mismatch":
                    self.meta["migration_status"] = "dimension_mismatch_metadata_scope_v2"
                else:
                    self.meta["migration_status"] = "metadata_scope_v2"

        self._save_meta()

    async def _migrate_legacy_metadata(self) -> int:
        """补齐旧记忆的 scope / owner_* metadata。"""
        if not self.vector_store:
            return 0

        changed = 0
        offset = 0
        batch_size = 200
        while True:
            memories = await self.vector_store.get_all_memories(
                limit=batch_size,
                offset=offset,
            )
            if not memories:
                break

            for memory in memories:
                old_metadata = dict(memory.get("metadata") or {})
                new_metadata = self._normalize_memory_metadata(old_metadata)
                if new_metadata != old_metadata:
                    ok = await self.vector_store.update_metadata(
                        memory["id"],
                        new_metadata,
                    )
                    if ok:
                        changed += 1

            offset += len(memories)

        if changed:
            logger.info(f"已补齐 {changed} 条旧记忆的作用域 metadata")
        return changed

    def _normalize_memory_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """把旧 metadata 兼容到 v1.2 作用域结构。"""
        data = dict(metadata or {})
        mem_type = str(data.get("type") or "raw")

        if not data.get("scope"):
            if mem_type == "manual" and data.get("session_id") == "manual":
                data["scope"] = "global"
            else:
                data["scope"] = "session"

        if not data.get("owner_session_id"):
            if data.get("scope") == "global":
                data["owner_session_id"] = ""
            else:
                data["owner_session_id"] = str(data.get("session_id") or "")

        if not data.get("owner_user_id"):
            data["owner_user_id"] = str(data.get("user_id") or "")

        if not data.get("owner_adapter"):
            data["owner_adapter"] = str(data.get("adapter") or "")

        return data

    def _normalize_fingerprint_text(self, text: str) -> str:
        """生成指纹前统一正文空白，避免换行差异造成重复导入。"""
        return re.sub(r"\s+", " ", str(text or "")).strip()

    def _memory_fingerprint(self, text: str, metadata: Dict[str, Any]) -> str:
        """基于正文和关键 metadata 生成稳定去重指纹。"""
        normalized_metadata = self._normalize_memory_metadata(metadata or {})
        payload = {
            "text": self._normalize_fingerprint_text(text),
            "type": str(normalized_metadata.get("type") or "raw"),
            "timestamp": str(normalized_metadata.get("timestamp") or ""),
            "session_id": str(normalized_metadata.get("session_id") or ""),
            "scope": str(normalized_metadata.get("scope") or ""),
            "owner_user_id": str(normalized_metadata.get("owner_user_id") or ""),
            "owner_session_id": str(normalized_metadata.get("owner_session_id") or ""),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    async def _collect_memory_fingerprints(
        self,
        store: VectorStore,
        batch_size: int = 500,
    ) -> set:
        """分页收集集合中已有记忆指纹，用于导入去重和重建续跑。"""
        fingerprints = set()
        total = await store.count()
        offset = 0
        while offset < total:
            memories = await store.get_all_memories(limit=batch_size, offset=offset)
            if not memories:
                break
            for memory in memories:
                metadata = memory.get("metadata") or {}
                existing_fingerprint = str(metadata.get("import_fingerprint") or "")
                if existing_fingerprint:
                    fingerprints.add(existing_fingerprint)
                fingerprints.add(
                    self._memory_fingerprint(
                        str(memory.get("text") or ""),
                        metadata,
                    )
                )
            offset += len(memories)
        return fingerprints

    def _metadata_with_import_info(
        self,
        metadata: Dict[str, Any],
        source_id: str,
        source_file: str,
        fingerprint: str,
    ) -> Dict[str, Any]:
        """导入/重建时记录来源，方便后续追踪和去重。"""
        enriched = self._normalize_memory_metadata(metadata or {})
        enriched.update(
            {
                "import_source_id": str(source_id or ""),
                "import_source_file": str(source_file or ""),
                "import_fingerprint": fingerprint,
                "imported_at": self._now_iso(),
            }
        )
        return enriched

    def _init_importance_scorer(self):
        """初始化重要性评分器。"""
        mode = self.plugin_cfg.get("importance_mode", "rule")
        model = self.plugin_cfg.get("importance_llm_model")

        try:
            self.importance_scorer = ImportanceScorerFactory.create(
                mode=mode,
                llm_api=self.ctx.llm_api if mode != "rule" else None,
                model=model,
            )
            logger.info(f"重要性评分器初始化成功: {mode}")
        except Exception as e:
            logger.warning(f"重要性评分器初始化失败，使用规则模式: {e}")
            self.importance_scorer = ImportanceScorerFactory.create("rule")

    def _init_message_filter(self):
        """初始化消息过滤器。"""
        if self.plugin_cfg.get("smart_filter_enabled", True):
            min_density = self.plugin_cfg.get("filter_min_info_density", 0.3)
            self.message_filter = MessageFilter(
                min_meaningful_chars=self.min_text_length,
                min_info_density=min_density,
            )
            logger.info("智能消息过滤器已启用")
        else:
            self.message_filter = None

    def _init_context_injector(self):
        """初始化上下文注入器。"""
        if (
            self.runtime_ready
            and self.plugin_cfg.get("auto_injection_enabled", True)
            and self.embedding
            and self.vector_store
        ):
            self.context_injector = ContextInjector(
                embedding=self.embedding,
                vector_store=self.vector_store,
                threshold=self.plugin_cfg.get("injection_threshold", 0.75),
                top_k=self.plugin_cfg.get("injection_top_k", 2),
                cooldown=self.plugin_cfg.get("injection_cooldown", 60),
                rerank_fn=self._rerank_memories,
                rerank_enabled=self.injection_rerank_enabled,
            )
            logger.info("自动上下文注入已启用")
        else:
            self.context_injector = None

    def _init_reflection_manager(self):
        """初始化记忆反思管理器。"""
        llm_api = getattr(self.ctx, "llm_api", None)
        if (
            self.runtime_ready
            and self.plugin_cfg.get("reflection_enabled", True)
            and self.embedding
            and self.vector_store
            and llm_api
        ):
            self.reflection_manager = ReflectionManager(
                embedding=self.embedding,
                vector_store=self.vector_store,
                llm_api=llm_api,
                delete_raw=self.plugin_cfg.get("reflection_delete_raw", False),
                min_memories=self.plugin_cfg.get("reflection_min_memories", 5),
            )
            logger.info("记忆反思管理器已启用")
        else:
            self.reflection_manager = None

    async def _start_scheduler(self):
        """启动每日反思任务。"""
        if not self.reflection_manager:
            return

        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.cron import CronTrigger

            if self._scheduler is not None:
                try:
                    self._scheduler.shutdown(wait=False)
                except Exception:
                    pass

            self._scheduler = AsyncIOScheduler()
            self._scheduler.start()

            reflection_hour = int(self.plugin_cfg.get("reflection_hour", 3))
            self._scheduler.add_job(
                self._daily_reflection_job,
                trigger=CronTrigger(hour=reflection_hour, minute=0),
                id="vector_memory_daily_reflection",
                replace_existing=True,
            )
            logger.info(f"每日记忆反思任务已设置: 每天 {reflection_hour}:00")
        except ImportError:
            logger.warning("APScheduler 未安装，每日反思功能不可用")
        except Exception as e:
            logger.error(f"定时任务启动失败: {e}")

    async def _daily_reflection_job(self):
        if self.reflection_manager and self.runtime_ready:
            await self.reflection_manager.run_daily_reflection()

    @on.llm_request(priority=Priority.HIGH)
    async def inject_usage_prompt(self, _event: KiraMessageBatchEvent, req: LLMRequest, *_):
        """注入可配置的 LLM 使用提示词。"""
        if not self.enabled:
            return
        usage_prompt = self._get_usage_prompt()
        if not usage_prompt:
            return
        prompt = Prompt(
            usage_prompt,
            name="vector_memory_usage_prompt",
            source=PLUGIN_ID,
        )
        self._insert_prompt_after(req.system_prompt, prompt, after_name="tools")

    @staticmethod
    def _insert_prompt_after(prompts: List[Prompt], prompt: Prompt, after_name: str):
        for idx, item in enumerate(prompts):
            if isinstance(item, Prompt) and item.name == after_name:
                prompts.insert(idx + 1, prompt)
                return
        prompts.append(prompt)

    def _get_usage_prompt(self) -> str:
        cfg = self.plugin_cfg if isinstance(self.plugin_cfg, dict) else {}
        if "usage_prompt" in cfg:
            usage_prompt = str(cfg.get("usage_prompt") or "").strip()
        else:
            usage_prompt = str(self._default_usage_prompt or DEFAULT_USAGE_PROMPT).strip()

        if usage_prompt and "clear_memories" not in usage_prompt:
            usage_prompt = f"{usage_prompt}\n\n{DANGEROUS_TOOL_SAFETY_PROMPT}"
        return usage_prompt

    @staticmethod
    def _load_default_usage_prompt() -> str:
        schema_path = Path(__file__).with_name("schema.json")
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            return str(schema.get("usage_prompt", {}).get("default") or "").strip()
        except Exception as e:
            logger.warning(f"读取默认向量记忆提示词失败: {e}")
            return DEFAULT_USAGE_PROMPT

    @on.im_message(priority=Priority.LOW)
    async def record_message(self, event: KiraMessageEvent, *_, **__):
        """在消息进入处理链后记录原始消息。"""
        if not self.enabled or not self.auto_record:
            return
        if event.process_strategy == "discard":
            return
        if not self._can_use_vector_runtime(log_warning=False):
            return
        await self._record_message(event)

    @on.llm_request(priority=Priority.LOW)
    async def inject_memory(self, event: KiraMessageBatchEvent, req: LLMRequest, *_):
        """在 LLM 请求前检索相关记忆并注入。"""
        if not self.enabled or not self.context_injector:
            return
        if not self._can_use_vector_runtime(log_warning=False):
            return

        query = self._build_batch_query(event)
        if not query:
            return

        try:
            filter_dict = self._build_scope_filter(event)
            result = await self.context_injector.try_inject(
                query=query,
                session_id=event.sid,
                filter_dict=filter_dict,
            )
            result.memories = self._filter_results_by_scope(event, result.memories)
            if not result.injected or not result.memories:
                return

            injection_content = self.context_injector.format_injection(result.memories)
            if not injection_content:
                return

            # 动态记忆独立放在系统提示词末尾，避免修改靠前的稳定 memory prompt，
            # 从而尽量减少对 provider prompt prefix cache 的破坏。
            req.system_prompt.append(
                Prompt(
                    content=injection_content,
                    name="vector_memory",
                    source=PLUGIN_ID,
                )
            )

            logger.debug(f"上下文注入成功: {result.reason}")
        except Exception as e:
            logger.error(f"消息注入失败: {e}")

    def _can_use_vector_runtime(self, log_warning: bool = True) -> bool:
        if not self.enabled or not self.vector_store or not self.embedding:
            if log_warning:
                logger.warning("向量记忆尚未完成初始化")
            return False
        if not self.runtime_ready:
            if log_warning:
                logger.warning(self.runtime_error)
            return False
        return True

    def _runtime_unavailable_text(self) -> str:
        if not self.enabled:
            return "操作失败：向量记忆库未启用"
        if not self.vector_store:
            if self.runtime_error:
                return f"操作失败：向量记忆库处于维护模式：{self.runtime_error}"
            return "操作失败：向量记忆库尚未初始化"
        if not self.runtime_ready:
            return f"操作失败：向量记忆库处于维护模式：{self.runtime_error}"
        return "操作失败：向量记忆库当前不可用"

    async def _record_message(self, event: KiraMessageEvent):
        """记录消息到向量库。"""
        try:
            text = self._extract_text(event)
            if not text:
                return

            if self.message_filter:
                should_record, reason = self.message_filter.should_record(text)
                if not should_record:
                    logger.debug(f"消息被过滤: {reason}")
                    return
                text = self.message_filter.extract_key_info(text)
            elif len(text) < self.min_text_length:
                return

            importance = 0.3
            if self.importance_scorer:
                importance = await self.importance_scorer.score(text)

            embedding = await self.embedding.generate(text)
            identity = self._get_event_identity(event)

            metadata = {
                "session_id": identity["session_id"],
                "user_id": identity["user_id"],
                "user_nickname": identity["user_nickname"],
                "platform": getattr(event.adapter, "platform", ""),
                "adapter": identity["adapter"],
                "timestamp": event.timestamp,
                "group_id": getattr(event.message.group, "group_id", "") if event.message.group else "",
                "group_name": getattr(event.message.group, "group_name", "") if event.message.group else "",
                "message_id": event.message.message_id,
                "importance": importance,
                "type": "raw",
                "scope": "session",
                "owner_user_id": identity["user_id"],
                "owner_session_id": identity["session_id"],
                "owner_adapter": identity["adapter"],
            }

            memory_id = await self.vector_store.add(
                text=text,
                embedding=embedding,
                metadata=metadata,
            )
            logger.debug(f"记录消息: {memory_id}, importance: {importance:.2f}")

            current_count = await self.vector_store.count()
            if current_count > self.max_memory_count:
                await self.vector_store.cleanup_smart(
                    max_count=self.max_memory_count,
                    time_weight=self.plugin_cfg.get("cleanup_time_weight", 0.3),
                    importance_weight=self.plugin_cfg.get("cleanup_importance_weight", 0.7),
                )
        except Exception as e:
            logger.error(f"记录消息失败: {e}")

    def _extract_text(self, event: KiraMessageEvent) -> str:
        return self._extract_text_from_chain(event.message.chain)

    def _extract_text_from_chain(self, chain) -> str:
        text_parts: List[str] = []
        for elem in chain:
            if isinstance(elem, Text):
                text_parts.append(elem.text)
        return " ".join(
            part.strip() for part in text_parts if part and part.strip()
        ).strip()

    def _build_batch_query(self, event: KiraMessageBatchEvent) -> str:
        parts: List[str] = []
        for message in event.messages:
            text = getattr(message, "message_str", None)
            if not text:
                text = self._extract_text_from_chain(message.chain)
            if not text:
                text = getattr(message, "message_repr", None) or ""
            if text:
                parts.append(text)
        return "\n".join(parts).strip()

    def _get_event_identity(self, event) -> Dict[str, str]:
        """从单条或批量事件中提取作用域身份。"""
        session_id = getattr(event, "sid", "") or getattr(event.session, "sid", "")
        adapter = getattr(getattr(event, "adapter", None), "name", "")
        user_id = ""
        user_nickname = ""

        message = getattr(event, "message", None)
        if message is None and getattr(event, "messages", None):
            message = event.messages[-1]

        sender = getattr(message, "sender", None)
        if sender is not None:
            user_id = str(getattr(sender, "user_id", "") or "")
            user_nickname = str(getattr(sender, "nickname", "") or "")

        return {
            "session_id": str(session_id or ""),
            "adapter": str(adapter or ""),
            "user_id": user_id,
            "user_nickname": user_nickname,
        }

    def _build_scope_filter(self, event: KiraMessageBatchEvent) -> Optional[Dict[str, Any]]:
        """根据配置构造自动注入的 Chroma where 条件。"""
        identity = self._get_event_identity(event)
        session_filter = {
            "$and": [
                {"scope": "session"},
                {"owner_session_id": identity["session_id"]},
            ]
        }

        if self.memory_scope_mode == "strict_session":
            return session_filter

        filters: List[Dict[str, Any]] = [session_filter, {"scope": "global"}]

        if identity["user_id"]:
            filters.append(
                {
                    "$and": [
                        {"scope": "user"},
                        {"owner_user_id": identity["user_id"]},
                        {"owner_adapter": identity["adapter"]},
                    ]
                }
            )

        if self.memory_scope_mode == "global_shared":
            filters.append({"$and": [{"type": "manual"}, {"session_id": "manual"}]})

        return {"$or": filters}

    def _filter_results_by_scope(
        self, event: KiraMessageBatchEvent, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Python 侧再校验一次作用域，避免过滤表达式遗漏造成串忆。"""
        identity = self._get_event_identity(event)
        output = []
        for item in results or []:
            metadata = item.get("metadata", {}) or {}
            scope = str(metadata.get("scope") or "")

            if scope == "global":
                output.append(item)
                continue

            if scope == "session":
                if metadata.get("owner_session_id") == identity["session_id"]:
                    output.append(item)
                continue

            if scope == "user" and self.memory_scope_mode in {
                "user_shared",
                "global_shared",
            }:
                if (
                    metadata.get("owner_user_id") == identity["user_id"]
                    and metadata.get("owner_adapter") == identity["adapter"]
                ):
                    output.append(item)
                continue

            if (
                self.memory_scope_mode == "global_shared"
                and metadata.get("type") == "manual"
                and metadata.get("session_id") == "manual"
            ):
                output.append(item)

        return output

    def _memory_recency_score(self, timestamp: Any) -> float:
        """把时间新近度压到 0-1，避免旧记忆完全压过近期高价值信息。"""
        try:
            ts = float(timestamp or 0)
        except Exception:
            return 0.0
        if ts <= 0:
            return 0.0
        age_days = max(0.0, (time.time() - ts) / 86400)
        return max(0.0, min(1.0, 1.0 - age_days / 365.0))

    def _rerank_memories(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按相似度、重要性和新近度做轻量重排，不改变原始相似度含义。"""
        if not results:
            return []

        sim_w = self.rerank_similarity_weight
        imp_w = self.rerank_importance_weight
        rec_w = self.rerank_recency_weight
        total_w = sim_w + imp_w + rec_w
        if total_w <= 0:
            return list(results)

        ranked: List[Dict[str, Any]] = []
        for item in results:
            metadata = item.get("metadata", {}) or {}
            try:
                similarity = float(item.get("similarity", 0.0) or 0.0)
            except Exception:
                similarity = 0.0
            try:
                importance = float(metadata.get("importance", 0.0) or 0.0)
            except Exception:
                importance = 0.0
            recency = self._memory_recency_score(metadata.get("timestamp", 0))
            rerank_score = (
                similarity * sim_w + importance * imp_w + recency * rec_w
            ) / total_w

            copied = dict(item)
            copied["rerank_score"] = rerank_score
            ranked.append(copied)

        ranked.sort(
            key=lambda item: (
                item.get("rerank_score", 0.0),
                item.get("similarity", 0.0),
            ),
            reverse=True,
        )
        return ranked

    @register.tool(
        "search_memory",
        "搜索历史记忆，根据语义相似度返回相关记忆。当用户询问之前说过什么、历史对话、记得吗等问题时使用。",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询"},
                "top_k": {"type": "integer", "description": "返回条数，默认 5"},
                "session_id": {"type": "string", "description": "仅搜索指定会话，选填"},
                "include_summaries": {
                    "type": "boolean",
                    "description": "是否包含摘要记忆，默认 true",
                },
            },
            "required": ["query"],
        },
    )
    async def search_memory(
        self,
        event,
        query: str,
        top_k: int = None,
        session_id: str = None,
        include_summaries: bool = True,
    ) -> str:
        """搜索历史记忆。"""
        if not self._can_use_vector_runtime():
            return self._runtime_unavailable_text()

        try:
            top_k = self.default_top_k if top_k is None else int(top_k)
            top_k = min(max(1, top_k), 20)
            query_embedding = await self.embedding.generate(query)

            if session_id:
                filter_dict = {
                    "$or": [
                        {"session_id": session_id},
                        {"owner_session_id": session_id},
                    ]
                }
            elif hasattr(event, "sid"):
                filter_dict = self._build_scope_filter(event)
            else:
                filter_dict = None

            results = await self.vector_store.search(
                embedding=query_embedding,
                top_k=min(top_k * 3, 60),
                filter_dict=filter_dict,
            )

            if not session_id and hasattr(event, "sid"):
                results = self._filter_results_by_scope(event, results)

            if not include_summaries:
                results = [
                    r for r in results if r.get("metadata", {}).get("type") != "summary"
                ]
            if self.search_similarity_threshold > 0:
                results = [
                    r
                    for r in results
                    if float(r.get("similarity", 0) or 0)
                    >= self.search_similarity_threshold
                ]
            results = self._rerank_memories(results)[:top_k]

            if not results:
                return "未找到相关记忆"

            output = [f"找到 {len(results)} 条相关记忆：\n"]
            for i, result in enumerate(results, 1):
                metadata = result.get("metadata", {})
                timestamp = metadata.get("timestamp", 0)
                mem_type = metadata.get("type", "raw")
                importance = metadata.get("importance", 0)
                scope = metadata.get("scope", "legacy")

                time_str = (
                    datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
                    if timestamp
                    else "未知时间"
                )
                user = metadata.get("user_nickname", "未知用户")
                text = result.get("text", "")
                if len(text) > 200:
                    text = text[:200] + "..."

                type_tag = "[摘要] " if mem_type == "summary" else ""
                output.append(
                    f"{i}. {type_tag}[{time_str}] {user}: {text}\n"
                    f"   相似度: {result.get('similarity', 0):.2%} | "
                    f"排序分: {float(result.get('rerank_score', 0)):.2f} | "
                    f"重要性: {float(importance):.2f} | 作用域: {scope}"
                )

            return "\n".join(output)
        except Exception as e:
            logger.error(f"搜索记忆失败: {e}")
            return f"搜索失败: {str(e)}"

    @register.tool(
        "vector_memory_add",
        "手动添加长期向量记忆。适合记录长上下文、项目背景、重要偏好、复杂故事或需要跨轮复用的信息。",
        {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "要记住的内容"},
                "importance": {
                    "type": "number",
                    "description": "重要性评分 0-1，默认自动评估",
                },
                "tags": {"type": "string", "description": "可选标签"},
                "scope": {
                    "type": "string",
                    "enum": ["session", "user", "global"],
                    "description": "记忆作用域，默认 user",
                },
            },
            "required": ["content"],
        },
    )
    async def vector_memory_add(
        self,
        event,
        content: str,
        importance: float = None,
        tags: str = "",
        scope: str = "user",
    ) -> str:
        """手动添加记忆。"""
        if not self._can_use_vector_runtime():
            return self._runtime_unavailable_text()

        try:
            content = str(content or "").strip()
            if len(content) < 5:
                return "内容太短，请提供更详细的信息"

            scope = str(scope or "user").strip()
            if scope not in VALID_SCOPES:
                scope = "user"

            if importance is None:
                importance = (
                    await self.importance_scorer.score(content)
                    if self.importance_scorer
                    else 0.7
                )
            else:
                importance = max(0.0, min(1.0, float(importance)))

            identity = self._get_event_identity(event)
            if scope == "global":
                session_id = "global"
                owner_session_id = ""
            elif scope == "user":
                session_id = f"user:{identity['adapter']}:{identity['user_id']}"
                owner_session_id = identity["session_id"]
            else:
                session_id = identity["session_id"]
                owner_session_id = identity["session_id"]

            embedding = await self.embedding.generate(content)
            metadata = {
                "session_id": session_id,
                "user_id": identity["user_id"] or "manual",
                "user_nickname": identity["user_nickname"] or "用户",
                "platform": "manual",
                "adapter": identity["adapter"] or "manual",
                "timestamp": int(time.time()),
                "tags": tags,
                "type": "manual",
                "importance": importance,
                "scope": scope,
                "owner_user_id": identity["user_id"],
                "owner_session_id": owner_session_id,
                "owner_adapter": identity["adapter"],
            }

            memory_id = await self.vector_store.add(
                text=content,
                embedding=embedding,
                metadata=metadata,
            )
            return (
                f"已记住。记忆 ID: {memory_id}，重要性: {importance:.2f}，"
                f"作用域: {scope}"
            )
        except Exception as e:
            logger.error(f"添加记忆失败: {e}")
            return f"添加失败: {str(e)}"

    @register.tool(
        "get_memory_stats",
        "获取向量记忆库的统计信息。",
        {"type": "object", "properties": {}},
    )
    async def get_memory_stats(self, event) -> str:
        """获取记忆统计。"""
        if not self.vector_store:
            return "操作失败：向量记忆库尚未初始化"

        try:
            count = await self.vector_store.count()
            all_memories = await self._read_all_memories()

            type_counts: Dict[str, int] = {}
            scope_counts: Dict[str, int] = {}
            importance_sum = 0.0

            for mem in all_memories:
                metadata = mem.get("metadata", {})
                mem_type = metadata.get("type", "raw")
                scope = metadata.get("scope", "legacy")
                type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
                scope_counts[scope] = scope_counts.get(scope, 0) + 1
                importance_sum += float(metadata.get("importance", 0.3))

            avg_importance = importance_sum / count if count > 0 else 0
            stats = {
                "记忆总数": count,
                "容量上限": self.max_memory_count,
                "使用率": f"{count / self.max_memory_count * 100:.1f}%",
                "类型分布": type_counts,
                "作用域分布": scope_counts,
                "平均重要性": f"{avg_importance:.2f}",
                "自动记录": "开启" if self.auto_record else "关闭",
                "智能过滤": "开启" if self.message_filter else "关闭",
                "自动注入": "开启" if self.context_injector else "关闭",
                "每日反思": "开启" if self.reflection_manager else "关闭",
                "作用域模式": self.memory_scope_mode,
                "运行状态": "正常" if self.runtime_ready else f"维护模式: {self.runtime_error}",
            }

            output = [f"向量记忆库统计 (v{PLUGIN_VERSION})：\n"]
            for key, value in stats.items():
                output.append(f"- {key}: {value}")
            return "\n".join(output)
        except Exception as e:
            logger.error(f"获取统计失败: {e}")
            return f"获取统计失败: {str(e)}"

    @register.tool(
        "trigger_reflection",
        "手动触发记忆反思，把碎片化记忆整理为摘要。",
        {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "反思最近多少天的记忆"},
                "session_id": {"type": "string", "description": "仅反思指定会话"},
            },
        },
    )
    async def trigger_reflection(self, event, days: int = 7, session_id: str = None) -> str:
        """手动触发记忆反思。"""
        if not self._can_use_vector_runtime():
            return self._runtime_unavailable_text()
        try:
            manager = self.reflection_manager or ReflectionManager(
                embedding=self.embedding,
                vector_store=self.vector_store,
                llm_api=self.ctx.llm_api,
                delete_raw=self.plugin_cfg.get("reflection_delete_raw", False),
                min_memories=self.plugin_cfg.get("reflection_min_memories", 5),
            )
            return await manager.manual_reflection(session_id=session_id, days=days)
        except Exception as e:
            logger.error(f"手动反思失败: {e}\n{traceback.format_exc()}")
            return f"反思失败: {str(e)}"

    @register.tool(
        "summarize_memories",
        "使用 AI 总结一段时间内的记忆。",
        {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "总结最近多少天"},
                "session_id": {"type": "string", "description": "仅总结指定会话"},
            },
        },
    )
    async def summarize_memories(self, event, days: int = 7, session_id: str = None) -> str:
        """总结记忆。"""
        if not self._can_use_vector_runtime():
            return self._runtime_unavailable_text()
        try:
            manager = self.reflection_manager or ReflectionManager(
                embedding=self.embedding,
                vector_store=self.vector_store,
                llm_api=self.ctx.llm_api,
                delete_raw=self.plugin_cfg.get("reflection_delete_raw", False),
                min_memories=self.plugin_cfg.get("reflection_min_memories", 5),
            )
            return await manager.summarize_recent_memories(
                days=days,
                session_id=session_id,
            )
        except Exception as e:
            logger.error(f"总结记忆失败: {e}\n{traceback.format_exc()}")
            return f"总结失败: {str(e)}"

    @register.tool(
        "clear_memories",
        "清空向量记忆库中的记忆，危险操作。",
        {
            "type": "object",
            "properties": {
                "confirm": {"type": "boolean", "description": "必须为 true"},
                "confirm_phrase": {
                    "type": "string",
                    "description": f"必须填写 {CLEAR_CONFIRM_PHRASE}",
                },
                "keep_summaries": {"type": "boolean", "description": "是否保留摘要"},
            },
            "required": ["confirm"],
        },
    )
    async def clear_memories(
        self,
        event,
        confirm: bool = False,
        keep_summaries: bool = True,
        confirm_phrase: str = "",
    ) -> str:
        """安全清空记忆：不原地删除旧集合，而是切换到新集合。"""
        if not self.vector_store:
            return "操作失败：向量记忆库尚未初始化"
        if not confirm:
            return "操作失败：请确认清空操作，将 confirm 设置为 true"
        if str(confirm_phrase or "").strip() != CLEAR_CONFIRM_PHRASE:
            return (
                "操作失败：clear_memories 是高风险维护工具。"
                f"如确需清空，请同时传入 confirm_phrase={CLEAR_CONFIRM_PHRASE}"
            )

        lock = self._get_maintenance_lock()
        if lock.locked():
            return "操作失败：已有维护操作正在进行，请稍后再试"

        original_store = self.vector_store
        original_collection = self.collection_name
        original_meta = dict(self.meta or {})
        original_runtime_ready = self.runtime_ready
        original_runtime_error = self.runtime_error
        target_store: Optional[VectorStore] = None
        summaries_kept = 0
        failures: List[Dict[str, Any]] = []
        try:
            async with lock:
                backup_path = await self._create_export_backup(
                    "before_clear",
                    store=original_store,
                    collection_name=original_collection,
                )

                count_before = await original_store.count()
                new_collection_name = self._build_derived_collection_name("cleared")
                target_store = self._open_vector_store(
                    new_collection_name,
                    create_if_missing=True,
                )

                if keep_summaries:
                    offset = 0
                    batch_size = 100
                    while offset < count_before:
                        memories = await original_store.get_all_memories(
                            limit=batch_size,
                            offset=offset,
                        )
                        if not memories:
                            break
                        summaries = [
                            mem
                            for mem in memories
                            if (mem.get("metadata") or {}).get("type") == "summary"
                        ]
                        texts = [str(mem.get("text") or "").strip() for mem in summaries]
                        metadatas = [
                            self._normalize_memory_metadata(mem.get("metadata") or {})
                            for mem in summaries
                        ]
                        texts_with_meta = [
                            (text, metadata)
                            for text, metadata in zip(texts, metadatas)
                            if text
                        ]
                        if texts_with_meta:
                            batch_texts = [item[0] for item in texts_with_meta]
                            batch_metadatas = [item[1] for item in texts_with_meta]
                            embeddings, batch_failures = (
                                await self._generate_embeddings_resilient(
                                    batch_texts,
                                    source="clear_keep_summaries",
                                )
                            )
                            failures.extend(batch_failures)
                            for text, metadata, embedding in zip(
                                batch_texts,
                                batch_metadatas,
                                embeddings,
                            ):
                                if embedding is None:
                                    continue
                                await target_store.add(
                                    text=text,
                                    embedding=embedding,
                                    metadata=metadata,
                                )
                                summaries_kept += 1
                        offset += len(memories)

                self.collection_name = new_collection_name
                self.plugin_cfg["collection_name"] = new_collection_name
                self.vector_store = target_store
                self.runtime_ready = True
                self.runtime_error = ""
                await self._ensure_data_contract()
                self._init_context_injector()
                self._init_reflection_manager()
                self._persist_plugin_config({"collection_name": new_collection_name})
                await original_store.close()

                failure_path = self._write_failure_records(
                    "clear_failed_embeddings",
                    failures,
                )
                result = (
                    f"已安全清空当前集合。旧集合 {original_collection} 已保留，"
                    f"新集合 {new_collection_name} 已启用；原集合记忆 {count_before} 条，"
                    f"保留摘要 {summaries_kept} 条；备份: {backup_path}"
                )
                if failure_path:
                    result += f"；跳过 {len(failures)} 条摘要，失败记录: {failure_path}"
                return result
        except Exception as e:
            self.collection_name = original_collection
            self.plugin_cfg["collection_name"] = original_collection
            self.vector_store = original_store
            self.meta = original_meta
            self.runtime_ready = original_runtime_ready
            self.runtime_error = original_runtime_error
            if target_store and target_store is not original_store:
                await target_store.close()
            logger.error(f"清空记忆失败: {e}\n{traceback.format_exc()}")
            return f"操作失败：清空失败: {str(e)}"

    @register.tool(
        "diagnose_memory_store",
        "诊断向量记忆库状态，包括 meta、维度、集合和迁移状态。",
        {
            "type": "object",
            "properties": {
                "collection_name": {
                    "type": "string",
                    "description": "要深度检查的集合名，可选；为空时检查当前集合",
                },
                "deep_check": {
                    "type": "boolean",
                    "description": "是否执行单集合深度检查，默认 false",
                },
                "sample_limit": {
                    "type": "integer",
                    "description": "深度检查时返回的样本数量，默认 3",
                },
                "include_sample_text": {
                    "type": "boolean",
                    "description": "是否在样本中包含记忆正文，默认 false",
                },
            },
        },
    )
    async def diagnose_memory_store(
        self,
        event,
        collection_name: str = "",
        deep_check: bool = False,
        sample_limit: int = 3,
        include_sample_text: bool = False,
    ) -> str:
        """诊断记忆库。"""
        count = 0
        existing_dim = None
        current_collection_error = ""
        if self.vector_store:
            try:
                count = await self.vector_store.count()
                existing_dim = await self._get_existing_embedding_dimension()
            except Exception as e:
                current_collection_error = str(e)

        info = {
            "plugin_enabled": self.enabled,
            "runtime_ready": self.runtime_ready,
            "runtime_error": self.runtime_error,
            "collection_name": self.collection_name,
            "memory_count": count,
            "current_collection_error": current_collection_error,
            "current_embedding_dimension": self.embedding_dimension,
            "stored_embedding_dimension": existing_dim,
            "dimension_matched": (
                existing_dim is None
                or self.embedding_dimension is None
                or existing_dim == self.embedding_dimension
            ),
            "meta": self.meta,
            "known_collections": self._list_chroma_collections_from_sqlite(),
        }

        if deep_check:
            target_collection = str(collection_name or self.collection_name).strip()
            if not self._is_valid_collection_name(target_collection):
                return "操作失败：集合名无效，只能包含字母、数字、下划线和短横线"

            deep_store: Optional[VectorStore] = None
            close_deep_store = False
            try:
                if target_collection == self.collection_name and self.vector_store:
                    deep_store = self.vector_store
                else:
                    deep_store = self._open_vector_store(
                        target_collection,
                        create_if_missing=False,
                    )
                    close_deep_store = True

                deep_count = await deep_store.count()
                sample_limit = min(max(0, int(sample_limit or 3)), 20)
                sample_rows = await deep_store.get_all_memories(
                    limit=max(1, sample_limit),
                    offset=0,
                    include_embeddings=True,
                )
                deep_dimension = None
                if sample_rows:
                    first_embedding = sample_rows[0].get("embedding")
                    if first_embedding is not None:
                        deep_dimension = len(first_embedding)

                missing_scope = 0
                missing_owner = 0
                type_counts: Dict[str, int] = {}
                scope_counts: Dict[str, int] = {}
                samples: List[Dict[str, Any]] = []
                offset = 0
                batch_size = 500
                while offset < deep_count:
                    memories = await deep_store.get_all_memories(
                        limit=batch_size,
                        offset=offset,
                    )
                    if not memories:
                        break
                    for memory in memories:
                        metadata = memory.get("metadata") or {}
                        mem_type = str(metadata.get("type") or "raw")
                        scope = str(metadata.get("scope") or "")
                        type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
                        scope_counts[scope or "missing"] = (
                            scope_counts.get(scope or "missing", 0) + 1
                        )
                        if not scope:
                            missing_scope += 1
                        if not metadata.get("owner_user_id") and not metadata.get(
                            "owner_session_id"
                        ):
                            missing_owner += 1
                        if len(samples) < sample_limit:
                            sample_item = {
                                "id": memory.get("id"),
                                "metadata": metadata,
                            }
                            if include_sample_text:
                                sample_item["text"] = memory.get("text", "")
                            samples.append(sample_item)
                    offset += len(memories)

                info["deep_check"] = {
                    "collection_name": target_collection,
                    "memory_count": deep_count,
                    "stored_embedding_dimension": deep_dimension,
                    "dimension_matched": (
                        deep_dimension is None
                        or self.embedding_dimension is None
                        or deep_dimension == self.embedding_dimension
                    ),
                    "missing_scope": missing_scope,
                    "missing_owner": missing_owner,
                    "type_distribution": type_counts,
                    "scope_distribution": scope_counts,
                    "samples": samples,
                }
            except Exception as e:
                info["deep_check"] = {
                    "collection_name": target_collection,
                    "error": str(e),
                }
            finally:
                if close_deep_store and deep_store:
                    await deep_store.close()
        return json.dumps(info, ensure_ascii=False, indent=2)

    def _build_dimension_collection_name(self) -> str:
        """根据当前 embedding 维度生成一个安全的新集合名。"""
        dimension = self.embedding_dimension or 0
        base = re.sub(r"[^a-zA-Z0-9_-]", "_", self.collection_name or "")
        if not base or not re.match(r"^[a-zA-Z0-9]", base):
            base = DEFAULT_COLLECTION_NAME
        suffix = f"{dimension}d"
        max_base_len = max(3, 63 - len(suffix) - 1)
        return f"{base[:max_base_len].rstrip('_-')}_{suffix}"

    @register.tool(
        "list_memory_collections",
        "列出 vector_memory 的 Chroma 集合及维度，只读工具。适合在切换集合或更换 embedding 模型前使用。",
        {"type": "object", "properties": {}},
    )
    async def list_memory_collections(self, event) -> str:
        """列出 Chroma 集合。"""
        rows: List[Dict[str, Any]] = []
        for item in self._list_chroma_collections_from_sqlite():
            name = str(item.get("name") or "")
            row = {
                "name": name,
                "sqlite_dimension": item.get("dimension"),
                "active": name == self.collection_name,
            }
            store: Optional[VectorStore] = None
            close_store = False
            try:
                if name == self.collection_name and self.vector_store:
                    store = self.vector_store
                    close_store = False
                else:
                    store = self._open_vector_store(name, create_if_missing=False)
                    close_store = True
                row["count"] = await store.count()
                row["sample_embedding_dimension"] = await self._get_store_embedding_dimension(
                    store
                )
                row["openable"] = True
            except Exception as e:
                row["openable"] = False
                row["error"] = str(e)
                close_store = False
            finally:
                if store and close_store:
                    await store.close()
            rows.append(row)

        return json.dumps(
            {
                "current_collection": self.collection_name,
                "current_embedding_dimension": self.embedding_dimension,
                "runtime_ready": self.runtime_ready,
                "runtime_error": self.runtime_error,
                "collections": rows,
            },
            ensure_ascii=False,
            indent=2,
        )

    @register.tool(
        "handle_embedding_model_change",
        "处理更换 embedding 模型后的记忆库维度变化。可只诊断、创建新空集合并切换，或转入重建索引流程。",
        {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["diagnose", "create_empty_collection", "reindex_to_new_collection"],
                    "description": "操作类型。diagnose 只读；create_empty_collection 创建新空集合并切换；reindex_to_new_collection 重建到新集合。",
                },
                "new_collection_name": {
                    "type": "string",
                    "description": "新集合名，可选；为空时按当前 embedding 维度自动生成。",
                },
                "source_collection_name": {
                    "type": "string",
                    "description": "重建索引时的源集合名，可选，默认当前集合。",
                },
                "confirm": {
                    "type": "boolean",
                    "description": "非 diagnose 操作必须为 true。",
                },
                "batch_size": {
                    "type": "integer",
                    "description": "重建索引批大小，默认 16。",
                },
                "resume": {
                    "type": "boolean",
                    "description": "重建索引目标集合非空时是否续跑。",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "重建索引只统计不写入，默认 false。",
                },
            },
        },
    )
    async def handle_embedding_model_change(
        self,
        event,
        action: str = "diagnose",
        new_collection_name: str = "",
        source_collection_name: str = "",
        confirm: bool = False,
        batch_size: int = 16,
        resume: bool = False,
        dry_run: bool = False,
    ) -> str:
        """让用户通过 LLM 工具处理 embedding 模型切换，不需要手改代码。"""
        action = str(action or "diagnose").strip()
        if action not in {"diagnose", "create_empty_collection", "reindex_to_new_collection"}:
            return "操作失败：action 必须是 diagnose、create_empty_collection 或 reindex_to_new_collection"

        current_dimension = self.embedding_dimension
        stored_dimension = None
        if self.vector_store:
            try:
                stored_dimension = await self._get_existing_embedding_dimension()
            except Exception:
                stored_dimension = None

        suggested_name = str(new_collection_name or "").strip() or self._build_dimension_collection_name()
        diagnosis = {
            "current_collection": self.collection_name,
            "current_embedding_dimension": current_dimension,
            "stored_collection_dimension": stored_dimension,
            "dimension_mismatch": (
                bool(current_dimension and stored_dimension)
                and current_dimension != stored_dimension
            ),
            "runtime_ready": self.runtime_ready,
            "runtime_error": self.runtime_error,
            "suggested_new_collection": suggested_name,
            "建议": [
                "测试数据或不需要旧记忆时，使用 action=create_empty_collection 创建新集合并切换。",
                "需要保留旧记忆内容时，使用 action=reindex_to_new_collection 重建到新集合，再调用 activate_collection 启用。",
                "不要在同一个集合里混用不同维度或不同语义空间的 embedding。",
            ],
        }

        if action == "diagnose":
            return json.dumps(diagnosis, ensure_ascii=False, indent=2)

        if not confirm and not dry_run:
            return "操作失败：非 diagnose 操作必须传入 confirm=true"
        if not self.embedding or not self.embedding_dimension:
            return "操作失败：当前 embedding 尚未初始化，不能处理模型切换"

        if action == "reindex_to_new_collection":
            return await self.reindex_memories(
                event=event,
                new_collection_name=suggested_name,
                confirm=confirm,
                source_collection_name=source_collection_name,
                batch_size=batch_size,
                resume=resume,
                dry_run=dry_run,
            )

        if not self._is_valid_collection_name(suggested_name):
            return "操作失败：新集合名无效，只能包含字母、数字、下划线和短横线"

        lock = self._get_maintenance_lock()
        if lock.locked():
            return "操作失败：已有维护操作正在进行，请稍后再试"

        original_store = self.vector_store
        original_collection = self.collection_name
        original_meta = dict(self.meta or {})
        original_runtime_ready = self.runtime_ready
        original_runtime_error = self.runtime_error
        target_store: Optional[VectorStore] = None
        close_target_on_error = False
        try:
            async with lock:
                target_existed = True
                try:
                    target_store = self._open_vector_store(
                        suggested_name,
                        create_if_missing=False,
                    )
                    close_target_on_error = True
                    target_count = await target_store.count()
                    if target_count > 0:
                        target_dimension = await self._get_store_embedding_dimension(
                            target_store
                        )
                        if target_dimension and target_dimension != self.embedding_dimension:
                            return (
                                f"操作失败：目标集合 {suggested_name} 维度为 "
                                f"{target_dimension}，当前 embedding 维度为 "
                                f"{self.embedding_dimension}，不能切换。"
                            )
                        return (
                            f"操作失败：目标集合 {suggested_name} 已存在且包含 "
                            f"{target_count} 条记忆。为避免误切换，请改用 "
                            "activate_collection 或 reindex_to_new_collection。"
                        )
                except Exception as e:
                    if not self._is_collection_not_found_error(e):
                        raise
                    target_existed = False
                    target_store = self._open_vector_store(
                        suggested_name,
                        create_if_missing=True,
                    )
                    close_target_on_error = True

                current_backup: Optional[Path] = None
                if original_store:
                    current_backup = await self._create_export_backup(
                        "before_embedding_model_change",
                        store=original_store,
                        collection_name=original_collection,
                    )

                self.collection_name = suggested_name
                self.plugin_cfg["collection_name"] = suggested_name
                self.vector_store = target_store
                self.runtime_ready = True
                self.runtime_error = ""
                await self._ensure_data_contract()
                self._init_context_injector()
                self._init_reflection_manager()
                self._persist_plugin_config({"collection_name": suggested_name})
                if original_store and original_store is not target_store:
                    await original_store.close()
                close_target_on_error = False

                return json.dumps(
                    {
                        "已切换到新集合": suggested_name,
                        "target_existed": target_existed,
                        "current_embedding_dimension": self.embedding_dimension,
                        "旧集合已保留": original_collection,
                        "当前集合备份": str(current_backup) if current_backup else "",
                        "下一步": "可以正常继续新增、搜索和自动注入；旧集合没有删除。",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            self.collection_name = original_collection
            self.plugin_cfg["collection_name"] = original_collection
            self.vector_store = original_store
            self.meta = original_meta
            self.runtime_ready = original_runtime_ready
            self.runtime_error = original_runtime_error
            if self.data_dir is not None:
                self._save_meta()
            if target_store and target_store is not original_store and close_target_on_error:
                await target_store.close()
            logger.error(f"处理 embedding 模型变更失败: {e}\n{traceback.format_exc()}")
            return f"操作失败：处理 embedding 模型变更失败: {str(e)}"

    @register.tool(
        "export_memories",
        "导出向量记忆为 JSONL 文件。",
        {
            "type": "object",
            "properties": {
                "include_embeddings": {
                    "type": "boolean",
                    "description": "是否包含 embedding，默认 false",
                },
                "file_name": {"type": "string", "description": "导出文件名，可选"},
                "source_collection": {
                    "type": "string",
                    "description": "要导出的集合名，可选；为空时导出当前集合",
                },
                "redact": {
                    "type": "boolean",
                    "description": "是否脱敏导出；为 true 时不导出完整正文和 embedding",
                },
            },
        },
    )
    async def export_memories(
        self,
        event,
        include_embeddings: bool = False,
        file_name: str = "",
        source_collection: str = "",
        redact: bool = False,
    ) -> str:
        """导出记忆。"""
        source_collection = str(source_collection or "").strip()
        if source_collection and not self._is_valid_collection_name(source_collection):
            return "操作失败：集合名无效，只能包含字母、数字、下划线和短横线"

        if not source_collection and not self.vector_store:
            return (
                "操作失败：当前向量记忆库尚未初始化；"
                "如需维护恢复，请指定 source_collection 导出某个已存在集合。"
            )

        export_store: Optional[VectorStore] = self.vector_store
        close_export_store = False
        try:
            if source_collection and (
                source_collection != self.collection_name or not self.vector_store
            ):
                export_store = self._open_vector_store(
                    source_collection,
                    create_if_missing=False,
                )
                close_export_store = True
            elif source_collection and source_collection == self.collection_name:
                export_store = self.vector_store

            export_prefix = file_name or (
                f"manual_export_{source_collection}" if source_collection else "manual_export"
            )
            export_path = self._build_export_path(export_prefix)
            count = await self._export_memories_to_path(
                export_path,
                include_embeddings=include_embeddings,
                store=export_store,
                redact=redact,
            )
            suffix = "（已脱敏）" if redact else ""
            return f"已导出 {count} 条记忆到: {export_path}{suffix}"
        except Exception as e:
            logger.error(f"导出记忆失败: {e}")
            return f"操作失败：导出失败: {str(e)}"
        finally:
            if close_export_store and export_store:
                await export_store.close()

    @register.tool(
        "import_memories",
        "从插件数据目录中的 JSONL 文件导入记忆。危险操作，会写入当前或指定集合。",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "插件数据目录内的 JSONL 路径"},
                "confirm": {"type": "boolean", "description": "必须为 true"},
                "target_collection": {"type": "string", "description": "目标集合名，可选"},
                "limit": {"type": "integer", "description": "最多导入条数，0 表示不限"},
                "batch_size": {"type": "integer", "description": "批大小，默认 32"},
                "mode": {
                    "type": "string",
                    "enum": ["append", "dedupe", "new_collection_only"],
                    "description": "导入模式，默认 dedupe",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "只分析不写入、不生成 embedding，默认 false",
                },
            },
            "required": ["path", "confirm"],
        },
    )
    async def import_memories(
        self,
        event,
        path: str,
        confirm: bool = False,
        target_collection: str = "",
        limit: int = 0,
        batch_size: int = 32,
        mode: str = "dedupe",
        dry_run: bool = False,
    ) -> str:
        """导入记忆。"""
        if not confirm and not dry_run:
            return "操作失败：请确认导入操作，将 confirm 设置为 true"
        mode = str(mode or "dedupe").strip()
        if mode not in VALID_IMPORT_MODES:
            return "操作失败：mode 必须是 append、dedupe 或 new_collection_only"
        if not dry_run and not self.embedding:
            return "操作失败：Embedding 未初始化，无法导入"

        requested_target = bool(str(target_collection or "").strip())
        target_collection = str(target_collection or self.collection_name).strip()
        if not self._is_valid_collection_name(target_collection):
            return "操作失败：目标集合名无效，只能包含字母、数字、下划线和短横线"
        if not self.vector_store and not requested_target:
            return (
                "操作失败：当前集合不可用。维护恢复时请指定新的 target_collection，"
                "避免继续写入可能损坏的当前集合。"
            )
        if target_collection == self.collection_name and not self.runtime_ready:
            return "操作失败：当前集合处于维护模式，不能直接导入；请指定新的 target_collection"

        lock = self._get_maintenance_lock()
        if lock.locked():
            return "操作失败：已有维护操作正在进行，请稍后再试"

        target_store: Optional[VectorStore] = None
        close_target = False
        try:
            async with lock:
                source_path = self._resolve_plugin_data_path(path)
                target_exists = True
                if target_collection == self.collection_name and self.vector_store:
                    target_store = self.vector_store
                else:
                    try:
                        target_store = self._open_vector_store(
                            target_collection,
                            create_if_missing=False,
                        )
                    except Exception as target_error:
                        if not self._is_collection_not_found_error(target_error):
                            raise
                        target_exists = False
                        if not dry_run:
                            target_store = self._open_vector_store(
                                target_collection,
                                create_if_missing=True,
                            )
                    close_target = bool(target_store) and (
                        target_collection != self.collection_name
                        or not self.vector_store
                    )

                target_count = await target_store.count() if target_store else 0
                if mode == "new_collection_only" and target_count > 0:
                    return (
                        f"操作失败：目标集合 {target_collection} 已存在且包含 "
                        f"{target_count} 条记忆，new_collection_only 模式拒绝写入。"
                    )

                existing_fingerprints = set()
                if mode == "dedupe" and target_store:
                    existing_fingerprints = await self._collect_memory_fingerprints(
                        target_store
                    )

                if not dry_run and target_count > 0:
                    await self._create_export_backup(
                        "before_import",
                        store=target_store,
                        collection_name=target_collection,
                    )

                imported = 0
                planned = 0
                skipped_embedding = 0
                skipped_duplicate = 0
                skipped_empty = 0
                scanned = 0
                failures: List[Dict[str, Any]] = []
                batch_records: List[Dict[str, Any]] = []

                async def flush_batch():
                    nonlocal imported, skipped_embedding, failures, batch_records
                    if not batch_records:
                        return
                    batch_texts = [item["text"] for item in batch_records]
                    embeddings, batch_failures = await self._generate_embeddings_resilient(
                        batch_texts,
                        source=f"import:{source_path.name}",
                    )
                    for failure in batch_failures:
                        try:
                            failed_index = int(failure.get("index", -1))
                        except Exception:
                            failed_index = -1
                        if 0 <= failed_index < len(batch_records):
                            failure["source_memory_id"] = batch_records[failed_index].get(
                                "source_id", ""
                            )
                            failure["fingerprint"] = batch_records[failed_index].get(
                                "fingerprint", ""
                            )
                    failures.extend(batch_failures)
                    for record, embedding in zip(batch_records, embeddings):
                        if embedding is None:
                            skipped_embedding += 1
                            continue
                        await target_store.add(
                            text=record["text"],
                            embedding=embedding,
                            metadata=record["metadata"],
                        )
                        imported += 1
                    batch_records = []

                batch_size = min(max(1, int(batch_size or 32)), 128)
                with source_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        text = str(item.get("text") or "").strip()
                        if not text:
                            skipped_empty += 1
                            continue
                        if limit and scanned >= int(limit):
                            break
                        scanned += 1
                        metadata = self._normalize_memory_metadata(
                            item.get("metadata") or {}
                        )
                        fingerprint = self._memory_fingerprint(text, metadata)
                        if mode == "dedupe" and fingerprint in existing_fingerprints:
                            skipped_duplicate += 1
                            continue
                        if mode == "dedupe":
                            # 同一个导入文件内的重复项也应跳过。
                            existing_fingerprints.add(fingerprint)
                        source_id = str(item.get("id") or "")
                        metadata = self._metadata_with_import_info(
                            metadata=metadata,
                            source_id=source_id,
                            source_file=source_path.name,
                            fingerprint=fingerprint,
                        )
                        planned += 1
                        if dry_run:
                            continue
                        batch_records.append(
                            {
                                "text": text,
                                "metadata": metadata,
                                "source_id": source_id,
                                "fingerprint": fingerprint,
                            }
                        )
                        if len(batch_records) >= batch_size:
                            await flush_batch()

                if dry_run:
                    return (
                        "导入 dry_run 完成："
                        f"文件={source_path.name}，目标集合={target_collection}，"
                        f"目标集合存在={target_exists}，目标现有={target_count}，"
                        f"扫描有效记忆={scanned}，预计写入={planned}，"
                        f"重复跳过={skipped_duplicate}，空文本跳过={skipped_empty}，"
                        f"模式={mode}"
                    )

                await flush_batch()
                failure_path = self._write_failure_records(
                    "import_failed_embeddings",
                    failures,
                )
                result = f"已导入 {imported} 条记忆到集合: {target_collection}"
                if skipped_duplicate:
                    result += f"，去重跳过 {skipped_duplicate} 条"
                if skipped_embedding:
                    result += f"，跳过 {skipped_embedding} 条无法生成 embedding 的记忆"
                if skipped_empty:
                    result += f"，空文本跳过 {skipped_empty} 条"
                if failure_path:
                    result += f"，失败记录: {failure_path}"
                result += f"，模式: {mode}"
                return result
        except Exception as e:
            logger.error(f"导入记忆失败: {e}\n{traceback.format_exc()}")
            return f"操作失败：导入失败: {str(e)}"
        finally:
            if close_target and target_store:
                await target_store.close()

    @register.tool(
        "reindex_memories",
        "用当前 embedding 重建索引到新集合。危险操作，但不会覆盖旧集合。",
        {
            "type": "object",
            "properties": {
                "new_collection_name": {"type": "string", "description": "新集合名"},
                "confirm": {"type": "boolean", "description": "必须为 true"},
                "source_collection_name": {"type": "string", "description": "源集合名，可选"},
                "batch_size": {"type": "integer", "description": "批大小，默认 16；大集合建议 16"},
                "resume": {
                    "type": "boolean",
                    "description": "目标集合非空时是否按指纹续跑，默认 false",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "只统计不写入、不生成 embedding，默认 false",
                },
            },
            "required": ["new_collection_name", "confirm"],
        },
    )
    async def reindex_memories(
        self,
        event,
        new_collection_name: str,
        confirm: bool = False,
        source_collection_name: str = "",
        batch_size: int = 16,
        resume: bool = False,
        dry_run: bool = False,
    ) -> str:
        """重建索引到新集合。"""
        if not confirm and not dry_run:
            return "操作失败：请确认重建索引操作，将 confirm 设置为 true"
        if not dry_run and not self.embedding:
            return "操作失败：Embedding 未初始化，无法重建索引"

        new_collection_name = str(new_collection_name or "").strip()
        if not self._is_valid_collection_name(new_collection_name):
            return "操作失败：新集合名无效，只能包含字母、数字、下划线和短横线"

        requested_source = bool(str(source_collection_name or "").strip())
        source_collection_name = str(source_collection_name or self.collection_name).strip()
        if not self._is_valid_collection_name(source_collection_name):
            return "操作失败：源集合名无效，只能包含字母、数字、下划线和短横线"
        if new_collection_name == source_collection_name:
            return "操作失败：新集合名不能与源集合相同"
        if not self.vector_store and not requested_source:
            return (
                "操作失败：当前集合不可用。维护恢复时请指定 source_collection_name。"
            )

        lock = self._get_maintenance_lock()
        if lock.locked():
            return "操作失败：已有维护操作正在进行，请稍后再试"

        source_store: Optional[VectorStore] = None
        target_store: Optional[VectorStore] = None
        close_source = False
        close_target = False
        start_time = time.time()
        try:
            async with lock:
                if source_collection_name == self.collection_name and self.vector_store:
                    source_store = self.vector_store
                else:
                    # 只打开已存在的源集合，避免拼错源集合名后生成空集合。
                    try:
                        source_store = self._open_vector_store(
                            source_collection_name,
                            create_if_missing=False,
                        )
                    except Exception as source_error:
                        if self._is_collection_not_found_error(source_error):
                            return (
                                f"操作失败：源集合 {source_collection_name} 不存在，"
                                "已取消重建，未创建空集合。"
                            )
                        raise
                    close_source = True

                if not dry_run:
                    await self._create_export_backup(
                        "before_reindex",
                        store=source_store,
                        collection_name=source_collection_name,
                    )

                target_count = 0
                target_exists = True
                try:
                    target_store = self._open_vector_store(
                        new_collection_name,
                        create_if_missing=False,
                    )
                    close_target = True
                    target_count = await target_store.count()
                    if target_count > 0 and not resume:
                        return (
                            f"目标集合 {new_collection_name} 已存在且包含 "
                            f"{target_count} 条记忆。为避免重复追加，请换一个新的集合名；"
                            "如需从半成品集合续跑，请显式传入 resume=true。"
                        )
                except Exception as target_error:
                    if not self._is_collection_not_found_error(target_error):
                        raise
                    target_exists = False
                    if not dry_run:
                        target_store = self._open_vector_store(
                            new_collection_name,
                            create_if_missing=True,
                        )
                        close_target = True

                existing_fingerprints = set()
                if resume and target_store:
                    existing_fingerprints = await self._collect_memory_fingerprints(
                        target_store
                    )

                total = await source_store.count()
                offset = 0
                processed = 0
                rebuilt = 0
                planned = 0
                skipped_embedding = 0
                skipped_duplicate = 0
                skipped_empty = 0
                failed = 0
                failure_path: Optional[Path] = None
                # 限制批大小，避免部分 embedding 服务因单次输入过大而失败。
                batch_size = min(max(1, int(batch_size or 16)), 256)
                logger.info(
                    f"开始重建向量记忆索引: {source_collection_name} -> "
                    f"{new_collection_name}，总数: {total}，批大小: {batch_size}，"
                    f"resume={resume}，dry_run={dry_run}"
                )

                while offset < total:
                    memories = await source_store.get_all_memories(
                        limit=batch_size,
                        offset=offset,
                    )
                    if not memories:
                        break
                    processed += len(memories)

                    texts: List[str] = []
                    metadatas: List[Dict[str, Any]] = []
                    source_ids: List[str] = []
                    fingerprints: List[str] = []
                    for mem in memories:
                        text = str(mem.get("text") or "").strip()
                        if not text:
                            skipped_empty += 1
                            continue
                        metadata = self._normalize_memory_metadata(
                            mem.get("metadata") or {}
                        )
                        fingerprint = self._memory_fingerprint(text, metadata)
                        if resume and fingerprint in existing_fingerprints:
                            skipped_duplicate += 1
                            continue
                        if resume:
                            existing_fingerprints.add(fingerprint)
                        source_id = str(mem.get("id") or "")
                        texts.append(text)
                        metadatas.append(
                            self._metadata_with_import_info(
                                metadata=metadata,
                                source_id=source_id,
                                source_file=f"reindex:{source_collection_name}",
                                fingerprint=fingerprint,
                            )
                        )
                        source_ids.append(source_id)
                        fingerprints.append(fingerprint)
                        planned += 1

                    if dry_run:
                        offset += len(memories)
                        continue

                    if texts:
                        embeddings, batch_failures = (
                            await self._generate_embeddings_resilient(
                                texts,
                                source=f"reindex:{source_collection_name}",
                                progress_every=max(1, min(16, batch_size)),
                            )
                        )
                        for failure in batch_failures:
                            try:
                                failed_index = int(failure.get("index", -1))
                            except Exception:
                                failed_index = -1
                            if 0 <= failed_index < len(source_ids):
                                failure["source_memory_id"] = source_ids[failed_index]
                                failure["fingerprint"] = fingerprints[failed_index]
                        failed += len(batch_failures)
                        failure_path = self._append_failure_records(
                            batch_failures,
                            path=failure_path,
                            prefix="reindex_failed_embeddings",
                        )
                        for text, metadata, embedding in zip(texts, metadatas, embeddings):
                            if embedding is None:
                                skipped_embedding += 1
                                continue
                            await target_store.add(
                                text=text,
                                embedding=embedding,
                                metadata=metadata,
                            )
                            rebuilt += 1

                    offset += len(memories)
                    elapsed = time.time() - start_time
                    logger.info(
                        f"重建索引进度: processed={processed}/{total}, "
                        f"rebuilt={rebuilt}, skipped_duplicate={skipped_duplicate}, "
                        f"skipped_embedding={skipped_embedding}, failed={failed}, "
                        f"elapsed={elapsed:.1f}s -> {new_collection_name}"
                    )

                if dry_run:
                    return (
                        "重建索引 dry_run 完成："
                        f"源集合={source_collection_name}，目标集合={new_collection_name}，"
                        f"目标集合存在={target_exists}，目标现有={target_count}，"
                        f"源总数={total}，预计写入={planned}，"
                        f"续跑去重跳过={skipped_duplicate}，空文本跳过={skipped_empty}，"
                        f"processed={processed}，resume={resume}"
                    )

                elapsed = time.time() - start_time
                final_target_count = await target_store.count() if target_store else 0
                result = (
                    f"已重建索引到新集合: {new_collection_name}；"
                    f"processed={processed}/{total}，rebuilt={rebuilt}，"
                    f"skipped_duplicate={skipped_duplicate}，"
                    f"skipped_embedding={skipped_embedding}，failed={failed}，"
                    f"目标集合当前 count={final_target_count}，耗时 {elapsed:.1f}s"
                )
                if skipped_duplicate:
                    result += f"，续跑去重跳过 {skipped_duplicate} 条"
                if skipped_embedding:
                    result += f"，跳过 {skipped_embedding} 条无法生成 embedding 的记忆"
                if skipped_empty:
                    result += f"，空文本跳过 {skipped_empty} 条"
                if failure_path:
                    result += f"，失败记录: {failure_path}"
                if failed or skipped_embedding:
                    result += "。如任务中断或需要补跑，建议使用 resume=true 续跑"
                result += "。如需启用，请调用 activate_collection。"
                return result
        except Exception as e:
            logger.error(f"重建索引失败: {e}\n{traceback.format_exc()}")
            return f"操作失败：重建失败: {str(e)}"
        finally:
            if close_source and source_store:
                await source_store.close()
            if close_target and target_store:
                await target_store.close()

    def _build_summary_scope_metadata(
        self,
        metadata: Dict[str, Any],
        target_scope: str,
        session_id: str = "",
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """为 summary 二次迁移生成新 metadata；无法安全判断归属时拒绝迁移。"""
        old_metadata = self._normalize_memory_metadata(metadata or {})
        if str(old_metadata.get("type") or "") != "summary":
            return None, "not_summary"

        current_session_id = str(
            old_metadata.get("owner_session_id")
            or old_metadata.get("session_id")
            or ""
        )
        if session_id and current_session_id != session_id:
            return None, "session_mismatch"

        new_metadata = dict(old_metadata)
        if target_scope == "global":
            new_metadata["scope"] = "global"
            new_metadata["session_id"] = "global"
            new_metadata["owner_user_id"] = ""
            new_metadata["owner_session_id"] = ""
            return new_metadata, "ok"

        if target_scope == "session":
            owner_session_id = current_session_id or session_id
            if not owner_session_id:
                return None, "missing_owner_session_id"
            new_metadata["scope"] = "session"
            new_metadata["session_id"] = owner_session_id
            new_metadata["owner_session_id"] = owner_session_id
            return new_metadata, "ok"

        if target_scope == "user":
            owner_user_id = str(
                old_metadata.get("owner_user_id")
                or old_metadata.get("user_id")
                or ""
            )
            if not owner_user_id or owner_user_id == "system":
                return None, "missing_owner_user_id"
            owner_adapter = str(
                old_metadata.get("owner_adapter")
                or old_metadata.get("adapter")
                or ""
            )
            new_metadata["scope"] = "user"
            new_metadata["session_id"] = f"user:{owner_adapter}:{owner_user_id}"
            new_metadata["owner_user_id"] = owner_user_id
            new_metadata["owner_adapter"] = owner_adapter
            return new_metadata, "ok"

        return None, "invalid_target_scope"

    @register.tool(
        "plan_summary_scope_migration",
        "只读分析 summary 记忆作用域迁移计划，不修改任何数据。",
        {
            "type": "object",
            "properties": {
                "target_scope": {
                    "type": "string",
                    "enum": ["user", "global", "session"],
                    "description": "目标作用域，默认 user",
                },
                "session_id": {
                    "type": "string",
                    "description": "仅分析指定会话的 summary，选填",
                },
                "sample_limit": {
                    "type": "integer",
                    "description": "返回样本数量，默认 5",
                },
            },
        },
    )
    async def plan_summary_scope_migration(
        self,
        event,
        target_scope: str = "user",
        session_id: str = "",
        sample_limit: int = 5,
    ) -> str:
        """只读分析 summary 作用域二次迁移。"""
        if not self.vector_store:
            return "操作失败：向量记忆库尚未初始化"
        target_scope = str(target_scope or "user").strip()
        if target_scope not in VALID_SCOPES:
            return "操作失败：target_scope 必须是 user、global 或 session"
        session_id = str(session_id or "").strip()
        sample_limit = min(max(0, int(sample_limit or 5)), 20)

        total = await self.vector_store.count()
        offset = 0
        batch_size = 500
        summary_count = 0
        migratable = 0
        unchanged = 0
        skipped: Dict[str, int] = {}
        current_scope_counts: Dict[str, int] = {}
        samples: List[Dict[str, Any]] = []

        while offset < total:
            memories = await self.vector_store.get_all_memories(
                limit=batch_size,
                offset=offset,
            )
            if not memories:
                break
            for memory in memories:
                metadata = self._normalize_memory_metadata(memory.get("metadata") or {})
                if str(metadata.get("type") or "") != "summary":
                    continue
                summary_count += 1
                old_scope = str(metadata.get("scope") or "missing")
                current_scope_counts[old_scope] = current_scope_counts.get(old_scope, 0) + 1
                new_metadata, reason = self._build_summary_scope_metadata(
                    metadata,
                    target_scope=target_scope,
                    session_id=session_id,
                )
                if reason != "ok" or new_metadata is None:
                    skipped[reason] = skipped.get(reason, 0) + 1
                    continue
                if new_metadata == metadata:
                    unchanged += 1
                    continue
                migratable += 1
                if len(samples) < sample_limit:
                    samples.append(
                        {
                            "id": memory.get("id"),
                            "old_scope": old_scope,
                            "target_scope": target_scope,
                            "session_id": metadata.get("session_id", ""),
                            "owner_user_id": metadata.get("owner_user_id", ""),
                            "owner_session_id": metadata.get("owner_session_id", ""),
                            "text_preview": str(memory.get("text") or "")[:120],
                        }
                    )
            offset += len(memories)

        return json.dumps(
            {
                "collection_name": self.collection_name,
                "target_scope": target_scope,
                "session_id_filter": session_id,
                "summary_count": summary_count,
                "current_scope_distribution": current_scope_counts,
                "migratable": migratable,
                "unchanged": unchanged,
                "skipped": skipped,
                "samples": samples,
                "write_required": (
                    "migrate_summary_scope(dry_run=false, confirm=true, "
                    f"confirm_phrase={SUMMARY_SCOPE_CONFIRM_PHRASE})"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )

    @register.tool(
        "migrate_summary_scope",
        "迁移 summary 记忆作用域；默认 dry_run，只更新 metadata，不改正文和 embedding。",
        {
            "type": "object",
            "properties": {
                "target_scope": {
                    "type": "string",
                    "enum": ["user", "global", "session"],
                    "description": "目标作用域，默认 user",
                },
                "session_id": {"type": "string", "description": "仅迁移指定会话，选填"},
                "dry_run": {"type": "boolean", "description": "只分析不写入，默认 true"},
                "confirm": {"type": "boolean", "description": "真实迁移必须为 true"},
                "confirm_phrase": {
                    "type": "string",
                    "description": f"真实迁移必须填写 {SUMMARY_SCOPE_CONFIRM_PHRASE}",
                },
            },
        },
    )
    async def migrate_summary_scope(
        self,
        event,
        target_scope: str = "user",
        session_id: str = "",
        dry_run: bool = True,
        confirm: bool = False,
        confirm_phrase: str = "",
    ) -> str:
        """迁移 summary 作用域；真实写入前会自动备份当前集合。"""
        if not self.vector_store:
            return "操作失败：向量记忆库尚未初始化"
        target_scope = str(target_scope or "user").strip()
        if target_scope not in VALID_SCOPES:
            return "操作失败：target_scope 必须是 user、global 或 session"
        if not dry_run:
            if not confirm:
                return "操作失败：真实迁移需要 confirm=true"
            if str(confirm_phrase or "").strip() != SUMMARY_SCOPE_CONFIRM_PHRASE:
                return (
                    "操作失败：真实迁移 summary 作用域需要确认短语 "
                    f"{SUMMARY_SCOPE_CONFIRM_PHRASE}"
                )

        lock = self._get_maintenance_lock()
        if lock.locked():
            return "操作失败：已有维护操作正在进行，请稍后再试"

        session_id = str(session_id or "").strip()
        total = await self.vector_store.count()
        offset = 0
        batch_size = 500
        scanned = 0
        planned = 0
        changed = 0
        unchanged = 0
        skipped: Dict[str, int] = {}
        backup_path: Optional[Path] = None

        try:
            async with lock:
                if not dry_run:
                    backup_path = await self._create_export_backup(
                        "before_migrate_summary_scope",
                        store=self.vector_store,
                        collection_name=self.collection_name,
                    )
                while offset < total:
                    memories = await self.vector_store.get_all_memories(
                        limit=batch_size,
                        offset=offset,
                    )
                    if not memories:
                        break
                    for memory in memories:
                        metadata = self._normalize_memory_metadata(
                            memory.get("metadata") or {}
                        )
                        if str(metadata.get("type") or "") != "summary":
                            continue
                        scanned += 1
                        new_metadata, reason = self._build_summary_scope_metadata(
                            metadata,
                            target_scope=target_scope,
                            session_id=session_id,
                        )
                        if reason != "ok" or new_metadata is None:
                            skipped[reason] = skipped.get(reason, 0) + 1
                            continue
                        if new_metadata == metadata:
                            unchanged += 1
                            continue
                        planned += 1
                        if dry_run:
                            continue
                        ok = await self.vector_store.update_metadata(
                            str(memory.get("id") or ""),
                            new_metadata,
                        )
                        if ok:
                            changed += 1
                    offset += len(memories)

            result = (
                f"summary 作用域迁移{' dry_run' if dry_run else ''}完成："
                f"集合={self.collection_name}，目标作用域={target_scope}，"
                f"扫描 summary={scanned}，预计更新={planned}，已更新={changed}，"
                f"无需更新={unchanged}，跳过={skipped}"
            )
            if backup_path:
                result += f"，备份: {backup_path}"
            return result
        except Exception as e:
            logger.error(f"迁移 summary 作用域失败: {e}\n{traceback.format_exc()}")
            return f"操作失败：迁移 summary 作用域失败: {str(e)}"

    @register.tool(
        "activate_collection",
        "切换当前使用的 Chroma 集合。危险操作，会修改插件配置。",
        {
            "type": "object",
            "properties": {
                "collection_name": {"type": "string", "description": "要启用的集合名"},
                "confirm": {"type": "boolean", "description": "必须为 true"},
            },
            "required": ["collection_name", "confirm"],
        },
    )
    async def activate_collection(
        self, event, collection_name: str, confirm: bool = False
    ) -> str:
        """切换当前集合。"""
        if not confirm:
            return "操作失败：请确认切换集合操作，将 confirm 设置为 true"
        collection_name = str(collection_name or "").strip()
        if not self._is_valid_collection_name(collection_name):
            return "操作失败：集合名无效，只能包含字母、数字、下划线和短横线"

        lock = self._get_maintenance_lock()
        if lock.locked():
            return "操作失败：已有维护操作正在进行，请稍后再试"

        target_store: Optional[VectorStore] = None
        original_store = self.vector_store
        original_collection = self.collection_name
        original_meta = dict(self.meta or {})
        original_runtime_ready = self.runtime_ready
        original_runtime_error = self.runtime_error
        try:
            async with lock:
                if collection_name == self.collection_name and self.vector_store:
                    backup = await self._create_export_backup(
                        "before_activate_collection_current",
                        store=self.vector_store,
                        collection_name=collection_name,
                    )
                    await self._ensure_data_contract()
                    return (
                        f"集合 {collection_name} 已是当前集合，已重新检查数据契约。"
                        f"备份: {backup}"
                    )

                # 先只打开已存在集合，避免用户拼错名称时创建空集合。
                target_store = self._open_vector_store(
                    collection_name,
                    create_if_missing=False,
                )

                current_backup: Optional[Path] = None
                if self.vector_store:
                    current_backup = await self._create_export_backup(
                        "before_activate_collection_current",
                        store=self.vector_store,
                        collection_name=self.collection_name,
                    )
                target_backup = await self._create_export_backup(
                    "before_activate_collection_target",
                    store=target_store,
                    collection_name=collection_name,
                )

                self.collection_name = collection_name
                self.plugin_cfg["collection_name"] = collection_name
                self.vector_store = target_store
                self.enabled = bool(self.plugin_cfg.get("enabled", True))
                self.runtime_ready = True
                self.runtime_error = ""
                await self._ensure_data_contract()
                self._init_context_injector()
                self._init_reflection_manager()
                self._persist_plugin_config({"collection_name": collection_name})
                if original_store and original_store is not target_store:
                    await original_store.close()
                result = f"已切换到集合: {collection_name}。"
                if current_backup:
                    result += f"当前集合备份: {current_backup}；"
                else:
                    result += "当前集合不可用，已跳过当前集合备份；"
                result += f"目标集合备份: {target_backup}"
                return result
        except Exception as e:
            self.collection_name = original_collection
            self.plugin_cfg["collection_name"] = original_collection
            self.vector_store = original_store
            self.meta = original_meta
            self.runtime_ready = original_runtime_ready
            self.runtime_error = original_runtime_error
            if self.data_dir is not None:
                self._save_meta()
            if target_store and target_store is not original_store:
                await target_store.close()
            logger.error(f"切换集合失败: {e}")
            return f"操作失败：切换失败: {str(e)}"

    @register.tool(
        "delete_memory_collection",
        "删除一个非当前 Chroma 集合。极高风险维护工具，删除前会自动备份目标集合。",
        {
            "type": "object",
            "properties": {
                "collection_name": {"type": "string", "description": "要删除的集合名"},
                "confirm": {"type": "boolean", "description": "必须为 true"},
                "confirm_phrase": {
                    "type": "string",
                    "description": f"必须填写 {DELETE_COLLECTION_CONFIRM_PHRASE}",
                },
            },
            "required": ["collection_name", "confirm"],
        },
    )
    async def delete_memory_collection(
        self,
        event,
        collection_name: str,
        confirm: bool = False,
        confirm_phrase: str = "",
    ) -> str:
        """删除非当前集合；仅用于清理确认无用的测试集合。"""
        if not confirm:
            return "操作失败：请确认删除集合操作，将 confirm 设置为 true"
        if str(confirm_phrase or "").strip() != DELETE_COLLECTION_CONFIRM_PHRASE:
            return (
                "操作失败：delete_memory_collection 是极高风险维护工具。"
                "如确需删除集合，请同时传入 "
                f"confirm_phrase={DELETE_COLLECTION_CONFIRM_PHRASE}"
            )

        collection_name = str(collection_name or "").strip()
        if not self._is_valid_collection_name(collection_name):
            return "操作失败：集合名无效，只能包含字母、数字、下划线和短横线"
        if collection_name == self.collection_name:
            return "操作失败：不能删除当前激活集合，请先切换到其他健康集合"

        lock = self._get_maintenance_lock()
        if lock.locked():
            return "操作失败：已有维护操作正在进行，请稍后再试"

        target_store: Optional[ChromaVectorStore] = None
        try:
            async with lock:
                target_store = self._open_vector_store(
                    collection_name,
                    create_if_missing=False,
                )
                target_count = await target_store.count()
                backup = await self._create_export_backup(
                    "before_delete_collection",
                    store=target_store,
                    collection_name=collection_name,
                )
                await target_store.delete_collection()
                return (
                    f"已删除集合 {collection_name}，删除前记忆 {target_count} 条；"
                    f"备份: {backup}"
                )
        except Exception as e:
            logger.error(f"删除集合失败: {e}\n{traceback.format_exc()}")
            return f"操作失败：删除集合失败: {str(e)}"
        finally:
            if target_store:
                await target_store.close()

    @register.tool(
        "update_memory",
        "按 memory_id 编辑单条记忆。危险操作；改正文会重新生成 embedding。",
        {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "记忆 ID"},
                "content": {
                    "type": "string",
                    "description": "新的正文；留空则不修改正文",
                },
                "importance": {
                    "type": "number",
                    "description": "新的重要性 0-1；不传则不修改",
                },
                "tags": {"type": "string", "description": "新的标签；留空则不修改"},
                "scope": {
                    "type": "string",
                    "enum": ["", "session", "user", "global"],
                    "description": "新的作用域；留空则不修改",
                },
                "confirm": {"type": "boolean", "description": "必须为 true"},
            },
            "required": ["memory_id", "confirm"],
        },
    )
    async def update_memory(
        self,
        event,
        memory_id: str,
        content: str = "",
        importance: float = None,
        tags: str = "",
        scope: str = "",
        confirm: bool = False,
    ) -> str:
        """编辑单条记忆。"""
        if not self._can_use_vector_runtime():
            return self._runtime_unavailable_text()
        if not confirm:
            return "操作失败：请确认编辑记忆操作，将 confirm 设置为 true"

        memory_id = str(memory_id or "").strip()
        if not memory_id:
            return "操作失败：memory_id 不能为空"

        existing = await self.vector_store.get_by_id(memory_id)
        if not existing:
            return "操作失败：未找到该记忆"

        current_text = str(existing.get("text") or "")
        metadata = self._normalize_memory_metadata(existing.get("metadata") or {})
        updated_fields: List[str] = []
        new_text: Optional[str] = None
        new_embedding: Optional[List[float]] = None

        content = str(content or "").strip()
        if content and content != current_text:
            new_text = content
            new_embedding = await self.embedding.generate(content)
            updated_fields.append("content")

        if importance is not None and str(importance).strip() != "":
            metadata["importance"] = max(0.0, min(1.0, float(importance)))
            updated_fields.append("importance")

        tags = str(tags or "").strip()
        if tags:
            metadata["tags"] = tags
            updated_fields.append("tags")

        scope = str(scope or "").strip()
        if scope:
            if scope not in VALID_SCOPES:
                return "操作失败：scope 必须是 session、user 或 global"
            identity = self._get_event_identity(event)
            metadata["scope"] = scope
            if scope == "global":
                metadata["session_id"] = "global"
                metadata["owner_user_id"] = ""
                metadata["owner_session_id"] = ""
            elif scope == "user":
                owner_user_id = identity["user_id"] or str(
                    metadata.get("owner_user_id") or metadata.get("user_id") or ""
                )
                if not owner_user_id:
                    return "操作失败：无法确定 user 作用域的 owner_user_id"
                owner_adapter = identity["adapter"] or str(
                    metadata.get("owner_adapter") or metadata.get("adapter") or ""
                )
                metadata["session_id"] = f"user:{owner_adapter}:{owner_user_id}"
                metadata["owner_user_id"] = owner_user_id
                metadata["owner_adapter"] = owner_adapter
                metadata["owner_session_id"] = (
                    identity["session_id"] or metadata.get("owner_session_id", "")
                )
            else:
                owner_session_id = identity["session_id"] or str(
                    metadata.get("owner_session_id") or metadata.get("session_id") or ""
                )
                if not owner_session_id:
                    return "操作失败：无法确定 session 作用域的 owner_session_id"
                metadata["session_id"] = owner_session_id
                metadata["owner_session_id"] = owner_session_id
                metadata["owner_user_id"] = identity["user_id"] or metadata.get(
                    "owner_user_id",
                    "",
                )
                metadata["owner_adapter"] = identity["adapter"] or metadata.get(
                    "owner_adapter",
                    "",
                )
            updated_fields.append("scope")

        if not updated_fields:
            return "操作失败：没有可更新的字段"

        metadata["updated_at"] = self._now_iso()
        metadata["updated_by"] = "update_memory"
        await self._create_export_backup(
            "before_update_memory",
            store=self.vector_store,
            collection_name=self.collection_name,
        )
        ok = await self.vector_store.update_memory(
            memory_id=memory_id,
            text=new_text,
            embedding=new_embedding,
            metadata=metadata,
        )
        if not ok:
            return "操作失败：更新记忆失败"

        return json.dumps(
            {
                "memory_id": memory_id,
                "reembedded": new_embedding is not None,
                "updated_fields": updated_fields,
            },
            ensure_ascii=False,
            indent=2,
        )

    @register.tool(
        "delete_memory",
        "按 memory_id 删除一条记忆。危险操作。",
        {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "记忆 ID"},
                "confirm": {"type": "boolean", "description": "必须为 true"},
            },
            "required": ["memory_id", "confirm"],
        },
    )
    async def delete_memory(self, event, memory_id: str, confirm: bool = False) -> str:
        """按 ID 删除记忆。"""
        if not self.vector_store:
            return "操作失败：向量记忆库尚未初始化"
        if not confirm:
            return "操作失败：请确认删除操作，将 confirm 设置为 true"
        try:
            await self._create_export_backup("before_delete")
            ok = await self.vector_store.delete(memory_id)
            return "已删除记忆" if ok else "操作失败：未找到或无法删除该记忆"
        except Exception as e:
            logger.error(f"删除记忆失败: {e}")
            return f"操作失败：删除失败: {str(e)}"

    @register.tool(
        "get_memory_by_id",
        "按 memory_id 查看单条记忆。",
        {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "记忆 ID"},
                "include_embedding": {
                    "type": "boolean",
                    "description": "是否返回 embedding，默认 false",
                },
            },
            "required": ["memory_id"],
        },
    )
    async def get_memory_by_id(
        self, event, memory_id: str, include_embedding: bool = False
    ) -> str:
        """按 ID 查看记忆。"""
        if not self.vector_store:
            return "操作失败：向量记忆库尚未初始化"
        memory = await self.vector_store.get_by_id(
            memory_id,
            include_embedding=include_embedding,
        )
        if not memory:
            return "未找到该记忆"
        return json.dumps(memory, ensure_ascii=False, indent=2)

    async def _read_all_memories(
        self,
        store: Optional[VectorStore] = None,
        include_embeddings: bool = False,
        batch_size: int = 500,
    ) -> List[Dict[str, Any]]:
        target = store or self.vector_store
        if not target:
            return []
        total = await target.count()
        memories: List[Dict[str, Any]] = []
        offset = 0
        while offset < total:
            batch = await target.get_all_memories(
                limit=batch_size,
                offset=offset,
                include_embeddings=include_embeddings,
            )
            if not batch:
                break
            memories.extend(batch)
            offset += len(batch)
        return memories

    def _build_export_path(self, prefix: str) -> Path:
        if self.data_dir is None:
            raise RuntimeError("插件数据目录未初始化")
        safe_prefix = re.sub(r"[^a-zA-Z0-9_\-.]", "_", prefix).strip("._")
        if not safe_prefix:
            safe_prefix = "export"
        if not safe_prefix.endswith(".jsonl"):
            safe_prefix = f"{safe_prefix}_{int(time.time())}.jsonl"
        export_dir = self.data_dir / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        return export_dir / safe_prefix

    async def _create_export_backup(
        self,
        reason: str,
        store: Optional[VectorStore] = None,
        collection_name: str = "",
    ) -> Path:
        if self.data_dir is None:
            raise RuntimeError("插件数据目录未初始化")
        backup_dir = self.data_dir / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        reason_parts = [reason]
        if collection_name:
            reason_parts.append(collection_name)
        safe_reason = (
            re.sub(r"[^a-zA-Z0-9_\-.]", "_", "_".join(reason_parts)).strip("._")
            or "backup"
        )
        path = backup_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_reason}.jsonl"
        await self._export_memories_to_path(path, include_embeddings=False, store=store)
        return path

    async def _export_memories_to_path(
        self,
        path: Path,
        include_embeddings: bool = False,
        store: Optional[VectorStore] = None,
        batch_size: int = 500,
        redact: bool = False,
    ) -> int:
        target = store or self.vector_store
        if not target:
            return 0
        path.parent.mkdir(parents=True, exist_ok=True)
        total = await target.count()
        exported = 0
        offset = 0
        with path.open("w", encoding="utf-8") as f:
            while offset < total:
                memories = await target.get_all_memories(
                    limit=batch_size,
                    offset=offset,
                    include_embeddings=include_embeddings and not redact,
                )
                if not memories:
                    break
                for mem in memories:
                    if redact:
                        metadata = mem.get("metadata") or {}
                        safe_mem = {
                            "id": mem.get("id"),
                            "text_redacted": True,
                            "text_length": len(str(mem.get("text") or "")),
                            "metadata": metadata,
                            "timestamp": metadata.get("timestamp", ""),
                            "type": metadata.get("type", ""),
                            "scope": metadata.get("scope", ""),
                        }
                        f.write(json.dumps(safe_mem, ensure_ascii=False) + "\n")
                    else:
                        f.write(json.dumps(mem, ensure_ascii=False) + "\n")
                    exported += 1
                offset += len(memories)
        return exported

    def _resolve_plugin_data_path(self, path: str) -> Path:
        if self.data_dir is None:
            raise RuntimeError("插件数据目录未初始化")
        candidate = (self.data_dir / path).resolve()
        root = self.data_dir.resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("只能读取插件数据目录内的文件")
        if not candidate.exists():
            raise FileNotFoundError(f"文件不存在: {candidate}")
        return candidate

    def _is_valid_collection_name(self, name: str) -> bool:
        return bool(re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{2,62}$", name or ""))

    def _is_collection_not_found_error(self, error: Exception) -> bool:
        """兼容不同 Chroma 版本的集合不存在错误文本。"""
        message = str(error).lower()
        return (
            "does not exist" in message
            or "not found" in message
            or "doesn't exist" in message
            or "not exists" in message
        )

    def _persist_plugin_config(self, updates: Dict[str, Any]):
        """轻量持久化插件配置，避免 activate_collection 重启后失效。"""
        config_path = get_config_path() / "plugins" / f"{PLUGIN_ID}.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        cfg: Dict[str, Any] = {}
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                cfg = {}
        cfg.update(updates)
        config_path.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )
