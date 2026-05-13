"""
向量记忆库插件 v1.1
提供基于语义相似度的长期记忆检索能力

新增功能：
- 记忆重要性评分
- 自动上下文注入
- 每日记忆反思
- 智能消息过滤
"""

import asyncio
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from core.plugin import BasePlugin, logger, on, Priority, register
from core.prompt_manager import Prompt
from core.chat import KiraMessageEvent, KiraMessageBatchEvent
from core.chat.message_elements import Text
from core.provider import LLMRequest, EmbeddingModelClient

from .embeddings import EmbeddingFactory, BaseEmbedding
from .vector_store import ChromaVectorStore, VectorStore
from .importance import ImportanceScorerFactory, BaseImportanceScorer
from .filter import MessageFilter
from .injector import ContextInjector
from .reflection import ReflectionManager


class ProviderEmbeddingAdapter(BaseEmbedding):
    """Adapt a host EmbeddingModelClient to the plugin's embedding interface."""

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
    """
    向量记忆库插件 v1.1

    工具：
    - search_memory: 语义搜索历史记忆
    - vector_memory_add: 手动添加长篇记忆
    - get_memory_stats: 获取统计信息
    - summarize_memories: 总结记忆
    - trigger_reflection: 手动触发记忆反思
    """

    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)

        self.enabled: bool = True
        self.embedding: Optional[BaseEmbedding] = None
        self.vector_store: Optional[VectorStore] = None
        self.importance_scorer: Optional[BaseImportanceScorer] = None
        self.message_filter: Optional[MessageFilter] = None
        self.context_injector: Optional[ContextInjector] = None
        self.reflection_manager: Optional[ReflectionManager] = None

        # 插件数据目录
        self.data_dir: Optional[Path] = None

        # 配置项
        self.auto_record: bool = True
        self.min_text_length: int = 10
        self.max_memory_count: int = 10000
        self.default_top_k: int = 5

        # 定时任务调度器
        self._scheduler = None

    async def initialize(self):
        """初始化插件"""
        # 检查是否启用
        self.enabled = self.plugin_cfg.get("enabled", True)
        if not self.enabled:
            logger.info("向量记忆库插件已禁用")
            return

        # 使用主项目提供的插件专属数据目录，避免和其他插件混用路径
        self.data_dir = self.ctx.get_plugin_data_dir() or (Path("data/plugin_data") / "vector_memory")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 读取配置
        self._normalize_embedding_config()
        self.auto_record = self.plugin_cfg.get("auto_record", True)
        self.min_text_length = self.plugin_cfg.get("min_text_length", 10)
        self.max_memory_count = self.plugin_cfg.get("max_memory_count", 10000)
        self.default_top_k = self.plugin_cfg.get("search_top_k", 5)

        # ========== 初始化 Embedding ==========
        await self._init_embedding()
        if not self.enabled:
            return

        # ========== 初始化向量存储 ==========
        await self._init_vector_store()
        if not self.enabled:
            return

        # ========== 初始化重要性评分器 ==========
        self._init_importance_scorer()

        # ========== 初始化消息过滤器 ==========
        self._init_message_filter()

        # ========== 初始化上下文注入器 ==========
        self._init_context_injector()

        # ========== 初始化记忆反思管理器 ==========
        self._init_reflection_manager()

        # ========== 启动定时任务 ==========
        await self._start_scheduler()

        logger.info("向量记忆库插件 v1.1 初始化完成")

    async def _init_embedding(self):
        """初始化 Embedding"""
        embedding_source = str(self.plugin_cfg.get("embedding_source") or "system_default").strip()

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

    def _resolve_embedding_client(self, embedding_source: str) -> EmbeddingModelClient:
        """解析并校验插件可用的 embedding client。"""
        if embedding_source == "system_default":
            client = self.ctx.get_default_embedding_client()
            if client is None:
                raise ValueError("主项目默认 embedding 未配置，请先在系统模型配置中设置 default_embedding")
            return client

        if embedding_source == "custom_model":
            model_uuid = str(self.plugin_cfg.get("embedding_model_uuid") or "").strip()
            if not model_uuid:
                raise ValueError("embedding_source 为 custom_model 时，必须先在插件配置里选择一个 embedding 模型")

            try:
                client = self.ctx.get_embedding_client(model_uuid)
            except Exception as e:
                raise ValueError(f"获取指定 embedding 模型失败: {e}") from e

            if client is None:
                raise ValueError("未找到所选 embedding 模型，请重新在主项目模型配置中选择可用的 embedding 模型")

            return client

        raise ValueError(f"不支持的 embedding_source: {embedding_source}")

    def _normalize_embedding_config(self):
        """将旧配置迁移到新的云端 embedding 组合方案。"""
        embedding_source = str(self.plugin_cfg.get("embedding_source") or "").strip()
        if embedding_source:
            return

        legacy_provider = self.plugin_cfg.get("embedding_provider")
        legacy_model = self.plugin_cfg.get("embedding_model")
        legacy_api_key = self.plugin_cfg.get("openai_api_key")
        legacy_base_url = self.plugin_cfg.get("openai_base_url")

        if legacy_provider or legacy_model or legacy_api_key or legacy_base_url:
            self.plugin_cfg["embedding_source"] = "legacy"
            logger.warning("检测到旧的 embedding 配置，已自动迁移到主项目云端 embedding 方案。")

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
                            logger.info(f"使用系统配置的 API Key (provider: {provider_id})")
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

    async def _init_vector_store(self):
        """初始化向量存储"""
        try:
            if self.data_dir is None:
                raise RuntimeError("插件数据目录未初始化")
            self.vector_store = ChromaVectorStore(
                persist_dir=str(self.data_dir / "chroma_db"),
                collection_name="kira_memories",
            )
            memory_count = await self.vector_store.count()
            logger.info(f"向量存储初始化成功，现有记忆: {memory_count} 条")
        except Exception as e:
            logger.error(f"向量存储初始化失败: {e}")
            self.enabled = False

    def _init_importance_scorer(self):
        """初始化重要性评分器"""
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
        """初始化消息过滤器"""
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
        """初始化上下文注入器"""
        if self.plugin_cfg.get("auto_injection_enabled", True):
            self.context_injector = ContextInjector(
                embedding=self.embedding,
                vector_store=self.vector_store,
                threshold=self.plugin_cfg.get("injection_threshold", 0.75),
                top_k=self.plugin_cfg.get("injection_top_k", 2),
                cooldown=self.plugin_cfg.get("injection_cooldown", 60),
            )
            logger.info("自动上下文注入已启用")
        else:
            self.context_injector = None

    def _init_reflection_manager(self):
        """初始化记忆反思管理器"""
        if self.plugin_cfg.get("reflection_enabled", True):
            self.reflection_manager = ReflectionManager(
                embedding=self.embedding,
                vector_store=self.vector_store,
                llm_api=self.ctx.llm_api,
                delete_raw=self.plugin_cfg.get("reflection_delete_raw", False),
                min_memories=self.plugin_cfg.get("reflection_min_memories", 5),
            )
            logger.info("记忆反思管理器已启用")
        else:
            self.reflection_manager = None

    async def _start_scheduler(self):
        """启动定时任务调度器"""
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

            # 添加每日反思任务
            reflection_hour = self.plugin_cfg.get("reflection_hour", 3)

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
        """每日反思任务"""
        if self.reflection_manager:
            await self.reflection_manager.run_daily_reflection()

    async def terminate(self):
        """清理资源"""
        # 停止定时任务
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

        # 关闭向量存储
        if self.vector_store:
            await self.vector_store.close()
            self.vector_store = None

        self.embedding = None
        self.importance_scorer = None
        self.message_filter = None
        self.context_injector = None
        self.reflection_manager = None

        logger.info("向量记忆库插件已关闭")

    # ========== 消息钩子 ==========

    @on.im_message(priority=Priority.LOW)
    async def record_message(self, event: KiraMessageEvent, *_, **__):
        """在消息进入处理链后记录原始消息。"""
        if not self.enabled or not self.auto_record:
            return

        if event.process_strategy == "discard":
            return

        await self._record_message(event)

    @on.llm_request(priority=Priority.LOW)
    async def inject_memory(self, event: KiraMessageBatchEvent, req: LLMRequest, *_):
        """在 LLM 请求前检索相关记忆并注入到系统提示中。"""
        if not self.enabled or not self.context_injector:
            return

        query = self._build_batch_query(event)
        if not query:
            return

        try:
            result = await self.context_injector.try_inject(
                query=query, session_id=event.sid
            )
            if not result.injected:
                return

            injection_content = self.context_injector.format_injection(result.memories)
            if not injection_content:
                return

            for prompt in req.system_prompt:
                if prompt.name == "memory":
                    if prompt.content:
                        prompt.content += "\n"
                    prompt.content += injection_content
                    break
            else:
                req.system_prompt.append(
                    Prompt(
                        content=injection_content,
                        name="vector_memory",
                        source="vector_memory",
                    )
                )

            logger.debug(f"上下文注入成功: {result.reason}")
        except Exception as e:
            logger.error(f"消息注入失败: {e}")

    async def _record_message(self, event: KiraMessageEvent):
        """记录消息到向量库"""
        try:
            # 提取文本
            text = self._extract_text(event)
            if not text:
                return

            # 过滤检查
            if self.message_filter:
                should_record, reason = self.message_filter.should_record(text)
                if not should_record:
                    logger.debug(f"消息被过滤: {reason}")
                    return
                # 提取关键信息
                text = self.message_filter.extract_key_info(text)
            else:
                # 基础长度检查
                if len(text) < self.min_text_length:
                    return

            # 计算重要性评分
            importance = 0.3
            if self.importance_scorer:
                importance = await self.importance_scorer.score(text)

            # 生成 Embedding
            embedding = await self.embedding.generate(text)

            # 构建会话 ID
            session_id = self._build_session_id(event)

            # 存储
            metadata = {
                "session_id": session_id,
                "user_id": getattr(event.message.sender, "user_id", ""),
                "user_nickname": getattr(event.message.sender, "nickname", ""),
                "platform": getattr(event.adapter, "platform", ""),
                "adapter": getattr(event.adapter, "name", ""),
                "timestamp": event.timestamp,
                "group_id": getattr(event.message.group, "group_id", "") if event.message.group else "",
                "group_name": getattr(event.message.group, "group_name", "") if event.message.group else "",
                "message_id": event.message.message_id,
                "importance": importance,
                "type": "raw",
            }

            memory_id = await self.vector_store.add(
                text=text, embedding=embedding, metadata=metadata
            )

            logger.debug(f"记录消息: {memory_id}, importance: {importance:.2f}")

            # 检查是否需要清理
            current_count = await self.vector_store.count()
            if current_count > self.max_memory_count:
                time_weight = self.plugin_cfg.get("cleanup_time_weight", 0.3)
                importance_weight = self.plugin_cfg.get(
                    "cleanup_importance_weight", 0.7
                )

                await self.vector_store.cleanup_smart(
                    max_count=self.max_memory_count,
                    time_weight=time_weight,
                    importance_weight=importance_weight,
                )

        except Exception as e:
            logger.error(f"记录消息失败: {e}")

    def _extract_text(self, event: KiraMessageEvent) -> str:
        """从消息事件中提取文本"""
        return self._extract_text_from_chain(event.message.chain)

    def _extract_text_from_chain(self, chain) -> str:
        text_parts: List[str] = []
        for elem in chain:
            if isinstance(elem, Text):
                text_parts.append(elem.text)

        text = " ".join(part.strip() for part in text_parts if part and part.strip()).strip()
        return text

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

    def _build_session_id(self, event: KiraMessageEvent) -> str:
        """构建会话 ID"""
        return event.session.sid

    # ========== 工具方法 ==========

    @register.tool(
        "search_memory",
        "搜索历史记忆，根据语义相似度返回相关的历史对话。当用户询问'之前说过什么'、'记得吗'、'历史对话'等问题时使用此工具。",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询，描述你想找的记忆内容",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回最相关的记忆条数，默认 5",
                },
                "session_id": {
                    "type": "string",
                    "description": "仅搜索指定会话的记忆（可选）",
                },
                "include_summaries": {
                    "type": "boolean",
                    "description": "是否包含记忆摘要，默认 true",
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
        """搜索历史记忆"""
        if not self.enabled:
            return "向量记忆库未启用"

        try:
            if top_k is None:
                top_k = self.default_top_k
            top_k = min(max(1, top_k), 20)

            import re
            
            # Check if query is actually a memory ID
            if re.match(r'^mem_\d+_[a-f0-9]+$', query.strip()):
                mem_id = query.strip()
                # ChromaDB get by ID
                id_results = await asyncio.to_thread(
                    self.vector_store.collection.get,
                    ids=[mem_id],
                    include=["documents", "metadatas"],
                )
                if id_results and id_results.get("ids") and len(id_results["ids"]) > 0:
                    results = [
                        {
                            "id": id_results["ids"][0],
                            "text": id_results["documents"][0],
                            "metadata": id_results["metadatas"][0],
                            "similarity": 1.0  # Exact match
                        }
                    ]
                else:
                    results = []
            else:
                query_embedding = await self.embedding.generate(query)
    
                filter_dict = None
                if session_id:
                    filter_dict = {"session_id": session_id}
    
                results = await self.vector_store.search(
                    embedding=query_embedding, top_k=top_k, filter_dict=filter_dict
                )

            # 可选过滤摘要
            if not include_summaries:
                results = [
                    r for r in results if r.get("metadata", {}).get("type") != "summary"
                ]

            if not results:
                return "未找到相关记忆"

            output = [f"找到 {len(results)} 条相关记忆：\n"]

            for i, result in enumerate(results, 1):
                metadata = result.get("metadata", {})
                timestamp = metadata.get("timestamp", 0)
                mem_type = metadata.get("type", "raw")
                importance = metadata.get("importance", 0)

                if timestamp:
                    time_str = datetime.fromtimestamp(timestamp).strftime(
                        "%Y-%m-%d %H:%M"
                    )
                else:
                    time_str = "未知时间"

                user = metadata.get("user_nickname", "未知用户")
                text = result.get("text", "")
                similarity = result.get("similarity", 0)

                if len(text) > 200:
                    text = text[:200] + "..."

                type_tag = "[摘要] " if mem_type == "summary" else ""

                output.append(
                    f"{i}. {type_tag}[{time_str}] {user}: {text}\n"
                    f"   相似度: {similarity:.2%} | 重要性: {importance:.2f}"
                )

            return "\n".join(output)

        except Exception as e:
            logger.error(f"搜索记忆失败: {e}")
            return f"搜索失败: {str(e)}"

    @register.tool(
        "vector_memory_add",
        "手动添加长篇记忆到向量记忆库。用于记录大段的情感、故事回忆或复杂的上下文语境。注意区分职责：\n"
        "- 碎片化重点（如“晚安”）可优先用其他核心记忆系统；\n"
        "- 客观属性/节点事件（如“我叫Kira”）优先用画像系统；\n"
        "当用户说'记住这个'且内容较长/无特定类别时使用本工具。",
        {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "要记住的内容"},
                "importance": {
                    "type": "number",
                    "description": "重要性评分（0-1），默认自动评估",
                },
                "tags": {
                    "type": "string",
                    "description": "标签，用于分类（可选）",
                },
            },
            "required": ["content"],
        },
    )
    async def vector_memory_add(
        self, event, content: str, importance: float = None, tags: str = ""
    ) -> str:
        """手动添加记忆"""
        if not self.enabled:
            return "向量记忆库未启用"

        try:
            if len(content) < 5:
                return "内容太短，请提供更详细的信息"

            # 自动或手动重要性评分
            if importance is None:
                if self.importance_scorer:
                    importance = await self.importance_scorer.score(content)
                else:
                    importance = 0.7  # 手动添加默认较高
            else:
                importance = max(0.0, min(1.0, importance))

            embedding = await self.embedding.generate(content)

            metadata = {
                "session_id": "manual",
                "user_id": "manual",
                "user_nickname": "用户",
                "platform": "manual",
                "adapter": "manual",
                "timestamp": int(time.time()),
                "tags": tags,
                "type": "manual",
                "importance": importance,
            }

            memory_id = await self.vector_store.add(
                text=content, embedding=embedding, metadata=metadata
            )

            return f"已记住！记忆 ID: {memory_id}，重要性: {importance:.2f}"

        except Exception as e:
            logger.error(f"添加记忆失败: {e}")
            return f"添加失败: {str(e)}"

    @register.tool(
        "get_memory_stats",
        "获取向量记忆库的统计信息。",
        {"type": "object", "properties": {}},
    )
    async def get_memory_stats(self, event) -> str:
        """获取记忆统计"""
        if not self.enabled:
            return "向量记忆库未启用"

        try:
            count = await self.vector_store.count()
            max_count = self.max_memory_count

            # 统计类型分布
            all_memories = await self.vector_store.get_all_memories(limit=10000)

            type_counts = {"raw": 0, "summary": 0, "manual": 0}
            importance_sum = 0.0

            for mem in all_memories:
                metadata = mem.get("metadata", {})
                mem_type = metadata.get("type", "raw")
                type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
                importance_sum += float(metadata.get("importance", 0.3))

            avg_importance = importance_sum / count if count > 0 else 0

            stats = {
                "记忆总数": count,
                "容量上限": max_count,
                "使用率": f"{count / max_count * 100:.1f}%",
                "原始记忆": type_counts.get("raw", 0),
                "摘要记忆": type_counts.get("summary", 0),
                "手动记忆": type_counts.get("manual", 0),
                "平均重要性": f"{avg_importance:.2f}",
                "自动记录": "开启" if self.auto_record else "关闭",
                "智能过滤": "开启" if self.message_filter else "关闭",
                "自动注入": "开启" if self.context_injector else "关闭",
                "每日反思": "开启" if self.reflection_manager else "关闭",
                "重要性模式": self.plugin_cfg.get("importance_mode", "rule"),
            }

            output = ["向量记忆库统计 (v1.1)：\n"]
            for key, value in stats.items():
                output.append(f"- {key}: {value}")

            return "\n".join(output)

        except Exception as e:
            logger.error(f"获取统计失败: {e}")
            return f"获取统计失败: {str(e)}"

    @register.tool(
        "trigger_reflection",
        "手动触发记忆反思，将碎片化记忆整理为摘要。",
        {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "反思多少天内的记忆，默认 7 天",
                },
                "session_id": {
                    "type": "string",
                    "description": "仅反思指定会话的记忆（可选）",
                },
            },
        },
    )
    async def trigger_reflection(self, event, days: int = 7, session_id: str = None) -> str:
        """手动触发记忆反思"""
        if not self.enabled:
            return "向量记忆库未启用"

        try:
            manager = self.reflection_manager or ReflectionManager(
                embedding=self.embedding,
                vector_store=self.vector_store,
                llm_api=self.ctx.llm_api,
                delete_raw=self.plugin_cfg.get("reflection_delete_raw", False),
                min_memories=self.plugin_cfg.get("reflection_min_memories", 5),
            )
            result = await manager.manual_reflection(
                session_id=session_id, days=days
            )
            return result
        except Exception as e:
            logger.error(f"手动反思失败: {e}\n{traceback.format_exc()}")
            return f"反思失败: {str(e)}"

    @register.tool(
        "summarize_memories",
        "使用 AI 总结一段时间内的记忆。",
        {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "总结最近多少天的记忆，默认 7 天",
                },
                "session_id": {
                    "type": "string",
                    "description": "仅总结指定会话的记忆（可选）",
                },
            },
        },
    )
    async def summarize_memories(self, event, days: int = 7, session_id: str = None) -> str:
        """总结记忆"""
        if not self.enabled:
            return "向量记忆库未启用"

        try:
            manager = self.reflection_manager or ReflectionManager(
                embedding=self.embedding,
                vector_store=self.vector_store,
                llm_api=self.ctx.llm_api,
                delete_raw=self.plugin_cfg.get("reflection_delete_raw", False),
                min_memories=self.plugin_cfg.get("reflection_min_memories", 5),
            )
            return await manager.summarize_recent_memories(days=days, session_id=session_id)

        except Exception as e:
            logger.error(f"总结记忆失败: {e}\n{traceback.format_exc()}")
            return f"总结失败: {str(e)}"

    @register.tool(
        "clear_memories",
        "清空向量记忆库中的所有记忆（危险操作！）",
        {
            "type": "object",
            "properties": {
                "confirm": {"type": "boolean", "description": "确认清空，必须为 true"},
                "keep_summaries": {
                    "type": "boolean",
                    "description": "是否保留摘要记忆，默认 true",
                },
            },
            "required": ["confirm"],
        },
    )
    async def clear_memories(
        self, event, confirm: bool = False, keep_summaries: bool = True
    ) -> str:
        """清空记忆"""
        if not self.enabled:
            return "向量记忆库未启用"

        if not confirm:
            return "请确认清空操作（设置 confirm 为 true）"

        try:
            if keep_summaries:
                # 仅删除非摘要记忆
                all_memories = await self.vector_store.get_all_memories(limit=100000)

                ids_to_delete = []
                for mem in all_memories:
                    if mem.get("metadata", {}).get("type") != "summary":
                        ids_to_delete.append(mem.get("id"))

                if ids_to_delete:
                    for mem_id in ids_to_delete:
                        await self.vector_store.delete(mem_id)

                return f"已清空 {len(ids_to_delete)} 条记忆（保留了摘要）"
            else:
                count_before = await self.vector_store.count()
                await self.vector_store.clear()
                return f"已清空 {count_before} 条记忆"

        except Exception as e:
            logger.error(f"清空记忆失败: {e}")
            return f"清空失败: {str(e)}"
