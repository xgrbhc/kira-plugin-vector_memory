# vector_memory 向量记忆库插件

`vector_memory` 是 KiraAI 的长期向量记忆插件。它复用主项目中已配置的 Embedding 模型，并使用 ChromaDB 在本地持久化记忆。

## 主要能力

- 自动记录有价值的对话文本。
- 按语义相似度搜索历史记忆。
- 在 LLM 请求前自动注入相关记忆；动态记忆会独立追加到系统提示词末尾，尽量保护前缀缓存。
- 按 `session`、`user`、`global` 分层隔离记忆，减少串忆。
- 支持记忆重要性评分、每日反思摘要和智能清理。
- 提供诊断、脱敏导出、导入去重、重建索引续跑、summary 作用域迁移、单条编辑、切换集合、按 ID 查看/删除等维护工具。

## 数据与配置位置

插件配置文件：

```text
data/config/plugins/vector_memory.json
```

插件数据目录：

```text
data/plugin_data/vector_memory/
```

主要数据：

```text
data/plugin_data/vector_memory/chroma_db/   # ChromaDB 持久化目录
data/plugin_data/vector_memory/meta.json    # 插件数据契约与维度信息
data/plugin_data/vector_memory/backups/     # 危险操作前自动导出的备份
data/plugin_data/vector_memory/exports/     # 手动导出文件
```

升级或迁移前，建议完整备份 `data/plugin_data/vector_memory/` 和 `data/config/plugins/vector_memory.json`。

## 关键配置

- `usage_prompt`：可编辑的 LLM 使用提示词，用来指导模型什么时候调用记忆工具。
- `embedding_source`：`system_default` 或 `custom_model`。
- `embedding_model_uuid`：当 `embedding_source=custom_model` 时选择具体 embedding 模型。
- `collection_name`：当前使用的 Chroma 集合名，默认 `kira_memories`。
- `memory_scope_mode`：自动注入作用域策略。
  - `strict_session`：只注入当前会话记忆。
  - `user_shared`：注入当前会话、当前用户、全局记忆，默认推荐。
  - `global_shared`：更偏全局召回，但串忆风险更高。
- `auto_record`：是否自动记录消息。
- `search_similarity_threshold`：只影响 `search_memory`，默认 `0.0` 表示不额外过滤。
- `auto_injection_enabled`：是否自动注入相关记忆。
- `injection_rerank_enabled`：自动注入候选是否启用轻量重排，默认开启。
- `rerank_similarity_weight` / `rerank_importance_weight` / `rerank_recency_weight`：搜索和注入重排权重，默认 `0.70 / 0.20 / 0.10`。
- `reflection_enabled`：是否启用每日反思。

## 记忆作用域

新写入记忆会包含以下 metadata：

- `scope`：`session`、`user` 或 `global`。
- `owner_user_id`：归属用户。
- `owner_session_id`：归属会话。
- `owner_adapter`：归属适配器。

自动记录的原始对话默认使用 `scope=session`。手动添加记忆时，`vector_memory_add` 支持指定 `scope`，默认 `user`。

## 工具

- `search_memory`：按语义搜索历史记忆。
- `vector_memory_add`：手动添加长期记忆，支持 `scope`。
- `get_memory_stats`：查看统计信息。
- `summarize_memories`：总结近期记忆。
- `trigger_reflection`：手动触发记忆反思。
- `clear_memories`：安全清空记忆，危险操作；会先自动备份，再切换到新集合，不会原地批量删除旧集合。
- `diagnose_memory_store`：诊断当前记忆库状态；可用 `deep_check=true` 对单个集合做深度检查。
- `list_memory_collections`：只读列出 Chroma 集合、记忆数量和 embedding 维度，适合切换集合或更换模型前使用。
- `export_memories`：导出 JSONL；可用 `source_collection` 指定导出某个已存在集合，`redact=true` 时不导出完整正文和 embedding。
- `import_memories`：从 JSONL 导入并重新生成 embedding，默认 `mode=dedupe` 避免重复导入。
- `reindex_memories`：用当前 embedding 重建到新集合；目标集合非空时必须显式 `resume=true` 才会续跑。
- `handle_embedding_model_change`：处理更换 embedding 模型后的维度变化；可只诊断、创建新空集合并切换，或转入重建索引流程。
- `plan_summary_scope_migration`：只读分析 summary 记忆是否适合迁移到 `user`、`global` 或 `session` 作用域。
- `migrate_summary_scope`：迁移 summary 记忆作用域；默认 `dry_run=true`，真实写入必须确认短语 `MIGRATE_SUMMARY_SCOPE`。
- `update_memory`：按 ID 编辑单条记忆；只改 metadata 不重算 embedding，改正文会重新生成 embedding。
- `activate_collection`：切换当前集合；切换前会备份当前集合和目标集合，并拒绝切换到不存在的集合。
- `delete_memory_collection`：删除非当前集合，极高风险；删除前会自动备份目标集合。
- `delete_memory`：按 ID 删除记忆。
- `get_memory_by_id`：按 ID 查看记忆。

危险工具需要 `confirm=true`，并会先导出备份。`clear_memories` 还需要 `confirm_phrase=CLEAR_VECTOR_MEMORY`，避免 LLM 或误操作直接清空主集合。`migrate_summary_scope` 真实写入需要 `confirm_phrase=MIGRATE_SUMMARY_SCOPE`。`delete_memory_collection` 还需要 `confirm_phrase=DELETE_VECTOR_MEMORY_COLLECTION`，并且禁止删除当前激活集合。`activate_collection` 会在写入配置前确认目标集合存在，避免拼错集合名后创建空集合。

维护工具有共用锁：当 `import_memories`、`reindex_memories`、`activate_collection`、`clear_memories`、`delete_memory_collection` 其中一个正在运行时，其他维护操作会返回“已有维护操作正在进行，请稍后再试”。这样可以避免重建索引时又切换集合、导入数据或删除集合。

`import_memories` 支持 `dry_run=true`，此时只分析 JSONL 和目标集合，不写入、不生成 embedding。导入模式包括 `append`、`dedupe`、`new_collection_only`，默认推荐 `dedupe`。

`reindex_memories` 支持 `dry_run=true` 和 `resume=true`。`dry_run=true` 只统计预计写入数量；`resume=true` 用于半成品目标集合续跑，会按指纹跳过已经存在的记忆。大集合建议使用 `batch_size=16`，这样遇到内容过滤时，整批回退成本更低。

`reindex_memories` 会按批调用 embedding，并在日志里输出 `processed/total`、`rebuilt`、`skipped_duplicate`、`skipped_embedding`、`failed` 和 `elapsed`。如果批量 embedding 被服务商内容过滤，插件会逐条回退，失败记录会边处理边写入 `exports/*failed_embeddings*.jsonl`，避免任务中断后丢失排查线索。不要在真实聊天里连续测试多个重建任务；如果只是验证危险工具是否可调用，建议先使用很小的测试集合。

`search_memory` 和自动注入会使用轻量重排函数，但不会改变 Chroma 距离度量，也不会改写任何旧向量。输出里仍保留原始相似度，方便判断阈值是否合适。

自动注入采用缓存友好策略：插件不会再修改已有的 `memory` prompt，而是新增一个独立的 `vector_memory` prompt 并追加到 `system_prompt` 末尾。这样可以让主项目中较稳定的角色、人设、工具和长期规则尽量保持在前面，减少动态记忆对 provider prompt prefix cache 的破坏。不同服务商的缓存策略不完全一致，因此这只能降低影响，不能保证所有请求都命中缓存。

导出、备份和失败记录可能包含私人记忆。即使使用 `redact=true`，metadata 中仍可能包含会话、用户或时间信息，请妥善保存这些文件。

如果当前激活集合损坏或无法初始化，插件会进入维护模式，但 `diagnose_memory_store`、指定集合的 `export_memories`、指定新集合的 `import_memories`、`activate_collection` 仍可用于恢复。

## 更换 Embedding 模型

如果更换了主项目默认 embedding 或插件自定义 embedding，旧集合可能出现维度不一致。例如旧集合是 4096 维，新模型是 2560 维。插件会进入维护模式，避免把不同维度的向量写进同一个集合。

用户不需要手改代码，推荐通过 LLM 调用：

```text
handle_embedding_model_change(action="diagnose")
```

如果旧记忆只是测试数据，或不需要保留旧向量内容，可以创建新的空集合并切换：

```text
handle_embedding_model_change(action="create_empty_collection", confirm=true)
```

旧集合会保留，不会删除。插件会自动生成类似 `kira_memories_2560d` 的集合名，也可以显式传入 `new_collection_name`。

如果需要保留旧记忆内容，应重建索引：

```text
handle_embedding_model_change(
  action="reindex_to_new_collection",
  new_collection_name="kira_memories_2560d",
  confirm=true,
  batch_size=16
)
```

重建完成并确认可用后，再调用 `activate_collection` 切换到新集合。

说明：只改插件时，WebUI 的 `Chroma 集合名称` 暂时不能做成真正动态读取 Chroma 的下拉框；主项目配置表单目前只支持 schema 里的静态选项。插件侧提供 `list_memory_collections` 和 `handle_embedding_model_change`，让用户可以通过聊天完成集合查看、创建、重建和切换。

## 旧数据兼容

插件会尽量兼容旧版 `data/plugin_data/vector_memory/chroma_db`：

1. 如果旧库没有 `meta.json`，首次启动会先导出备份，再生成 `meta.json`。
2. 旧记忆会补齐 `scope`、`owner_user_id`、`owner_session_id`、`owner_adapter`。
3. 自动迁移只更新 metadata，不修改 document，不重算旧 embedding。
4. 如果检测到当前 embedding 维度和旧集合维度不一致，插件会进入维护模式，禁止新增、搜索、自动注入和反思，但仍允许诊断、导出和重建索引。

如需更换 embedding 模型，推荐流程：

1. 调用 `export_memories` 导出旧记忆。
2. 调用 `reindex_memories` 重建到新集合。
3. 检查新集合可用后，调用 `activate_collection` 切换集合。
4. 保留旧集合一段时间，确认无误后再手动清理。

## 依赖

```text
APScheduler>=3.10,<4
chromadb>=0.4.0
```

插件不依赖 Mem0，也不会自动安装 Mem0。
