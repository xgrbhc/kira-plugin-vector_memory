# vector_memory 向量记忆库插件

一个 KiraAI 的简易长期记忆插件。只使用主项目配置好的云端 Embedding，使用 ChromaDB 做本地持久化向量存储。

## 它做什么

- 自动记录重要对话信息
- 语义搜索历史记忆
- 在 LLM 请求前注入相关记忆
- 按重要性整理、总结和清理记忆

## 依赖

- 必需：`chromadb>=0.4.0`
- 可选：`APScheduler>=3.10,<4`，用于每日反思

如果通过 KiraAI 的 GitHub 插件安装器安装，通常会自动安装 `requirements.txt`。

## 配置

配置文件：`data/config/vector_memory.json`

只保留两种 Embedding 方式：

- `system_default`：使用主项目默认 embedding
- `custom_model`：手动选择主项目里的 embedding 模型

最小可用配置：

```json
{
  "enabled": true,
  "embedding_source": "system_default",
  "embedding_model_uuid": "",
  "auto_record": true,
  "min_text_length": 10,
  "search_top_k": 5,
  "max_memory_count": 10000
}
```

常用开关：

- `auto_record`：自动记录消息
- `smart_filter_enabled`：智能过滤无意义消息
- `auto_injection_enabled`：自动上下文注入
- `reflection_enabled`：每日记忆反思

## 工具

- `search_memory`：按语义搜索记忆
- `vector_memory_add`：手动添加记忆
- `get_memory_stats`：查看统计信息
- `summarize_memories`：总结一段时间内的记忆
- `trigger_reflection`：手动触发反思
- `clear_memories`：清空记忆库，危险操作

## 数据存储

插件数据默认放在：

```text
data/plugin_data/vector_memory/chroma_db
```

迁移或备份时，至少保留：

- `data/config/vector_memory.json`
- `data/plugin_data/vector_memory/`

## 安装后检查

启动后，日志里通常会看到：

- Embedding 初始化成功
- 向量存储初始化成功
- 重要性评分器初始化成功
- 自动上下文注入已启用
- 记忆反思管理器已启用

## 常见问题

### 为什么不提供本地模型选项？

因为这个插件的定位就是复用主项目的云端 embedding，不再维护独立的本地模型入口。

### 为什么提示 default_embedding 未配置？

因为当前选了 `system_default`，但主项目还没有设置默认 embedding。请先在主项目模型配置里补上。

### 为什么反思任务没执行？

先检查是否安装了 `APScheduler`。没有它时，反思只支持手动触发。

### 记忆会无限增长吗？

不会。插件会按照 `max_memory_count` 做清理，但仍建议定期备份数据。

## 适合谁

- 需要长期记住用户偏好、身份信息、项目上下文的场景
- 需要从历史对话里快速找回信息的场景
- 想要增强对话连续性，但不想引入复杂外部存储的场景

## 项目结构

```text
data/plugins/vector_memory/
├── manifest.json
├── schema.json
├── main.py
├── embeddings.py
├── vector_store.py
├── importance.py
├── filter.py
├── injector.py
└── reflection.py
```

