# Anima Prompt ComfyUI 节点实施计划

## 目标

构建一个专门为 Anima DiT 绘图模型生成 Prompt 的 ComfyUI 自定义节点包。
节点使用 LLM 和 `data/tags.csv` 中的标签分类树，从用户输入中选择有效标签，
生成自然语言描述，并将两部分组合成最终 Prompt。

输出示例：

```text
1girl,solo,A girl is standing in a classroom, looking toward the window.
```

标签组使用英文逗号分隔。自然语言描述允许正常使用逗号，不限制为无逗号句子。

## 已确认决策

- 本地 LLM 使用 GGUF 模型，通过 llama.cpp 和 `llama-cpp-python` 加载。
- API 加载器使用 OpenAI API。
- 两种加载器统一输出 ComfyUI 类型 `ANIMA_LLM`。
- 标签组与自然语言描述使用英文逗号连接。
- OpenAI 密钥从 `OPENAI_API_KEY` 环境变量读取，不写入 ComfyUI 工作流。
- 不把完整标签库一次性发送给 LLM，只发送相关分类分支和受限候选集。

## 当前数据

`data/tags.csv` 当前包含：

- 163,253 个唯一标签。
- 21,692 个带分类路径的标签。
- 141,561 个无分类路径的标签。
- 最多七级分类。
- 字段为 `tag`、`category`、`post_count`、`classification_1` 至
  `classification_7`。
- 只保留 General、Character、Species；Artist、Copyright、Meta、Lore、
  Contributor 已移除。

## 范围

### 包含

- ComfyUI 节点注册。
- CSV 解析、校验、索引和缓存。
- 分类树构建。
- 本地 GGUF LLM 加载。
- OpenAI LLM 连接配置。
- 用户意图分析、候选标签检索、标签选择和自然语言生成。
- Prompt 校验与拼装。
- 单元测试和安装文档。

### 不包含

- Anima DiT 模型加载和采样。
- LLM 训练或微调。
- 在线下载或更新标签数据。
- 自定义 JavaScript 标签浏览器等前端扩展。
- OpenAI 以外的专用 API 适配。

## 节点设计

### Local LLM Loader

从指定的 ComfyUI 模型目录加载 GGUF 模型并输出 `ANIMA_LLM`。

计划提供以下控制项：

- GGUF 模型名称。
- 上下文长度。
- GPU 加载层数。
- CPU 线程数。
- Batch 大小。
- 可选随机种子。

配置未改变时复用已加载模型，并在 ComfyUI 卸载模型时释放资源。

### OpenAI LLM Loader

创建 OpenAI 后端的 `ANIMA_LLM`，加载节点时不立即发送请求。

计划提供以下控制项：

- OpenAI 模型名称。
- API Key 环境变量名，默认 `OPENAI_API_KEY`。
- 请求超时。
- 最大重试次数。

### Anima Prompt Generator

接收 `ANIMA_LLM` 和用户文本，输出：

- `prompt`：标签组与自然语言描述拼接后的最终文本。
- `tag_group`：经过校验的逗号分隔标签组。
- `description`：LLM 生成的自然语言描述。

计划提供以下控制项：

- 用户文本。
- 最大标签数量。
- Temperature。
- 最大生成 Token 数。
- 本地推理可选随机种子。

## 内部模块

- `__init__.py`：ComfyUI 节点映射和显示名称。
- `nodes/llm_loaders.py`：GGUF 和 OpenAI 加载器节点。
- `nodes/prompt_generator.py`：Prompt 生成节点和输出拼装。
- `services/llm_provider.py`：统一 `ANIMA_LLM` 接口及两种后端封装。
- `services/tag_index.py`：CSV 校验、分类树、查找索引、排序和缓存。
- `services/prompt_pipeline.py`：分阶段 LLM 请求、解析、校验和回退逻辑。
- `tests/`：使用小型 CSV 和模拟 LLM 的测试。

## Prompt 处理流程

1. 根据 `data/tags.csv` 的修改时间加载或复用标签索引缓存。
2. 让 LLM 结构化分析用户输入中的主体、数量、外观、服装、姿势、动作、
   表情、场景、构图和风格。
3. 根据分析结果选择相关分类分支，并生成规范化的英文搜索词。
4. 从分类树和独立的无分类标签索引中召回数量受限的候选标签。
5. 按精确匹配、分类相关性和 `post_count` 对候选标签排序。
6. 让 LLM 只能从候选集合中选择标签，并返回结构化结果。
7. 校验每个标签是否存在，执行去重、数量限制并拒绝模型臆造标签。
8. 让 LLM 根据原始需求和有效标签生成自然语言描述。
9. 按 `<tag_group>,<description>` 拼装结果，保留描述内部的逗号。

## 错误处理

- CSV 缺少字段或字段格式错误时报告具体行号。
- 拒绝无效的 `category` 和 `post_count`。
- 在调用 LLM 前处理空用户输入。
- 结构化响应解析失败时仅进行有限次数重试。
- GGUF 依赖缺失、模型文件无效、API Key 缺失、OpenAI 超时或重试耗尽时，
  返回清晰的 ComfyUI 节点错误。
- 没有有效候选标签时只生成自然语言，不臆造标签。
- 不静默接受 `data/tags.csv` 中不存在的标签。

## 实施清单

- [ ] 添加最小 ComfyUI 包结构和节点注册。
- [ ] 实现 CSV 解析、校验、分类树、索引和缓存。
- [ ] 定义统一的 `ANIMA_LLM` 接口。
- [ ] 实现 GGUF/llama.cpp 加载器和可选依赖检查。
- [ ] 实现使用环境变量密钥的 OpenAI 加载器。
- [ ] 实现结构化用户意图分析。
- [ ] 实现分类感知的候选检索和排序。
- [ ] 实现受约束的 LLM 标签选择和结果校验。
- [ ] 实现自然语言生成和最终 Prompt 拼装。
- [ ] 使用模拟 Provider 和小型 CSV 添加单元测试。
- [ ] 使用完整 `data/tags.csv` 验证数量、加载时间和内存占用。
- [ ] 添加 `README.md`，说明安装、依赖、节点输入输出和工作流示例。

## 验收标准

- ComfyUI 能发现三个节点且无导入错误。
- 两种加载器的输出都能连接到同一个 Prompt Generator 输入。
- 完整数据加载得到 163,253 个唯一标签，最大分类深度为七。
- 重复执行时不重新解析未变化的 CSV，也不重复加载配置相同的本地模型。
- 自动化测试不依赖真实 GGUF 模型或在线 OpenAI 请求。
- 所有选中标签都存在于 `data/tags.csv`，且没有重复标签。
- API Key 不出现在序列化工作流或日志中。
- 最终输出符合标签加自然语言格式，同时允许描述正常使用标点和逗号。
