# Anima Prompt

Anima Prompt 是一个 ComfyUI 自定义节点包。程序先从开启分类建立随机候选池，
LLM 根据完整绘图需求选择标签，最后再生成自然语言描述并输出：

```text
1girl,solo,A girl is standing in a classroom, looking toward the window.
```

## 节点

- **Anima Local LLM Loader**：从 `ComfyUI/models/LLM` 递归查找并加载 GGUF。
- **Anima OpenAI LLM Loader**：配置 OpenAI 模型与 API Key。
- **Anima Prompt Generator**：输出 `prompt`、`tag_group` 和 `description`。

两种加载器都输出 `ANIMA_LLM`，可直接连接到同一个 Generator 输入。

## 标签数据

当前包含 21,580 个带分类路径的唯一标签，分类路径最多七级。

标签由 Danbooru 与 e621 数据合并，因此 General 和 Character 各有两套 CSV
类别编码：

| 主类别 | CSV `category` | 标签数 |
| --- | --- | ---: |
| Character | `4`、`11` | 10,484 |
| General | `0`、`7` | 10,953 |
| Species | `12` | 143 |


加载标签库时会先执行兼容性校验：文件必须是有效 UTF-8/CSV，列集合必须精确为
`tag`、`category`、`post_count` 和 `classification_1` 至
`classification_7`，且至少包含一条数据。解析器还会逐行拒绝字段缺失、额外
字段、空或重复标签、非法类别、非法文章计数、NUL 字节和不连续的分类层级；
错误会包含具体行号，未通过校验的文件不会进入缓存或发送给 LLM。

### 候选范围

其中 20,016 个标签映射到 Generator 上的开关并可进入候选池：

| 开关 | 标签数 | 默认值 |
| --- | ---: | --- |
| General: Actions | 306 | 开启 |
| General: Attire and body accessories | 1,835 | 开启 |
| General: BDSM and torture | 68 | 开启 |
| General: Body | 1,534 | 开启 |
| General: Composition | 42 | 开启 |
| General: Image composition and style | 891 | 开启 |
| General: Creatures | 518 | 开启 |
| General: Food | 783 | 开启 |
| General: Games | 199 | 开启 |
| General: Lighting | 47 | 开启 |
| General: Misc objects | 888 | 开启 |
| General: Perspective/Depth | 6 | 开启 |
| General: Plants | 155 | 开启 |
| General: Real world | 685 | 开启 |
| General: Sex acts | 344 | 开启 |
| General: Sex objects | 92 | 开启 |
| General: Sexual positions | 50 | 开启 |
| General: Vehicles | 336 | 开启 |
| General: View Angle | 11 | 开启 |
| General: Weapons | 599 | 开启 |
| Character | 10,484 | 关闭 |
| Species | 143 | 关闭 |


## 安装

将本目录放在 `ComfyUI/custom_nodes/Anima-Prompt`，并在 ComfyUI 使用的 Python
环境安装所需后端：

```bash
# OpenAI 后端
python -m pip install -r requirements.txt

# 本地后端：请安装适合当前 CPU/CUDA/ROCm 环境的 llama-cpp-python
python -m pip install llama-cpp-python

# ROCm 7.2 + RDNA3/gfx1100 示例
CMAKE_ARGS="-DGGML_HIP=ON -DGPU_TARGETS=gfx1100" \
python -m pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
```

GPU 环境应按 `llama-cpp-python` 上游说明安装对应构建，不建议用通用 CPU wheel
覆盖已有的 CUDA/ROCm 构建。将 `.gguf` 文件放入 `ComfyUI/models/LLM` 后重启
ComfyUI。可用 `llama_cpp.llama_cpp.llama_print_system_info()` 检查输出是否包含
HIP/ROCm；同时存在独显和核显时，可在启动 ComfyUI 前设置 `HIP_VISIBLE_DEVICES`
只暴露用于推理的设备。

推荐在 OpenAI 后端启动前通过环境变量设置密钥：

```bash
export OPENAI_API_KEY="..."
```

也可以在节点的 `OPENAI_API_KEY` 字段直接填写；直接值优先于环境变量。该值会
保存在工作流 JSON 中，不应共享包含密钥的工作流。Loader 创建时不会发出网络
请求。

## 输入输出

Local Loader 提供上下文长度、GPU 层数、CPU 线程数和 batch 大小。相同模型与
配置会复用已加载实例；`gpu_layers=-1` 表示尽可能加载到 GPU。

OpenAI Loader 提供模型名、可直接填写的 `OPENAI_API_KEY`、密钥环境变量名、
超时和最大重试次数。直接字段留空时读取指定的环境变量。

Generator 接收用户文本、标签数量范围、自然语言句子数量范围、temperature、
最大生成 token 数和可选 seed（`-1` 表示不指定）。标签默认选择 8–24 个，
最大不能超过 50。自然语言默认生成 1–3 句，范围必须在 1–10 句内。程序先为每个
已开启且有数据的分类保留均衡随机候选，再加入用户输入中显式词项召回的标签，组成
最多 100 个候选，并在标签数量范围内随机确定最终数量。中文等无词项命中的输入仍
使用均衡随机候选。标签阶段使用零 temperature 和结构化输出约束；自然语言阶段使用
Generator 的 temperature。两个阶段会根据目标标签数和最大句数收紧输出 token 上限。
LLM 同时接收完整用户输入和按分类分组的候选池。程序会删除 LLM 返回的重复和池外
标签，并在传给描述 LLM 前随机补齐缺失分类；超过目标数量时只随机丢弃分类内的额外
标签，不会移除该分类的最后一个标签。
`min_tags` 和 `max_tags` 限制最终输出的标签总数，而不是单个分类的数量。指定 seed
时候选池、目标数量和修正结果均可复现，`-1` 时每次重新随机。`max_tags` 小于有
候选的已开启分类数，或所有分类无法提供 `min_tags` 个候选时会报错；LLM 返回的
结构无法解析时会进行有限次数重试。
自然语言阶段同时接收原始用户文本和最终校验后的标签。LLM 必须保留用户要求、
融合全部标签，并可补充不冲突的描述细节，使两部分形成连贯的最终描述。
标签组使用英文逗号连接，描述内部的逗号会保留。
输出标签中的下划线会替换为空格，圆括号会转义为 `\(` 和 `\)`。
管线不会按内容类型审查或省略标签；在线 API 服务自身的策略仍可能生效。

最小工作流连接如下：

```text
Anima Local LLM Loader ─┐
                        ├─ ANIMA_LLM → Anima Prompt Generator → prompt
Anima OpenAI LLM Loader ┘
```

两个 Loader 任选其一。将 Generator 的 `prompt` 输出连接到后续接收 Anima
提示词的文本输入即可；`tag_group` 与 `description` 可用于单独预览或调试。

## 测试

测试使用小型临时 CSV 和模拟 LLM，不需要真实 GGUF 或在线 API：

```bash
python -m pytest -q tests --import-mode=importlib
```
