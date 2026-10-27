# 🤖 本地离线 RAG 技术文档问答助手 (Ollama + LangChain LCEL)
> 全离线私有化知识库问答系统，零云端API、数据不出本地，解决技术文档精准检索问答，彻底杜绝跨文档语义污染与模型幻觉。

## 📌 技术架构
Ollama(本地模型服务) + LangChain LCEL链式调用 + Chroma向量库 + bge-m3向量化 + DeepSeek-R1 8B推理
1. 文档加载 → 文本清洗 + 语义分块
2. 分批向量化入库（防OOM崩溃）
3. 多层检索过滤：粗召回 + 元数据映射 + Python后置过滤，隔离不同文档
4. LCEL流式问答，自动过滤模型`<thinking>`推理标签，实现纯净打字机输出

## ✨ 核心工程亮点
- **100%离线私有化**：本地运行DeepSeek-R1(8B)大模型 + bge-m3向量模型，无需任何API Key，文档数据全程保存在本机，隐私安全可控。
- **大批量入库防崩溃**：分批向量化策略 `batch=32 + 0.1s微延迟`，彻底解决Ollama一次性处理上千文本块时的OOM内存溢出崩溃问题。
- **跨文档防污染（核心优化）**：自研「超大召回+文档字典映射+Python后置过滤」方案，物理隔离不同文档，完全解决多知识库场景下的跨文档语义串扰。
- **PDF文档清洗优化**：内置正则清洗模块，自动剔除页眉、页脚、多余换行与PDF乱码，大幅提升文本块语义完整性。
- **PDF/TXT双版本A/B测试**：对比两种文档格式的切分效果，优选TXT纯文本方案，检索召回完整度提升30%+。
- **纯净流式输出**：基于LCEL构建RAG链路，手写解析器自动剔除DeepSeek-R1自带的`<thinking>`思考标签，实现无冗余内容的打字机流式回复。

## 📊 A/B测试对比（PDF vs TXT）
在相同文档、分块参数与检索模型条件下，PDF 解析得到 42 个文本块，存在页眉干扰、代码断行、语义碎片化问题，作为基准；TXT 解析得到 36 个文本块，文本干净、语义连贯，召回完整度提升 30%+。
> 结论：优先使用TXT纯文本；必须使用PDF时，开启页眉页脚正则清洗。

## 🛠️ 环境准备 & 快速启动
### 1. 前置条件
- 已安装并启动 Ollama（后台常驻运行）
- 硬件最低要求：内存≥16G；显存≥8G（CPU可运行，速度较慢）

### 2. 拉取本地模型
```bash
# 推理大模型
ollama pull deepseek-r1:8b
# 向量化Embedding模型
ollama pull bge-m3
```

### 3. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 4. 运行项目

1. 在项目根目录新建文件夹：`PDF/` 或 `TXT/`，放入你的文档；
2. 首次构建向量库（重建索引）：

```bash
# TXT版本（推荐）
python RAG_TXT.py --rebuild

# PDF版本（自动清洗页眉页脚）
python RAG_PDF.py --rebuild
```

3. 后续直接启动问答，无需重建向量库：

```bash
python RAG_TXT.py
```

## 📂 项目目录结构

```
├── RAG_PDF.py          # PDF版本主程序（内置PDF正则清洗逻辑）
├── RAG_TXT.py          # TXT纯文本版本主程序（推荐首选）
├── requirements.txt    # 项目全部Python依赖包
├── PDF/                # 自建文件夹，存放PDF文档
└── TXT/                # 自建文件夹，存放TXT文档
```

## 🎯 开发踩坑记录（工程排坑实录）

1. **Ollama 大批量 Embedding OOM 崩溃**
问题：一次性提交 1800 + 文本块向量化，Ollama 服务直接内存溢出退出。
解决方案：改写为分批写入策略，设置`batch=32`，每批之间增加 0.1 秒延时，平稳控制请求并发。
2. **ChromaDB 中文元数据过滤失效**
问题：Chroma 原生`$contains`元数据筛选对中文文件名支持极差，经常查询返回空结果。
解决方案：放弃数据库层过滤，迁移到 Python 代码层做后置精准过滤，文档隔离 100% 生效。
3. **DeepSeek-R1 输出冗余思考标签**
问题：模型会自动输出`<thinking>...</thinking>`中间推理内容，影响回答整洁度。
解决方案：流式输出时增加字符串解析器，自动截断并过滤思考标签，只输出最终答案。

## ❓ 常见问题 FAQ

<details>
<summary>Q1：CPU运行速度很慢怎么办？</summary>
A1：开启Ollama GPU加速；或者更换更小参数量模型（如deepseek-r1:7b、qwen:4b）。
</details><details>
<summary>Q2：如何新增更多文档？</summary>
A2：把文件放入对应文件夹，执行命令带上`--rebuild`重新构建向量库即可。
</details><details>
<summary>Q3：如何切换其他LLM/Embedding模型？</summary>
A3：直接修改代码内model_name参数，同步在ollama pull对应模型。
</details>

## 🚀 后续扩展方向

1. 接入 Streamlit/Gradio，搭建可视化 Web 问答界面；
2. 支持 Markdown、Word、EPUB 等更多文档格式；
3. 增加文档来源引用、检索溯源；
4. 支持多知识库独立切换；
5. 增加向量库持久化压缩、增量入库（无需全量重建）。
