# -*- coding: utf-8 -*-
"""
本地离线 RAG 技术文档问答助手 ( PDF 版)
- 目录: ./PDF
- 优化: PDF文本清洗、Prompt瘦身、字典路由映射、兜底降级
"""

import sys
import json
import shutil
import argparse
import urllib.request
import time
import re
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Windows 控制台 UTF-8 
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ================= 配置区 =================
BASE_URL = "http://localhost:11434"
LLM_MODEL = "deepseek-r1:8b"
EMBED_MODEL = "bge-m3"

DOCS_DIR = Path("./PDF")  # 存放 PDF 的文件夹
PERSIST_DIR = Path("./chroma_db")
META_FILE = PERSIST_DIR / "meta.json"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 8
NUM_CTX = 8192
MIN_CHARS_PER_PAGE = 50
HISTORY_TURNS = 3
# ==========================================

SEP = "─" * 60


# ---------- 文本清洗器 ----------
def clean_pdf_text(text):
    """清洗 PDF 抽出的文本，去除页眉页脚和多余空行"""
    # 去除多余的换行（把多个换行替换为一个）
    text = re.sub(r'\n+', '\n', text)
    # 去除常见的页眉页脚（如“第 1 页”）
    text = re.sub(r'第\s*\d+\s*页', '', text)
    # 去除连续的空格
    text = re.sub(r' +', ' ', text)
    return text.strip()


# ---------- 环境预检 ----------
def check_ollama() -> bool:
    try:
        urllib.request.urlopen(f"{BASE_URL}/api/tags", timeout=3)
        return True
    except Exception:
        return False


# ---------- 加载与切分 ----------
def load_and_split():
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"❌ {DOCS_DIR} 下没有 PDF 文件")

    all_docs = []
    for pdf in pdfs:
        print(f"📄 加载 {pdf.name} ...", end=" ")
        pages = PyMuPDFLoader(str(pdf)).load()
        for page in pages:
            # 核心优化：抽取出来的文本立刻进清洗器
            page.page_content = clean_pdf_text(page.page_content)

        total = sum(len(p.page_content) for p in pages)
        print(f"{len(pages)} 页, {total} 字符")
        all_docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", " ", ""],
    )
    splits = splitter.split_documents(all_docs)
    print(f"✂️ 切分完成：{len(splits)} 个文本块")
    return splits


# ---------- 向量库管理 ----------
def build_or_load_vectorstore(rebuild: bool = False):
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)

    if rebuild and PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)
        print("🗑️ 已删除旧向量库")

    if PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir()):
        if META_FILE.exists():
            meta = json.loads(META_FILE.read_text(encoding="utf-8"))
            if meta.get("embed_model") != EMBED_MODEL or meta.get("chunk_size") != CHUNK_SIZE:
                sys.exit("❌ 向量库参数不一致，请运行：python RAG_PDF.py --rebuild")
        print("📦 加载已有向量库（秒启动）")
        return Chroma(persist_directory=str(PERSIST_DIR), embedding_function=embeddings)

    splits = load_and_split()
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    META_FILE.write_text(json.dumps({"embed_model": EMBED_MODEL, "chunk_size": CHUNK_SIZE}), encoding="utf-8")

    print("🧠 正在向量化（分批写入防崩溃）...")
    vs = Chroma(persist_directory=str(PERSIST_DIR), embedding_function=embeddings)
    batch_size = 32
    total = len(splits)
    for i in range(0, total, batch_size):
        vs.add_documents(splits[i:i + batch_size])
        print(f"   已处理 {min(i + batch_size, total)} / {total} 个文本块")
        time.sleep(0.1)

    print("✅ 向量库构建完成")
    return vs


# ---------- 辅助 ----------
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def format_history(history):
    if not history: return "（无）"
    return "\n".join([f"{'用户' if isinstance(h, HumanMessage) else '助手'}：{h.content}" for h in history])


def print_sources(docs):
    print("\n📚 引用来源：")
    for d in docs:
        src = Path(d.metadata.get("source", "未知")).name
        page = d.metadata.get("page", 0) + 1
        print(f"  - {src} 第 {page} 页")


# ---------- LCEL 链 ----------
def build_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是严谨的技术文档助手。请严格仅基于【参考资料】回答问题。\n"
         "要求：\n"
         "1. 若参考资料不足以回答问题，必须直接回复：【在当前资料中没有找到相关信息】，绝不编造。\n"
         "2. 若问题要求列举，请将参考资料中提及的所有相关项完整列出。\n"
         "3. 回答直接、简洁。"),
        ("human", "【对话历史】\n{history}\n\n【参考资料】\n{context}\n\n【用户问题】\n{question}")
    ])
    return prompt | llm | StrOutputParser()


def stream_answer(chain, question, history, context):
    full, buf = "", ""
    thinking = True
    for chunk in chain.stream({"question": question, "context": context, "history": format_history(history)}):
        buf += chunk
        if thinking:
            if "" in buf:
                idx = buf.index("") + len("")
                tail = buf[idx:]
                if tail:
                    print(tail, end="", flush=True)
                    full += tail
                buf = ""
                thinking = False
            elif len(buf) > 2000:
                print(buf, end="", flush=True)
                full += buf
                buf = ""
                thinking = False
        else:
            print(buf, end="", flush=True)
            full += buf
            buf = ""
    if buf: print(buf, end="", flush=True)
    print()
    return full


# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="删除旧向量库并重建")
    args = parser.parse_args()

    print(SEP)
    print("🤖 本地离线 RAG 技术文档问答助手 (优化版)")
    print(f"   LLM: {LLM_MODEL}   |   Embedding: {EMBED_MODEL}")
    print(f"   目录: {DOCS_DIR}     |   top_k={TOP_K}")
    print(SEP)

    if not check_ollama():
        sys.exit(f"❌ 无法连接 Ollama（{BASE_URL}），请确认已启动。")

    vectorstore = build_or_load_vectorstore(rebuild=args.rebuild)
    global_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    llm = ChatOllama(model=LLM_MODEL, base_url=BASE_URL, temperature=0, num_ctx=NUM_CTX)
    chain = build_chain(llm)

    history = []
    last_docs = []

    keyword_mapping = {
        "Agent框架": ["8种主流Agent框架"],
        "Hugging Face": ["8种主流Agent框架"],
        "轻量级框架": ["8种主流Agent框架"],
        "机器学习": ["6个机器学习的核心算法"],
        "算法": ["6个机器学习的核心算法"],
        "Agent模式": ["5种Agent模式"],
        "模式": ["5种Agent模式"],
        "Prompt": ["14种常见的Prompt策略"]
    }

    print("\n✅ 就绪 | 命令：exit 退出 | /clear 清历史 | /sources 看来源 | /help")
    print(SEP)

    while True:
        try:
            q = input("\n👤 问题：").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not q: continue
        if q.lower() in ("exit", "quit"): break
        if q == "/clear":
            history.clear()
            print("🧹 已清空对话历史")
            continue
        if q == "/sources":
            if last_docs: print_sources(last_docs)
            continue

        # 1. 智能路由匹配
        target_files = []
        hit_kws = []
        for kw, files in keyword_mapping.items():
            if kw in q:
                hit_kws.append(kw)
                target_files.extend(files)

        if hit_kws:
            print(f"\n🎯 触发路由，关键词: {hit_kws} -> 锁定文件: {set(target_files)}")
            # 拉取 100 个候选
            raw_docs = vectorstore.as_retriever(search_kwargs={"k": 100}).invoke(q)
            # Python 过滤
            docs = [
                d for d in raw_docs
                if any(tf in Path(d.metadata.get("source", "")).name for tf in target_files)
            ][:12]

            # 兜底：如果过滤后为空，强行降级全局检索
            if not docs:
                print("  ⚠️ 定向过滤为空，自动降级为全局检索")
                docs = raw_docs[:TOP_K]
        else:
            docs = global_retriever.invoke(q)

        last_docs = docs
        print(f"🔍 最终保留 {len(docs)} 个相关片段")

        # 2. 生成
        context = format_docs(docs)
        print("💡 回答：", end="", flush=True)
        try:
            answer = stream_answer(chain, q, history, context)
        except Exception as e:
            print(f"\n⚠️ 调用失败：{e}")
            continue

        # 3. 记录
        history.append(HumanMessage(content=q))
        history.append(AIMessage(content=answer))
        if len(history) > HISTORY_TURNS * 2:
            history = history[-HISTORY_TURNS * 2:]

        print_sources(docs)


if __name__ == "__main__":
    main()