# 🦉 Audit-QA: SOTA Domain-Specific RAG Assistant

**Audit-QA** is a state-of-the-art Question-Answering system designed to empower Small and Medium Businesses (SMBs) with instant, accurate, and grounded information from their own knowledge bases.

By combining the low-latency power of **Groq** with a robust **Retrieval-Augmented Generation (RAG)** pipeline, this assistant provides a premium customer support experience that stays strictly within the boundaries of the provided business data.

## 🚀 Key Features

- **Inference Engine**: Llama-3-70B via Groq LPU (Ultra-fast).
- **Knowledge Base**: Multi-format support (PDF, JSON, Text).
- **Memory**: Dual-tier architecture (Short-term Redis & Long-term PostgreSQL).
- **Transparency**: Built-in Audit Logger for interaction tracking and analytics.

## 🛠️ Tech Stack

- **Framework**: Python, LangChain, Streamlit
- **LLM**: Groq (Llama-3-70B)
- **Vector DB**: ChromaDB
- **Cloud State**: Upstash (Redis), Supabase (PostgreSQL)

---

### 🚧 Work In Progress

This project is currently under active development. Modular components (Inference, Retrieval, Logger) are being built and integrated step-by-step to ensure professional code quality and robust system architecture.
