#  VietRAG - Chatbot Financial  
### Retrieval-Augmented Legal & Financial Assistant for Vietnam

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FAISS](https://img.shields.io/badge/FAISS-RAG-orange)
![LLM](https://img.shields.io/badge/Qwen-0.5B--Instruct-green)
![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)

---

**VietRAG** is a **Vietnamese Legal & Financial Chatbot** built on  
**Retrieval-Augmented Generation (RAG)** architecture combining **BM25**, **FAISS**,       
and **Qwen-0.5B-Instruct** LLM to deliver reliable and contextual answers.

The chatbot can
- Retrieve and explain laws, decrees, and financial regulations  
- Query market data such as CPI, USD/VND exchange rate, volume, return, RSI, ATR...  
- Define key economic and financial concepts  
- Provide basic advisory and guidance for investors
  

---

## System Architecture

| Layer | Component | Description |
|-------|------------|-------------|
| **1️⃣ Data Layer** | `processed/` | Contains laws, decrees, CPI, FX, OHLCV data, and QA corpus |
| **2️⃣ Retrieval Layer** | BM25 + FAISS | Hybrid retrieval combining keyword and semantic search |
| **3️⃣ Reasoning Layer** | Qwen-0.5B-Instruct | Language model reasoning and response generation |
| **4️⃣ Response Layer** | Rule + LLM | Formats and cites legal sources in natural Vietnamese |

---

## Project Structure

```bash
VietRAG-Chatbot-Financial/
│
├── chatbot.py              # Main chatbot (RAG + query processing)
├── build_index.py          # Build BM25 + FAISS indexes
├── utils_legal.py          # Legal lookup and formatting helpers
├── intent_detector.py      # User intent classification
├── requirements.txt        # Python dependencies
├── models/                 # Prebuilt FAISS, BM25, and metadata files (private)
└── processed/              # Preprocessed legal and financial datasets (private)
