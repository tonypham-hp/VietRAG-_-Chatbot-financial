# scripts/build_index.py 

import os, json, pickle
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi  

DATA_DIR = "processed"
MODEL_EMBED = "all-MiniLM-L6-v2"  
BM25_PATH = "models/bm25.pkl"
FAISS_PATH = "models/faiss.index"
META_PATH = "models/meta_detailed.json"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
BATCH_SIZE = 3000

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = str(text).replace("\r", " ").replace("\n", " ").strip()
    if len(text) < 100:
        return [text] if len(text) > 10 else []
    chunks = [text[i:i+size].strip() for i in range(0, len(text), size - overlap)]
    return [c for c in chunks if len(c) > 50]

def load_docs(data_dir):
    docs, metas = [], []
    print(f"Đang quét dữ liệu trong: {data_dir}\n")

    for root, _, files in os.walk(data_dir):
        for fname in files:
            path = os.path.join(root, fname)
            ext = fname.split(".")[-1].lower()
            try:
                if ext == "csv":
                    # 🔹 FIX: convert NaN → chuỗi rỗng
                    df = pd.read_csv(path).fillna("")
                    if {"Bộ luật", "Điều", "Khoản", "Nội dung"}.issubset(df.columns):
                        for _, row in df.iterrows():
                            content = str(row["Nội dung"]).strip()
                            if not content: continue
                            for ch in chunk_text(content):
                                docs.append(ch)
                                metas.append({
                                    "file": fname,
                                    "law": str(row.get("Bộ luật", "")).strip(),
                                    "dieu": str(row.get("Điều", "")).strip(),
                                    "khoan": str(row.get("Khoản", "")).strip(),
                                    "text": ch
                                })
                    else:
                        text = " ".join(df.astype(str).agg(" ".join, axis=1))
                        for ch in chunk_text(text):
                            docs.append(ch)
                            metas.append({"file": fname, "text": ch})

                elif ext == "parquet":
                    df = pd.read_parquet(path)
                    text = " ".join(df.astype(str).agg(" ".join, axis=1))
                    for ch in chunk_text(text):
                        docs.append(ch)
                        metas.append({"file": fname, "text": ch})

                elif ext == "txt":
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                    for ch in chunk_text(text):
                        docs.append(ch)
                        metas.append({"file": fname, "text": ch})

                print(f"Đọc: {fname}")
            except Exception as e:
                print(f"Lỗi đọc {fname}: {e}")

    print(f"\n📊 Tổng số đoạn đọc được: {len(docs)}")
    return docs, metas

def build_bm25(docs):
    print("\n🔍 Đang tạo chỉ mục BM25...")
    tokenized = [d.split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    os.makedirs(os.path.dirname(BM25_PATH), exist_ok=True)
    with open(BM25_PATH, "wb") as f: pickle.dump(bm25, f)
    print(f" Lưu BM25 tại: {BM25_PATH}")

def build_faiss(docs):
    print("\n🔢 Đang tạo FAISS embeddings (chia batch)...")
    model = SentenceTransformer(MODEL_EMBED)
    os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)
    dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dim)
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i+BATCH_SIZE]
        embeddings = model.encode(batch, show_progress_bar=True)
        index.add(embeddings)
    faiss.write_index(index, FAISS_PATH)
    print(f"Lưu FAISS tại: {FAISS_PATH}")

def save_meta(metas):
    clean_meta = []
    for m in metas:
        clean_meta.append({k: ("" if pd.isna(v) else v) for k, v in m.items()})
    meta = {
        "model_name": "Qwen/Qwen2-0.5B-Instruct",
        "num_chunks": len(clean_meta),
        "records": clean_meta
    }
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f" Lưu metadata tại: {META_PATH}")

if __name__ == "__main__":
    docs, metas = load_docs(DATA_DIR)
    if not docs:  
        print("Không tìm thấy dữ liệu hợp lệ.")
        exit(0)
    build_bm25(docs)
    build_faiss(docs)  
    save_meta(metas)
    print("\n Hoàn tất build_index.py ")


 




