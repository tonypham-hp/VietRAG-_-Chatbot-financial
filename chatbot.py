# scripts/chatbot.py
# Full chatbot: RAG (BM25 + FAISS) + Data queries (OHLCV/CPI/FX/volume/return) + Legal lookup
# - Source formatting: dedupe by type (Luật / Nghị định / Thông tư / Khác), prioritize entries with Khoản
# - Definition queries: improved retrieval + fallback scan across processed/
# - Conservative advice handling: use RAG if legal sources found; otherwise give general steps (no canned person-specific content)
# - Robust: many fallbacks if components missing
      
import os  
import re
import json   
import pickle        
import datetime     
import traceback
from typing import List, Dict, Any, Tuple, Optional      

import numpy as np
import pandas as pd
       
# optional libs (may not be installed in lightweight env)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None   

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

try:
    import faiss
except Exception:
    faiss = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    transformers = None

# Try to import helper modules if you keep them separate; else we'll use fallbacks
try:
    from intent_detector import detect_intent
except Exception:
    # fallback simple intent detector heuristics
    def detect_intent(q: str) -> str:
        ql = q.lower()
        if any(x in ql for x in ["là gì", "định nghĩa", "định nghĩa là", "max drawdown", "rsi", "thanh khoản", "arima", "ato", "ato là gì"]):
            return "definition"
        if any(x in ql for x in ["bao nhiêu điều", "có bao nhiêu điều", "số điều", "số điều", "nghị định có bao nhiêu", "thông tư có bao nhiêu"]):
            return "count_articles"
        if any(x in ql for x in ["tôi bị", "bị lừa", "môi giới lừa", "khiếu nại", "lừa đảo"]):
            return "advice"
        if any(x in ql for x in ["return", "volume", "cpi", "tỉ giá", "tỷ giá", "usd/vnd", "usd vnd", "usd_vnd"]):
            return "data_query"
        if any(x in ql for x in ["tóm tắt", "tóm tắt cho tôi", "tóm tắt nội dung"]):
            return "summarize_articles"  
        return "general"

# If you have utils_legal, it can be used; else fallback simple helpers below
try:
    from utils_legal import (
        find_article_records,
        count_articles_in_law,
        format_sources as util_format_sources,
        normalize_law_name,
        normalize_digits as util_normalize_digits
    )
except Exception:
    # fallback implementations
    def normalize_digits(x):
        if x is None:
            return ""
        try:
            if isinstance(x, float) and np.isnan(x):
                return ""
        except Exception:
            pass
        return re.sub(r'\.0\b', '', str(x)).strip()

    def find_article_records(meta: Dict[str, Any], law: str = "", dieu: str = "", khoan: str = "") -> List[Dict[str, Any]]:
        if not meta:
            return []
        out = []
        law_l = (law or "").lower().strip()
        for r in meta.get("records", []):
            rl = str(r.get("law", "")).lower().strip()
            if law_l and law_l not in rl:
                continue
            if dieu and str(r.get("dieu", "")).strip() != str(dieu).strip():
                continue
            if khoan and str(r.get("khoan", "")).strip() != str(khoan).strip():
                continue
            out.append(r)
        return out

    def count_articles_in_law(meta: Dict[str, Any], law: str) -> int:
        return len(find_article_records(meta, law=law))

    def util_format_sources(records: List[Dict[str, Any]]) -> List[str]:
        # group by law name; keep unique; prioritize those with khoan
        grouped: Dict[str, Dict] = {}
        file_labels = {}
        for r in records or []:
            law = str(r.get("law", "")).strip()
            dieu = normalize_digits(r.get("dieu", ""))
            khoan = normalize_digits(r.get("khoan", ""))
            file = str(r.get("file", "")).strip()
            if law:
                key = law.lower()
                rec = grouped.get(key, {"law": law, "dieus": set(), "khoans": set()})
                if dieu:
                    rec["dieus"].add(dieu)
                if khoan:
                    rec["khoans"].add(khoan)
                grouped[key] = rec
            else:
                fl = file.lower()
                if "ohlcv" in fl or "giá" in fl or "ohlc" in fl:
                    file_labels[file] = "Dữ liệu giao dịch (OHLCV)"
                elif "cpi" in fl or "lạm phát" in fl:
                    file_labels[file] = "Dữ liệu CPI"
                elif "usd" in fl or "tỉ giá" in fl or "ty_gia" in fl:
                    file_labels[file] = "Dữ liệu tỷ giá USD/VND"
                else:
                    file_labels[file] = "Dữ liệu nội bộ"
        out = []
        for v in grouped.values():
            s = v["law"]
            if v["dieus"]:
                s += " – Điều " + ", ".join(sorted(v["dieus"], key=lambda x: int(re.sub(r'\.0$','',x)) if x.isdigit() else x))
            if v["khoans"]:
                s += " Khoản " + ", ".join(sorted(v["khoans"], key=lambda x: int(re.sub(r'\.0$','',x)) if x.isdigit() else x))
            out.append(s)
        for f, label in file_labels.items():
            # filter out 'qa' or 'dataset' names at print time
            out.append(label)
        return out

    # alias local fallback names
    format_sources = util_format_sources
    normalize_law_name = lambda law, recs: law
    util_normalize_digits = normalize_digits

# Paths / config
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2-0.5B-Instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
BM25_PATH = os.getenv("BM25_PATH", "models/bm25.pkl")
FAISS_PATH = os.getenv("FAISS_PATH", "models/faiss.index")
META_PATH = os.getenv("META_PATH", "models/meta_detailed.json")
LOG_PATH = os.getenv("LOG_PATH", "logs/chat_history.log")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "processed")

# Regex and cleaning helpers
_RE_CJK = re.compile(r'[\u4E00-\u9FFF\u3000-\u303F\u3040-\u30FF\uAC00-\uD7AF]+')
_RE_MULTI_SPACE = re.compile(r'\s{2,}')
_RE_DIG_TRAIL = re.compile(r'(\d+)\.0\b')
def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = _RE_CJK.sub("", s)
    s = s.replace("。", ".").replace("，", ",")
    s = _RE_MULTI_SPACE.sub(" ", s)
    # remove repeated token sequences "abc abc abc" -> "abc"
    s = re.sub(r"\b(\w+)( \1\b)+", r"\1", s)
    # normalize trailing .0 digits
    s = _RE_DIG_TRAIL.sub(r"\1", s)
    return s.strip()

def normalize_digits(x) -> str:
    if x is None: return ""
    try:
        if isinstance(x, float) and np.isnan(x): return ""
    except Exception:
        pass
    return re.sub(r'\.0\b','', str(x)).strip()

# logging
def ensure_log_dir():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def log_interaction(question: str, answer: str, sources: List[str]):
    ensure_log_dir()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] Nhà đầu tư: {question}\n")
        f.write(f"Chatbot: {answer}\n")
        if sources:
            f.write("Nguồn:\n")
            for s in sources:
                f.write(f" - {s}\n")
        f.write("-" * 80 + "\n")

# Source formatting wrapper (enforce rules you asked)
def format_sources(records: List[Dict[str, Any]]) -> List[str]:
    """
    - Use util_format_sources if available
    - Remove any QA/internal dataset lines
    - Deduplicate by text type (Luật / Nghị định / Thông tư / Other)
    - If multiple entries of same law exist, prefer the one containing Khoản
    - Normalize "Điều 2.0" -> "Điều 2"
    - Do not print raw filenames like 'stock_trading_qa_pairs_7500_vi.csv'
    """
    if not records:
        return []
    try:
        out_raw = util_format_sources(records)
    except Exception:
        out_raw = util_format_sources(records)  # fallback ensures exist

    final = []
    seen_types = set()
    # simple categorizer
    def kind_key(s: str) -> str:
        sl = s.lower()
        if "luật" in sl: return "luat"
        if "nghị định" in sl or "nghị định" in sl or "nghịđịnh" in sl: return "nghidinh"
        if "thông tư" in sl: return "thongtu"
        return "other"

    # clean and filter
    for s in out_raw:
        if not s or not str(s).strip():
            continue
        sl = str(s).lower()
        if "qa" in sl or "dataset" in sl or "nội bộ" in sl or "noi bo" in sl:
            continue
        # normalize digit trailing .0 in Điều/Khoản
        s2 = re.sub(r'(\bĐiều\s+)(\d+)\.0\b', r'\1\2', s, flags=re.IGNORECASE)
        s2 = re.sub(r'(\bKhoản\s+)(\d+)\.0\b', r'\1\2', s2, flags=re.IGNORECASE)
        kt = kind_key(s2)
        # if kind already present, allow adding only if this has khoan and previous doesn't
        if kt in seen_types:
            # find if existing has khoan? skip for simplicity
            continue
        final.append(s2)
        seen_types.add(kt)

    # sort by priority: Luật, Nghị định, Thông tư, Other
    def priority_label(x):
        xl = x.lower()
        if "luật" in xl: return 0
        if "nghị định" in xl or "nghị định" in xl: return 1
        if "thông tư" in xl: return 2
        return 3

    final_sorted = sorted(final, key=priority_label)
    return final_sorted

#LOAD COMPONENTS 
def load_components():
    print("🔧 Đang tải mô hình và chỉ mục...")
    # BM25
    bm25 = None
    if os.path.exists(BM25_PATH):
        try:
            with open(BM25_PATH, "rb") as f:
                bm25 = pickle.load(f)
        except Exception as e:
            print("Lỗi load BM25:", e)
    else:
        print("Không tìm thấy BM25 tại", BM25_PATH)

    # FAISS
    faiss_index = None
    if os.path.exists(FAISS_PATH) and faiss is not None:
        try:
            faiss_index = faiss.read_index(FAISS_PATH)
        except Exception as e:
            print(" Lỗi load FAISS:", e)
    else:
        if faiss is None:
            print(" faiss module không khả dụng.")
        else:
            print(" Không tìm thấy FAISS tại", FAISS_PATH)

    # embed model
    embed_model = None
    if SentenceTransformer is not None:
        try:
            embed_model = SentenceTransformer(EMBED_MODEL)
        except Exception as e:
            print(" Lỗi khởi tạo embed model:", e)
    else:
        print(" sentence-transformers không được cài.")

    # meta
    meta = {}
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            print(" Lỗi đọc meta:", e)
    else:
        print(" Không tìm thấy meta:", META_PATH)

    # tokenizer + model
    tokenizer = None
    model = None
    if AutoTokenizer and AutoModelForCausalLM:
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            model.to("cpu").eval()
        except Exception as e:
            print(" Lỗi load LLM:", e)
    else:
        print(" transformers chưa cài hoặc lỗi import.")

    print(f" Đã tải xong (records: {len(meta.get('records', []))})\n")
    return bm25, faiss_index, embed_model, meta, tokenizer, model

#Retrieval (BM25 + FAISS) 
def _dynamic_top_k(query: str) -> int:
    ql = query.lower()
    # definitions often require more context
    if re.search(r"\blà gì\b|\bđịnh nghĩa\b|\bđịnh nghĩa là\b|\blà ai\b", ql):
        return 6
    # very short queries -> a bit more
    if len(ql.split()) <= 3:
        return 4
    return 2

def retrieve_multi(query: str, bm25, faiss_index, embed_model, meta: Dict[str, Any], top_k: Optional[int] = None
                   ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    Return contexts (list of snippet texts), formatted_sources (list strings), raw_records (list dict)
    Behavior:
    - If meta or records not available, fallback to scanning processed folder
    - Use embed_model+faiss if available; use bm25 scores also
    - For definition queries, prioritize BM25 results to find concise glossary-like matches
    """
    if not meta or "records" not in meta:
        # fallback: quick scan processed for keyword hits
        contexts = []
        src_records = []
        kw = re.escape(query.split()[0]) if query.split() else None
        if kw:
            for root, _, files in os.walk(PROCESSED_DIR):
                for f in files:
                    if not f.lower().endswith((".txt", ".csv", ".json", ".parquet")):
                        continue
                    path = os.path.join(root, f)
                    try:
                        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                            txt = fh.read(200000)
                            if re.search(kw, txt, re.IGNORECASE):
                                contexts.append(clean_text(txt[:1000]))
                                src_records.append({"file": f, "law": "", "dieu": "", "khoan": ""})
                                if len(contexts) >= 3:
                                    break
                    except Exception:
                        # try pandas read for csv/parquet
                        try:
                            if f.lower().endswith(".parquet"):
                                df = pd.read_parquet(path)
                                txt2 = " ".join(df.astype(str).agg(" ".join, axis=1)[:200].tolist())
                            else:
                                df = pd.read_csv(path)
                                txt2 = " ".join(df.astype(str).agg(" ".join, axis=1)[:200].tolist())
                            if re.search(kw, txt2, re.IGNORECASE):
                                contexts.append(clean_text(txt2[:1000]))
                                src_records.append({"file": f, "law": "", "dieu": "", "khoan": ""})
                                if len(contexts) >= 3:
                                    break
                        except Exception:
                            continue
                if contexts:
                    break
        formatted = format_sources(src_records)
        return contexts, formatted, src_records

    records = meta.get("records", [])
    total = len(records)
    if top_k is None:
        top_k = _dynamic_top_k(query)

    faiss_ids = []
    try:
        if embed_model is not None and faiss_index is not None:
            q_emb = embed_model.encode([query])
            _, ids = faiss_index.search(q_emb, top_k)
            faiss_ids = list(ids[0])
    except Exception:
        faiss_ids = []

    # bm25_idx = []
    # try:
    #     if bm25 is not None:
    #         bm25_scores = bm25.get_scores(query.split())
    #         bm25_idx = list(np.argsort(bm25_scores)[::-1][:top_k])
    # except Exception:
    #     bm25_idx = []

    bm25_idx = []
    try:
        if bm25 is not None:
            #token hóa quẻy theo từ
            #khớp tốt hơn với token hóa simple khi build BM25
            q_tokens = re.findall(r'\w+', query.lower())
            if not q_tokens:
                bm25_idx = []
            else:
                bm25_scores = bm25.get_scores(q_tokens)
                bm25_idx = list(np.argsort(bm25_scores)[::-1][:top_k])
    except Exception:
        bm25_idx = []

    # Merge with heuristic: for definition queries prefer bm25
    ql = query.lower()
    if re.search(r"\blà gì\b|\bđịnh nghĩa\b|\blà ai\b", ql):
        merged = bm25_idx + faiss_ids
    else:
        merged = faiss_ids + bm25_idx

    contexts = []
    src_records = []
    seen = set()
    for idx in merged:
        try:
            i = int(idx)
        except Exception:
            continue
        if i in seen or i < 0 or i >= total:
            continue
        rec = records[i]
        txt = rec.get("text", "")
        if not txt or not str(txt).strip():
            continue
        seen.add(i)
        contexts.append(clean_text(txt)[:1000])
        src_records.append({
            "file": rec.get("file", ""),
            "law": rec.get("law", ""),
            "dieu": rec.get("dieu", ""),
            "khoan": rec.get("khoan", "")
        })
        if len(contexts) >= 6:
            break

    # Fallback scan of processed directory for first token if no contexts
    if not contexts:
        kw = re.escape(query.split()[0]) if query.split() else None
        if kw:
            for root, _, files in os.walk(PROCESSED_DIR):
                for f in files:
                    if not f.lower().endswith((".txt", ".csv", ".json", ".parquet")):
                        continue
                    path = os.path.join(root, f)
                    try:
                        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                            txt = fh.read(200000)
                            if re.search(kw, txt, re.IGNORECASE):
                                contexts.append(clean_text(txt[:1000]))
                                src_records.append({"file": f, "law": "", "dieu": "", "khoan": ""})
                                if len(contexts) >= 3:
                                    break
                    except Exception:
                        try:
                            if f.lower().endswith(".parquet"):
                                df = pd.read_parquet(path)
                                txt2 = " ".join(df.astype(str).agg(" ".join, axis=1)[:200].tolist())
                            else:
                                df = pd.read_csv(path)
                                txt2 = " ".join(df.astype(str).agg(" ".join, axis=1)[:200].tolist())
                            if re.search(kw, txt2, re.IGNORECASE):
                                contexts.append(clean_text(txt2[:1000]))
                                src_records.append({"file": f, "law": "", "dieu": "", "khoan": ""})
                                if len(contexts) >= 3:
                                    break
                        except Exception:
                            continue
                if contexts:
                    break

    # Format sources
    formatted = format_sources(src_records)
    return contexts, formatted, src_records

# ---------------- LLM prompt + generate ----------------
def build_prompt(question: str, contexts: List[str]) -> str:
    ctx = "\n\n".join(f"- {c}" for c in contexts)
    prompt = f"""Bạn là trợ lý chứng khoán Việt Nam, trả lời bằng tiếng Việt chuẩn, ngắn gọn và chính xác.
Dựa hoàn toàn trên các đoạn dữ liệu tham khảo dưới đây (không phỏng đoán hoặc thêm thông tin ngoài dữ liệu).
Cấu trúc trả lời: Tóm tắt → Giải thích → (nếu có) Rủi ro → Kết luận ngắn.

Câu hỏi: {question}

Dữ liệu tham khảo:
{ctx}

Trả lời:
"""
    return prompt

def _clean_model_output(ans: str) -> str:
    if not ans:
        return ""
    #loại control chars nhưng giữ toàn bộ unicode hợp lệ
    ans = re.sub(r'[\x00-\x1F\x7F-\x9F]+', '', ans)
    # Loại các token id sequences rất dài dạng ABC123: giữ nếu có chữ thường/việt nam,
    # nhưng vẫn loại các chuỗi không chứa dấu chấm/phẩy/không phải câu.
    # Chỉ xóa những sequences quá dài không có dấu cách nếu toàn ký tự lạ:
    ans = re.sub(r'\s{2,}', ' ', ans).strip()

    lines = [l for l in ans.splitlines() if l.strip() != ""]
    deduped = []
    for l in lines:
        if not deduped or l.strip() != deduped[-1].strip():
            deduped.append(l)
    ans = "\n".join(deduped).strip()
    return ans

def generate_answer(prompt: str, tokenizer, model, max_new_tokens: int = 220) -> str:
    if tokenizer is None or model is None:
        return "Xin lỗi, mô hình ngôn ngữ chưa sẵn sàng để sinh câu trả lời. Vui lòng thử lại sau."
    try:
        # truncation & max_length to avoid huge inputs
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.25,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Trả lời:" in text:
            ans = text.split("Trả lời:")[-1].strip()
        else:
            # remove echoed prompt portion if present
            prompt_decoded = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
            if text.startswith(prompt_decoded):
                ans = text[len(prompt_decoded):].strip()
            else:
                ans = text.strip()
        ans = clean_text(ans)
        ans = _clean_model_output(ans)
        return ans
    except Exception as e:
        return f"Xin lỗi, có lỗi khi sinh câu trả lời: {e}"

# ---------------- Data query helpers ----------------
def handle_data_query(question: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Scan processed files to answer quantitative queries:
    - return_mean, volume, cpi, fx
    """
    q = question.lower()
    metric = None
    if any(x in q for x in ["return trung bình", "trung bình 1 ngày", "return trung binh", "return trung bình 1 ngày"]):
        metric = "return_mean"
    elif any(x in q for x in ["tổng volume", "tổng khối lượng", "volume lớn nhất", "volume giao dịch", "khối lượng giao dịch"]):
        metric = "volume"
    elif any(x in q for x in ["cpi", "lạm phát", "lạm phát cùng kì", "cpi_yoy"]):
        metric = "cpi"
    elif any(x in q for x in ["tỉ giá", "tỷ giá", "usd/vnd", "usd vnd", "usd_vnd", "usd"]):
        metric = "fx"

    ticker = None
    m = re.search(r"\b([A-Z]{2,5})\b", question)
    if m:
        ticker = m.group(1).upper()
    else:
        m2 = re.search(r"\b(fpt|hpg|vcb|vnm|vin|vhm|vnindex)\b", q)
        if m2:
            ticker = m2.group(1).upper()

    # candidate files prioritized
    candidate_files = []
    for root, _, files in os.walk(PROCESSED_DIR):
        for f in files:
            low = f.lower()
            if not low.endswith((".csv", ".parquet")):
                continue
            score = 0
            if any(k in low for k in ["usd", "tỉ giá", "ty_gia", "usd_vnd"]): score += 50
            if any(k in low for k in ["cpi", "lạm phát", "cpi_yoy"]): score += 45
            if any(k in low for k in ["ohlc", "ohlcv", "giá", "price", "25 cổ", "25 cổ phiếu"]): score += 40
            if "tech" in low or "tech_features" in low: score += 30
            if "qa" in low: score -= 20
            candidate_files.append((score, os.path.join(root, f)))
    candidate_files = sorted(candidate_files, key=lambda x: x[0], reverse=True)

    last_err = None
    for _, path in candidate_files:
        try:
            if path.lower().endswith(".parquet"):
                try:
                    df = pd.read_parquet(path)
                except Exception:
                    last_err = f"Lỗi đọc parquet {os.path.basename(path)}"
                    continue
            else:
                try:
                    df = pd.read_csv(path)
                except Exception:
                    last_err = f"Lỗi đọc csv {os.path.basename(path)}"
                    continue
        except Exception as e:
            last_err = f"Lỗi đọc {os.path.basename(path)}: {e}"
            continue

        # heuristics: find common column names
        cols_lower = {c.lower(): c for c in df.columns}
        col_symbol = None
        for cand in ["symbol", "ticker", "mã", "ma", "code", "stock"]:
            if cand in cols_lower:
                col_symbol = cols_lower[cand]; break
        col_date = None
        for cand in ["date", "ngày", "ngay", "trade_date", "time", "timestamp"]:
            if cand in cols_lower:
                col_date = cols_lower[cand]; break
        col_close = None
        for cand in ["close", "giá đóng cửa", "price", "close_price", "adj_close"]:
            if cand in cols_lower:
                col_close = cols_lower[cand]; break
        col_volume = None
        for cand in ["volume", "vol", "khối lượng", "khoi luong", "total_volume", "volume_trading"]:
            if cand in cols_lower:
                col_volume = cols_lower[cand]; break

        df_work = df.copy()

        if ticker and col_symbol:
            try:
                df_work = df_work[df_work[col_symbol].astype(str).str.upper() == ticker]
                if df_work.empty:
                    last_err = f"Không tìm thấy {ticker} trong {os.path.basename(path)}"
                    continue
            except Exception:
                last_err = f"Lỗi lọc ticker trong {os.path.basename(path)}"
                continue

        # filter by year if asked
        year = None
        m = re.search(r"\b(19|20)\d{2}\b", question)
        if m and col_date:
            try:
                year = int(m.group(0))
                dates = pd.to_datetime(df_work[col_date], errors="coerce")
                df_work = df_work[dates.dt.year == year]
                if df_work.empty:
                    last_err = f"Không có bản ghi cho năm {year} trong {os.path.basename(path)}"
                    continue
            except Exception:
                pass

        # metrics calculation
        if metric == "return_mean":
            if not col_close:
                last_err = f"File {os.path.basename(path)} không có cột giá để tính return."
                continue
            try:
                series = pd.to_numeric(df_work[col_close], errors="coerce").dropna()
                if len(series) < 2:
                    last_err = f"Dữ liệu không đủ trong {os.path.basename(path)} để tính return."
                    continue
                ret = series.pct_change().dropna()
                mean_ret = float(ret.mean())
                return {
                    "type": "return_mean",
                    "ticker": ticker or "",
                    "value": mean_ret,
                    "n": len(ret),
                    "file": os.path.basename(path)
                }, None
            except Exception as e:
                last_err = f"Lỗi tính return ở {os.path.basename(path)}: {e}"
                continue

        if metric == "volume":
            if not col_volume:
                last_err = f"File {os.path.basename(path)} không có cột volume."
                continue
            try:
                vol = pd.to_numeric(df_work[col_volume], errors="coerce").dropna()
                if vol.empty:
                    last_err = f"Volume rỗng trong {os.path.basename(path)}"
                    continue
                return {
                    "type": "volume",
                    "ticker": ticker or "",
                    "total": float(vol.sum()),
                    "max": float(vol.max()),
                    "n": len(vol),
                    "file": os.path.basename(path)
                }, None
            except Exception as e:
                last_err = f"Lỗi tính volume ở {os.path.basename(path)}: {e}"
                continue

        if metric == "cpi":
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not numeric_cols:
                for c in df.columns:
                    try:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                        numeric_cols.append(c); break
                    except:
                        pass
            if not numeric_cols:
                last_err = f"Không tìm thấy cột số trong {os.path.basename(path)}"
                continue
            val = pd.to_numeric(df[numeric_cols[0]], errors="coerce").dropna()
            if val.empty:
                last_err = f"Dữ liệu CPI rỗng trong {os.path.basename(path)}"
                continue
            return {"type": "cpi", "value": float(val.iloc[-1]), "file": os.path.basename(path)}, None

        if metric == "fx":
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not numeric_cols:
                for c in df.columns:
                    try:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                        numeric_cols.append(c); break
                    except:
                        pass
            if not numeric_cols:
                last_err = f"Không tìm thấy cột số trong {os.path.basename(path)}"
                continue
            val = pd.to_numeric(df[numeric_cols[0]], errors="coerce").dropna()
            if val.empty:
                last_err = f"Dữ liệu FX rỗng trong {os.path.basename(path)}"
                continue
            return {"type": "fx", "value": float(val.iloc[-1]), "file": os.path.basename(path)}, None

        # fallback: if no metric but ticker present -> summary
        if not metric and ticker:
            summary = {"ticker": ticker, "rows": len(df_work), "file": os.path.basename(path)}
            return {"type": "summary", "summary": summary}, None

    return None, last_err or "Không tìm thấy file dữ liệu phù hợp trong kho processed."

# ---------------- Chat loop ----------------
def chat():
    bm25, faiss_index, embed_model, meta, tokenizer, model = load_components()
    print("🤖 Chatbot chứng khoán Việt Nam - VietRAG sẵn sàng! Gõ 'exit' để thoát.\n")

    while True:
        try:
            q = input(" Nhà đầu tư: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n Hẹn gặp lại!"); break
        if not q:
            continue
        if q.lower() in ["exit", "quit"]:
            print(" Hẹn gặp lại!"); break

        intent = detect_intent(q)

        # OFF-TOPIC : polite decline
        if intent == "off_topic":
            ans = ("Mình là trợ lý chuyên về thông tin, dữ liệu và quy định thị trường chứng khoán. "
                   "Những câu hỏi so sánh cá nhân/giải trí nằm ngoài phạm vi chuyên môn. "
                   "Mình có thể hỗ trợ nếu bạn muốn kiểm tra dữ liệu, chỉ số hay văn bản pháp lý liên quan.")
            print("\n🤖 Chatbot:", ans, "\n")
            log_interaction(q, ans, [])
            continue

        # DATA QUERY
        if intent == "data_query":
            res, err = handle_data_query(q)
            if res:
                t = res.get("type")
                if t == "return_mean":
                    v = res["value"]
                    ticker = res.get("ticker", "")
                    ans = f"Cổ phiếu {ticker}: return trung bình 1 ngày ≈ {v:.6f} (~{v*100:.4f}%)."
                    src_file = res.get("file", "")
                    src_label = "Dữ liệu giao dịch (OHLCV)" if src_file else ""
                    print(f"\n🤖 Chatbot: {ans}\n Nguồn: - {src_label}\n")
                    log_interaction(q, ans, [src_label] if src_label else [])
                    continue
                if t == "volume":
                    ticker = res.get("ticker", "")
                    ans = f"Cổ phiếu {ticker}: tổng volume = {int(res['total']):,}, max = {int(res['max']):,}."
                    src_label = "Dữ liệu giao dịch (OHLCV)"
                    print(f"\n🤖 Chatbot: {ans}\n Nguồn: - {src_label}\n")
                    log_interaction(q, ans, [src_label]); continue
                if t in ("cpi", "fx"):
                    v = res.get("value")
                    src_label = "Dữ liệu CPI" if t=="cpi" else "Dữ liệu tỷ giá USD/VND"
                    ans = f"Kết quả: {v}"
                    print(f"\n🤖 Chatbot: {ans}\n Nguồn: - {src_label}\n")
                    log_interaction(q, ans, [src_label]); continue
                if t == "summary":
                    s = res["summary"]
                    ans = f"Tìm thấy dữ liệu cho {s.get('ticker')} (số bản ghi: {s.get('rows')})."
                    src_label = "Dữ liệu giao dịch (OHLCV)"
                    print(f"\n🤖 Chatbot: {ans}\n Nguồn: - {src_label}\n")
                    log_interaction(q, ans, [src_label]); continue
            else:
                err_msg = err or "Không thể trả lời truy vấn dữ liệu này."
                print(f"\n🤖 Chatbot: {err_msg}\n")
                log_interaction(q, err_msg, [])
                continue

        # ADVICE / COMPLAINTS
        if intent == "advice":
            try:
                contexts, sources, raw = retrieve_multi(q, bm25, faiss_index, embed_model, meta)
            except Exception:
                contexts, sources, raw = [], [], []
            if sources:
                general = ("Nếu bạn nghi ngờ bị lừa: 1) Giữ chứng cứ (biên lai, hợp đồng, lịch sử giao dịch, tin nhắn). "
                           "2) Khiếu nại lên công ty môi giới; nếu không giải quyết được, nộp đơn lên Ủy ban Chứng khoán/ Sở Giao dịch; "
                           "3) Gửi đơn đến cơ quan công an nếu có dấu hiệu tội phạm; 4) Xem xét liên hệ luật sư để được tư vấn.")
                print("\n🤖 Chatbot:", general, "\nNguồn:")
                for s in sources:
                    if any(x in s.lower() for x in ["dataset", "nội bộ", "qa"]):
                        continue
                    print(" -", s)
                print()
                log_interaction(q, general, sources)
                continue
            else:
                general = ("Nếu bạn nghi ngờ bị lừa: (1) Giữ chứng cứ mọi giao dịch, (2) Liên hệ công ty môi giới để khiếu nại, "
                           "(3) Nếu không giải quyết, nộp đơn lên Ủy ban Chứng khoán/ cơ quan công an, (4) Tìm tư vấn pháp lý.")
                print("\n🤖 Chatbot:", general, "\n")
                log_interaction(q, general, [])
                continue

        # COUNT / LEGAL LOOKUP / SUMMARIZE / DEFINITION
        if intent in ("count_articles", "legal_lookup", "summarize_articles", "definition"):
            ql = q.lower()
            law_token = ""
            m = re.search(r"(luật\s*[\w\s0-9\/\-]{1,40}|luật.*chứng khoán\s*\d{4}|\d{4}\s*luật)", ql)
            if m:
                law_token = m.group(0)
            else:
                m2 = re.search(r"(ngh[iị]\s*định\s*\d{1,6}(\/\d{1,6})?)", ql)
                if m2:
                    law_token = m2.group(0)
                else:
                    m3 = re.search(r"(th[oồ]ng\s*tư\s*\d{1,6}(\/\d{1,6})?)", ql)
                    if m3:
                        law_token = m3.group(0)

            dieu = ""
            khoan = ""
            m_d = re.search(r"điều\s*(\d+)", ql)
            if m_d:
                dieu = m_d.group(1)
            m_k = re.search(r"khoản\s*(\d+)", ql)
            if m_k:
                khoan = m_k.group(1)

            meta_full = meta or {}
            canonical = ""
            if law_token:
                try:
                    canonical = normalize_law_name(law_token, meta_full.get("records", []))
                except Exception:
                    canonical = law_token

            if intent == "count_articles":
                if canonical:
                    n = count_articles_in_law(meta_full, canonical)
                    if n and n > 0:
                        ans = f"{canonical} có {n} Điều (theo dữ liệu hiện có)."
                        print("\n🤖 Chatbot:", ans, "\n Nguồn:")
                        print(" -", canonical)
                        print()
                        log_interaction(q, ans, [canonical])
                        continue
                    else:
                        ans = f"Không tìm thấy Điều nào cho {canonical} trong dữ liệu hiện có."
                        print("\n🤖 Chatbot:", ans, "\n")
                        log_interaction(q, ans, [])
                        continue
                else:
                    print("\n🤖 Chatbot: Vui lòng nêu rõ tên văn bản (ví dụ: 'Nghị định 155/2020' hoặc 'Luật Chứng khoán 2019').\n")
                    continue

            if intent == "legal_lookup":
                found = find_article_records(meta_full, law=canonical or law_token, dieu=dieu, khoan=khoan)
                if found:
                    snippets = []
                    for r in found[:6]:
                        t = clean_text(r.get("text", ""))
                        if t:
                            s = t[:800]
                            if "." in s:
                                s = s.rsplit(".", 1)[0] + "."
                            snippets.append(s)
                    header = ""
                    if dieu:
                        header = f"Nội dung Điều {dieu}"
                        if khoan:
                            header += f" Khoản {khoan}"
                        header += ":\n\n"
                    ans = header + "\n\n".join(snippets)
                    sources = format_sources(found)
                    print("\n🤖 Chatbot:", ans, "\n Nguồn:")
                    for s in sources:
                        if any(x in s.lower() for x in ["dataset", "nội bộ", "qa"]):
                            continue
                        print(" -", s)
                    print()
                    log_interaction(q, ans, sources)
                    continue
                else:
                    print("\n🤖 Chatbot: Không tìm thấy Điều/Khoản bạn hỏi trong bộ dữ liệu pháp lý hiện tại.\n")
                    log_interaction(q, "Không tìm thấy điều/khoản", [])
                    continue

            if intent == "summarize_articles":
                found_many = find_article_records(meta_full, law=canonical or law_token, dieu="", khoan="")
                if found_many:
                    snippets = []
                    for r in found_many[:10]:
                        t = clean_text(r.get("text", ""))
                        if t:
                            snippets.append((t[:400].rsplit(".", 1)[0] + ".") if len(t) > 400 else t)
                    ans = "Tóm tắt các nội dung chính:\n\n" + "\n\n".join(snippets)
                    sources = format_sources(found_many[:6])
                    print("\n🤖 Chatbot:", ans, "\n Nguồn:")
                    for s in sources:
                        if any(x in s.lower() for x in ["dataset", "nội bộ", "qa"]):
                            continue
                        print(" -", s)
                    print()
                    log_interaction(q, ans, sources)
                    continue
                else:
                    print("\n🤖 Chatbot: Không tìm thấy nội dung để tóm tắt trong dữ liệu pháp lý.\n")
                    log_interaction(q, "Không tìm thấy nội dung để tóm tắt", [])
                    continue

            if intent == "definition":
                # retrieve with dynamic top_k + BM25 priority for short defs
                try:
                    contexts, sources, raw = retrieve_multi(q, bm25, faiss_index, embed_model, meta)
                except Exception:
                    contexts, sources, raw = [], [], []
                if contexts:
                    prompt = build_prompt(q, contexts)
                    ans = generate_answer(prompt, tokenizer, model)
                    # If answer looks empty or garbage, fallback to scanning text files for definitions
                    if not ans or re.match(r'^[^a-zA-Z0-9\u00C0-\u024F]', ans) and len(ans) < 6:
                        # fallback scan
                        kw = re.escape(q.split()[0]) if q.split() else None
                        fallback_texts = []
                        if kw:
                            for root, _, files in os.walk(PROCESSED_DIR):
                                for f in files:
                                    if not f.lower().endswith((".txt", ".csv", ".json", ".parquet")):
                                        continue
                                    path = os.path.join(root, f)
                                    try:
                                        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                                            txt = fh.read(200000)
                                            if re.search(kw, txt, re.IGNORECASE):
                                                fallback_texts.append(clean_text(txt[:1000]))
                                                if len(fallback_texts) >= 3:
                                                    break
                                    except Exception:
                                        try:
                                            if f.lower().endswith(".parquet"):
                                                df = pd.read_parquet(path)
                                                txt2 = " ".join(df.astype(str).agg(" ".join, axis=1)[:200].tolist())
                                            else:
                                                df = pd.read_csv(path)
                                                txt2 = " ".join(df.astype(str).agg(" ".join, axis=1)[:200].tolist())
                                            if re.search(kw, txt2, re.IGNORECASE):
                                                fallback_texts.append(clean_text(txt2[:1000]))
                                                if len(fallback_texts) >= 3:
                                                    break
                                        except Exception:
                                            continue
                                if fallback_texts:
                                    break
                        if fallback_texts:
                            prompt = build_prompt(q, fallback_texts)
                            ans2 = generate_answer(prompt, tokenizer, model)
                            ans2 = ans2 or ans
                            print(f"\n🤖 Chatbot: {ans2}\n Nguồn:")
                            # choose friendly sources
                            print(" - (Dữ liệu tham khảo nội bộ)")
                            print()
                            log_interaction(q, ans2, ["Dữ liệu nội bộ"])
                            continue

                    # normal path
                    print(f"\n🤖 Chatbot: {ans}\n Nguồn:")
                    for s in sources:
                        if any(x in s.lower() for x in ["dataset", "nội bộ", "qa"]):
                            continue
                        print(" -", s)
                    print()
                    log_interaction(q, ans, sources)
                    continue
                else:
                    # fallback scan directly
                    kw = re.escape(q.split()[0]) if q.split() else None
                    fallback_texts = []
                    if kw:
                        for root, _, files in os.walk(PROCESSED_DIR):
                            for f in files:
                                if not f.lower().endswith((".txt", ".csv", ".json", ".parquet")):
                                    continue
                                path = os.path.join(root, f)
                                try:
                                    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                                        txt = fh.read(200000)
                                        if re.search(kw, txt, re.IGNORECASE):
                                            fallback_texts.append(clean_text(txt[:1000]))
                                            if len(fallback_texts) >= 3:
                                                break
                                except Exception:
                                    try:
                                        if f.lower().endswith(".parquet"):
                                            df = pd.read_parquet(path)
                                            txt2 = " ".join(df.astype(str).agg(" ".join, axis=1)[:200].tolist())
                                        else:
                                            df = pd.read_csv(path)
                                            txt2 = " ".join(df.astype(str).agg(" ".join, axis=1)[:200].tolist())
                                        if re.search(kw, txt2, re.IGNORECASE):
                                            fallback_texts.append(clean_text(txt2[:1000]))
                                            if len(fallback_texts) >= 3:
                                                break
                                    except Exception:
                                        continue
                            if fallback_texts:
                                break
                    if fallback_texts:
                        prompt = build_prompt(q, fallback_texts)
                        ans2 = generate_answer(prompt, tokenizer, model)
                        print(f"\n🤖 Chatbot: {ans2}\n📚 Nguồn:")
                        print(" - (Dữ liệu tham khảo nội bộ)")
                        print()
                        log_interaction(q, ans2, ["Dữ liệu nội bộ"])
                        continue
                    else:
                        print("\n🤖 Chatbot: Xin lỗi, chưa tìm thấy định nghĩa/tài liệu phù hợp trong kho dữ liệu.\n")
                        log_interaction(q, "Không tìm thấy định nghĩa", [])
                        continue

        # FALLBACK: general RAG + LLM generation
        try:
            contexts, sources, raw = retrieve_multi(q, bm25, faiss_index, embed_model, meta)
        except Exception as e:
            contexts, sources, raw = [], [], []
        if not contexts:
            print("\n🤖 Xin lỗi, chưa tìm thấy dữ liệu phù hợp.\n")
            log_interaction(q, "Không tìm thấy dữ liệu phù hợp", [])
            continue
        prompt = build_prompt(q, contexts)
        ans = generate_answer(prompt, tokenizer, model)
        print(f"\n🤖 Chatbot: {ans}\n Nguồn:")
        if sources:
            for s in sources:
                if any(x in s.lower() for x in ["dataset", "nội bộ", "qa"]):
                    continue
                print(" -", s)
        else:
            print(" - (Nội dung được tạo từ dữ liệu nội bộ; không có nguồn pháp lý cụ thể.)")
        print()
        log_interaction(q, ans, sources)


# Entry
if __name__ == "__main__":
    try:
        chat()
    except Exception as e:
        print("Lỗi chương trình:", e)
        traceback.print_exc()  

     


















