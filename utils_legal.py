# scripts/utils_legal.py
# v11.6 - legal utilities
# - normalize digits (drop .0)
# - find_article_records usable by chatbot
# - format_sources: one-line per law, prefer Điều with Khoản, remove internal filenames

import re
from typing import List, Dict, Optional

def normalize_digits(x) -> str:
    try:
        if x is None or (isinstance(x, float) and str(x).lower() == "nan"): return ""
        s = str(x).strip()
        if s == "": return ""
        xf = float(s)
        xi = int(xf)
        return str(xi)
    except:
        return str(x).strip()

def normalize_law_name(query: str, records: List[Dict]) -> Optional[str]:
    """Try to map a loose query to a canonical law name present in records."""
    if not query:
        return None
    q = query.lower().strip()
    # exact substring match
    names = {}
    for r in records:
        ln = (r.get("law") or "").strip()
        if ln:
            names.setdefault(ln, 0)
            names[ln] += 1
    if not names:
        return None
    # exact substring
    for ln in names:
        if q in ln.lower() or ln.lower() in q:
            return ln
    # year/number matching
    digits = re.findall(r"\d{2,4}", q)
    for ln in names:
        for d in digits:
            if d in ln:
                return ln
    # fallback: choose most common name containing 'chứng' if query has 'chứng'
    if "chứng" in q:
        for ln in names:
            if "chứng" in ln.lower() and "khoán" in ln.lower():
                return ln
    return None

def find_article_records(meta: Dict, law: str = "", dieu: str = "", khoan: str = "") -> List[Dict]:
    """
    Search meta['records'] for matching law/dieu/khoan.
    Returns list of records (file, law, dieu, khoan, text).
    """
    out = []
    if not meta or "records" not in meta:
        return out
    records = meta["records"]
    canonical = None
    if law:
        canonical = normalize_law_name(law, records) or law
    for r in records:
        law_r = (r.get("law") or "").strip()
        dieu_r = normalize_digits(r.get("dieu",""))
        khoan_r = normalize_digits(r.get("khoan",""))
        # law filter
        if canonical:
            if canonical.lower() not in law_r.lower():
                continue
        # dieu filter
        if dieu:
            if not dieu_r:
                continue
            if normalize_digits(dieu) != dieu_r:
                continue
        # khoan filter
        if khoan:
            if not khoan_r:
                continue
            if normalize_digits(khoan) != khoan_r:
                continue
        out.append({
            "file": r.get("file",""),
            "law": law_r,
            "dieu": dieu_r,
            "khoan": khoan_r,
            "text": r.get("text","")
        })
    return out

def count_articles_in_law(meta: Dict, law_query: str) -> int:
    if not meta or "records" not in meta:
        return 0
    canonical = normalize_law_name(law_query, meta["records"]) or law_query
    s = set()
    for r in meta["records"]:
        law_r = (r.get("law") or "").strip()
        if canonical.lower() in law_r.lower():
            d = normalize_digits(r.get("dieu",""))
            if d:
                try:
                    s.add(int(d))
                except:
                    pass
    return len(s)

def _is_internal_file_label(s: str) -> bool:
    if not s: return False
    sl = s.lower()
    internal_keys = [".csv", ".parquet", "ohlc", "ohlcv", "giá", "price", "processed", "stock", "qa", "qa_pairs", "tech_features"]
    return any(k in sl for k in internal_keys)

def format_sources(records: List[Dict]) -> List[str]:
    """ Gom nhóm nguồn theo loại luật, mỗi loại chỉ 1 dòng duy nhất """

    def kind_of(rec):
        law = (rec.get("law") or "").lower()
        if "luật" in law: return "luat"
        if "nghị" in law or "nghi" in law: return "nghidinh"
        if "thông" in law or "thong" in law: return "thongtu"
        return "other"

    grouped = {}
    for r in records:
        k = kind_of(r)
        grouped.setdefault(k, []).append(r)

    out_lines = []

    def pick_best(lst):
        def safe_int(x):
            try:
                return int(float(x))
            except:
                return 9999
        with_k = [r for r in lst if r.get("khoan")]
        if with_k:
            return sorted(with_k, key=lambda r: safe_int(r.get("dieu", "")))[0]
        with_d = [r for r in lst if r.get("dieu")]
        if with_d:
            return sorted(with_d, key=lambda r: safe_int(r.get("dieu", "")))[0]
        return lst[0]

    for kind in ["luat", "nghidinh", "thongtu", "other"]:
        recs = grouped.get(kind, [])
        if not recs:
            continue
        rep = pick_best(recs)
        law_name = (rep.get("law") or "").strip()
        file = (rep.get("file") or "").lower()
        d = normalize_digits(rep.get("dieu", ""))
        k = normalize_digits(rep.get("khoan", ""))

        # bỏ file data nội bộ
        if not law_name:
            if any(x in file for x in ["ohlcv", "price", "giá"]):  
                out_lines.append("Dữ liệu giao dịch (OHLCV)")
            elif any(x in file for x in ["usd", "tỉ giá", "ty_gia"]):
                out_lines.append("Dữ liệu tỉ giá USD/VND")
            elif any(x in file for x in ["cpi", "lạm phát"]):
                out_lines.append("Dữ liệu vĩ mô (CPI)")
            elif "qa" in file:
                out_lines.append("QA dataset (nội bộ)")
            continue

        if d:
            if k:
                out_lines.append(f"{law_name} – Điều {d} Khoản {k}")
            else:
                out_lines.append(f"{law_name} – Điều {d}")
        else:
            out_lines.append(law_name)

    seen, final = set(), []
    for i in out_lines:
        if i not in seen:
            seen.add(i)
            final.append(i)
    return final
