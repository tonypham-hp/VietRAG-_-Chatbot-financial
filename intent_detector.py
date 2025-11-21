# scripts/intent_detector.py
# intent detector: legal / data / technical / advice / off_topic / count / summarize / general

import re

def detect_intent(question: str) -> str:
    """
    Trả về một trong:
    'off_topic', 'data_query', 'technical', 'advice', 'count_articles',
    'summarize_articles', 'legal_lookup', 'general'
    """
    q = (question or "").lower().strip()   

    # OFF-TOPIC (so sánh cá nhân, xúc phạm, đời tư, v.v.)
    if re.search(r"\b(ai|tôi|mình).*(đẹp|xinh|đẹp trai|đẹp hơn|hát hay|hát tốt|giỏi|giọng|so sánh|ai tốt hơn|xấu|cao|thấp)\b", q):
        return "off_topic"

    # DATA QUERIES (price/ohlcv/fx/cpi/volume/return)
    if re.search(r"\b(return|trung bình 1 ngày|return trung bình|tỉ lệ return|volume giao dịch|tổng volume|volume|cpi|lạm phát|tỉ giá|usd/vnd|usd vnd|tỷ giá|tỉ giá)\b", q):
        return "data_query"

    # TECHNICAL INDICATORS
    if re.search(r"\b(rsi|macd|arima|monte carlo|max drawdown|drawdown|bollinger|volatility|obv|on-balance|rsi14)\b", q):
        return "technical"

    # ADVICE / COMPLAINTS
    if re.search(r"\b(lừa đảo|môi giới lừa|bị lừa|khiếu nại|phải làm sao|tố cáo|báo cơ quan|bị lừa đảo|tư vấn)\b", q):
        return "advice"

    # COUNT articles (how many Điều)
    if re.search(r"(bao nhiêu điều|có bao nhiêu điều|tổng cộng \d+ điều|mấy điều|số điều)", q):
        return "count_articles"

    # SUMMARIZE
    if re.search(r"(tóm tắt|tóm tắt.*điều|tóm tắt.*nghị định|tóm tắt.*thông tư|tóm tắt.*luật|tóm tắt.*\d+ điều)", q):
        return "summarize_articles"

    # LEGAL LOOKUP: explicit "Điều", "Khoản", "Điểm" or mentions law/regulation
    if re.search(r"(điều\s*\d+)|(khoản\s*\d+)|(điểm\s*[a-z0-9])", q) or re.search(r"\b(luật|ngh[iị]\s*định|th[oồ]ng\s*tư|thông tư|quy chế|quy định)\b", q):
        return "legal_lookup"

    # default
    return "general"

