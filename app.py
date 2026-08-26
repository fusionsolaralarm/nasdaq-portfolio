import base64
import io
import json
import os
import time
from datetime import datetime, timezone
from typing import Optional
 
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from PIL import Image
 
try:
    import openai
except ImportError:  # pragma: no cover
    openai = None
 
 
# =========================================================
# PAGE CONFIG
# =========================================================
 
st.set_page_config(
    page_title="SIMON Quant & Portfolio AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
BENCHMARK_TICKER = "SPY"
RISK_FREE_RATE = 0.04  # annualized, used for Sharpe ratio
 
 
# =========================================================
# THEME
# =========================================================
 
st.sidebar.header("🎨 ตั้งค่าการแสดงผล")
 
theme_mode = st.sidebar.selectbox(
    "เลือกธีมหน้าจอ",
    ["โหมดกลางคืน (Dark Mode)", "โหมดกลางวัน (Light Mode)"],
)
 
IS_DARK = theme_mode.startswith("โหมดกลางคืน")
 
if IS_DARK:
    bg_color = "#0e1117"
    card_bg = "#161b22"
    text_color = "#c9d1d9"
    heading_color = "#58a6ff"
    border_color = "#30363d"
    plot_template = "plotly_dark"
    up_color, down_color = "#3fb950", "#f85149"
else:
    bg_color = "#ffffff"
    card_bg = "#f6f8fa"
    text_color = "#24292e"
    heading_color = "#0366d6"
    border_color = "#e1e4e8"
    plot_template = "plotly_white"
    up_color, down_color = "#1a7f37", "#cf222e"
 
 
# =========================================================
# CSS
# =========================================================
 
st.markdown(
    f"""
    <style>
    .main {{ background-color: {bg_color}; }}
 
    .stock-card, .article-box, .small-card {{
        background-color: {card_bg};
        border: 1px solid {border_color};
        border-radius: 12px;
        padding: 18px 20px;
        color: {text_color};
        margin-bottom: 14px;
    }}
    .article-box {{ line-height: 1.7; }}
 
    .verdict-box-yes, .action-box-buy {{
        background-color: rgba(35, 134, 54, 0.15);
        border: 1px solid #238636;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 10px;
        color: {text_color};
    }}
    .verdict-box-no, .action-box-sell {{
        background-color: rgba(218, 54, 51, 0.15);
        border: 1px solid #da3633;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 10px;
        color: {text_color};
    }}
 
    h1, h2, h3 {{ color: {heading_color}; }}
    .app-subtitle {{ text-align:center; color:{text_color}; opacity:0.85; }}
    </style>
    """,
    unsafe_allow_html=True,
)
 
 
# =========================================================
# STORAGE
# =========================================================
 
PORTFOLIO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio_data.json")
 
DEFAULT_PORTFOLIO = pd.DataFrame(
    [
        {"Ticker": "AAPL", "Shares": 10.0, "Buy Price (USD)": 170.0, "Asset Class": "Growth"},
        {"Ticker": "TSLA", "Shares": 5.0, "Buy Price (USD)": 220.0, "Asset Class": "Growth"},
    ]
)
 
 
def load_portfolio_from_disk() -> pd.DataFrame:
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                records = json.load(f)
            loaded = pd.DataFrame(records)
            required = {"Ticker", "Shares", "Buy Price (USD)"}
            if not loaded.empty and required.issubset(loaded.columns):
                if "Asset Class" not in loaded.columns:
                    loaded["Asset Class"] = "Growth"
                return loaded[["Ticker", "Shares", "Buy Price (USD)", "Asset Class"]]
        except Exception:
            pass
    return DEFAULT_PORTFOLIO.copy()
 
 
def save_portfolio_to_disk(df: pd.DataFrame) -> None:
    try:
        df.to_json(PORTFOLIO_FILE, orient="records", force_ascii=False)
    except Exception as e:
        st.sidebar.error(f"บันทึกพอร์ตลงดิสก์ไม่สำเร็จ: {e}")
 
 
# =========================================================
# HEADER
# =========================================================
 
st.markdown(
    f"""
    <h1 style="text-align:center; color:{heading_color};">
        📊 SIMON QUANT &amp; PORTFOLIO AI ADVISOR
    </h1>
    <p class="app-subtitle">
        ระบบสแกนหุ้นสหรัฐฯ วิเคราะห์หุ้นรายตัว และวิเคราะห์พอร์ตด้วยข้อมูลเชิงปริมาณ
    </p>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")
 
 
# =========================================================
# SYMBOL LIST
# =========================================================
 
@st.cache_data
def get_curated_symbols():
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX", "AMD", "INTC",
        "CRM", "ADBE", "ORCL", "IBM", "NOW", "SNOW", "PLTR", "DDOG", "CRWD", "JPM",
        "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "PYPL", "WMT", "COST",
        "TGT", "HD", "NKE", "SBUX", "MCD", "KO", "PEP", "DIS", "UNH", "JNJ",
        "PFE", "ABBV", "LLY", "MRK", "AMGN", "TMO", "XOM", "CVX", "CAT", "BA",
        "GE", "HON", "UPS", "LMT", "ASTS", "LUNR", "IONQ", "SOUN", "RGTI", "QBTS",
        "ACHR", "JOBY", "RIVN", "HOOD", "ARM", "SMCI", "RDDT", "DJT", "BBAI", "HLGN",
        "CLSK", "RIOT", "MARA", "HUT", "BITF", "OPEN", "LMND", "AVGO", "ACN", "CSCO",
        "TXN", "QCOM", "AMAT", "INTU", "BKNG", "ISRG",
    ]
 
 
@st.cache_data(ttl=86400)
def get_full_us_market_symbols():
    try:
        nasdaq_url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
        other_url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
 
        nasdaq_df = pd.read_csv(nasdaq_url, sep="|").iloc[:-1]
        nasdaq_df = nasdaq_df[nasdaq_df["Test Issue"] == "N"]
        nasdaq_df = nasdaq_df.rename(columns={"Security Name": "Name"})[["Symbol", "Name", "ETF"]]
 
        other_df = pd.read_csv(other_url, sep="|").iloc[:-1]
        other_df = other_df[other_df["Test Issue"] == "N"]
        other_df = other_df.rename(columns={"ACT Symbol": "Symbol", "Security Name": "Name"})[
            ["Symbol", "Name", "ETF"]
        ]
 
        combined = pd.concat([nasdaq_df, other_df], ignore_index=True)
        combined["ETF"] = combined["ETF"].astype(str).str.upper().eq("Y")
        combined["Symbol"] = combined["Symbol"].astype(str).str.strip()
        combined = combined[combined["Symbol"].str.match(r"^[A-Za-z.]+$", na=False)]
        combined["Symbol"] = combined["Symbol"].str.replace(".", "-", regex=False)
        combined = combined.drop_duplicates(subset="Symbol").sort_values("Symbol").reset_index(drop=True)
 
        if combined.empty:
            raise ValueError("รายชื่อหุ้นว่างเปล่า")
        return combined
 
    except Exception as e:
        st.sidebar.warning(f"ดึงรายชื่อหุ้นทั้งหมดไม่สำเร็จ ({e}) ใช้รายชื่อคัดสรรแทน")
        curated = get_curated_symbols()
        return pd.DataFrame({"Symbol": curated, "Name": curated, "ETF": False})
 
 
# =========================================================
# ROBUST DOWNLOAD HELPER (retry once)
# =========================================================
 
def robust_download(*args, retries: int = 1, **kwargs):
    """Wrap yf.download with one retry on failure/empty result."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            hist = yf.download(*args, **kwargs)
            if hist is not None and not hist.empty:
                return hist
        except Exception as e:
            last_err = e
        if attempt < retries:
            time.sleep(0.6)
    if last_err:
        raise last_err
    return pd.DataFrame()
 
 
# =========================================================
# QUANT MODEL
# =========================================================
 
def compute_simons_row(ticker, close_series):
    if close_series is None:
        return None
    close_series = close_series.dropna()
    if len(close_series) < 30:
        return None
 
    current_p = float(close_series.iloc[-1])
    price_1m_ago = float(close_series.iloc[-min(20, len(close_series))])
    price_1w_ago = float(close_series.iloc[-min(5, len(close_series))])
 
    return_1m = (current_p - price_1m_ago) / price_1m_ago * 100
    return_1w = (current_p - price_1w_ago) / price_1w_ago * 100
    sma_50 = float(close_series.rolling(min(30, len(close_series))).mean().iloc[-1])
    volatility = float(close_series.pct_change().std() * np.sqrt(252) * 100)
 
    score = 50
    score += 25 if current_p > sma_50 else -20
    score += 20 if return_1m > 0 else -15
    score += 15 if volatility < 40 else -10
    score = max(0, min(100, score))
 
    if score >= 70:
        signal = "🟢 ซื้อสะสม (Strong Buy)"
    elif score >= 45:
        signal = "🟡 ถือ / เฝ้าสังเกต (Hold)"
    else:
        signal = "🔴 ควรขาย / หลีกเลี่ยง (Sell / Avoid)"
 
    if return_1w > 0.5:
        trend = "ขาขึ้น"
    elif return_1w < -0.5:
        trend = "ขาลง"
    else:
        trend = "ไซด์เวย์"
 
    return {
        "Ticker": ticker,
        "Simons Signal": signal,
        "Score": score,
        "Price (USD)": current_p,
        "Momentum 1M (%)": return_1m,
        "Momentum 1W (%)": return_1w,
        "Trend": trend,
        "Volatility (%)": volatility,
        "SMA 50": sma_50,
    }
 
 
# =========================================================
# FETCH MARKET DATA
# =========================================================
 
@st.cache_data(ttl=1800)
def fetch_stock_data_and_simons_logic(ticker_list):
    data_list = []
    for ticker in ticker_list:
        try:
            hist = robust_download(ticker, period="6mo", progress=False, auto_adjust=True)
            if hist.empty:
                continue
            close_series = hist["Close"].iloc[:, 0] if isinstance(hist.columns, pd.MultiIndex) else hist["Close"]
            row = compute_simons_row(ticker, close_series)
            if row:
                data_list.append(row)
        except Exception:
            continue
    return pd.DataFrame(data_list)
 
 
def fetch_stock_data_batched(ticker_list, chunk_size=50):
    all_rows = []
    total = len(ticker_list)
    progress_bar = st.progress(0.0, text="กำลังสแกนตลาด...")
 
    for i in range(0, total, chunk_size):
        chunk = ticker_list[i:i + chunk_size]
        try:
            data = yf.download(chunk, period="6mo", group_by="ticker", threads=True, progress=False, auto_adjust=True)
        except Exception:
            data = None
 
        if data is not None and not data.empty:
            for ticker in chunk:
                try:
                    if len(chunk) == 1 or not isinstance(data.columns, pd.MultiIndex):
                        close_series = data["Close"]
                    else:
                        if ticker not in data.columns.get_level_values(0):
                            continue
                        close_series = data[ticker]["Close"]
                    row = compute_simons_row(ticker, close_series)
                    if row:
                        all_rows.append(row)
                except Exception:
                    continue
 
        done = min(i + chunk_size, total)
        progress_bar.progress(done / total, text=f"สแกนแล้ว {done:,}/{total:,} ตัว | สำเร็จ {len(all_rows):,} ตัว")
 
    progress_bar.empty()
    return pd.DataFrame(all_rows)
 
 
# =========================================================
# COMPANY INFO
# =========================================================
 
@st.cache_data(ttl=1800)
def get_company_business_info(ticker_symbol):
    try:
        info = yf.Ticker(ticker_symbol).info
        return (
            info.get("longName", ticker_symbol),
            info.get("sector", "N/A"),
            info.get("industry", "N/A"),
            info.get("longBusinessSummary", "ไม่มีข้อมูลสรุปธุรกิจ"),
        )
    except Exception:
        return (ticker_symbol, "N/A", "N/A", "ไม่สามารถดึงข้อมูลธุรกิจได้")
 
 
@st.cache_data(ttl=1800)
def get_sector_only(ticker_symbol) -> str:
    try:
        info = yf.Ticker(ticker_symbol).info
        return info.get("sector") or "N/A"
    except Exception:
        return "N/A"
 
 
# =========================================================
# FORMATTING HELPERS
# =========================================================
 
def format_money(value):
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except Exception:
        return "N/A"
    if abs(value) >= 1e12:
        return f"${value / 1e12:.2f}T"
    if abs(value) >= 1e9:
        return f"${value / 1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:.2f}M"
    return f"${value:,.0f}"
 
 
def pct_text(value):
    return "N/A" if value is None else f"{float(value) * 100:+.2f}%"
 
 
def metric_or_na(value, suffix=""):
    return "N/A" if value is None else f"{float(value):.2f}{suffix}"
 
 
# =========================================================
# MARKET FRESHNESS / QUOTE STATUS
# =========================================================
 
@st.cache_data(ttl=60)
def get_market_quote_snapshot(ticker_symbol):
    """ดึงราคาล่าสุดที่ Yahoo Finance เปิดให้เข้าถึง พร้อมเวลาของแท่งล่าสุด
 
    หมายเหตุ: yfinance ไม่รับประกัน real-time feed ระดับโบรกเกอร์
    จึงแสดงสถานะเป็น Latest / Delayed แทนการอ้างว่า real-time 100%
    """
    try:
        hist = robust_download(ticker_symbol, period="1d", interval="1m", progress=False, auto_adjust=False, prepost=False)
        if hist is None or hist.empty:
            return {"price": None, "timestamp": None, "status": "❌ ไม่มี", "age_minutes": None}
 
        close = hist["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        if close.empty:
            return {"price": None, "timestamp": None, "status": "❌ ไม่มีราคาล่าสุด", "age_minutes": None}
 
        ts = close.index[-1]
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.tz_localize("UTC")
        now = datetime.now(timezone.utc)
        age_minutes = max(0.0, (now - ts.to_pydatetime().astimezone(timezone.utc)).total_seconds() / 60.0)
 
        if age_minutes <= 5:
            status = "🟢 ล่าสุดจากตลาด / Latest (อาจมี Delay)"
        elif age_minutes <= 30:
            status = "🟡 แบบมี Delay / Delayed"
        else:
            status = "🔴 ข้อมูลเก่า / Stale"
 
        return {
            "price": float(close.iloc[-1]),
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "status": status,
            "age_minutes": age_minutes,
        }
    except Exception as e:
        return {"price": None, "timestamp": None, "status": f"⚠️ ดึงข้อมูลไม่ได้: {e}", "age_minutes": None}
 
 
# =========================================================
# THAI / ENGLISH BUSINESS LEARNING HELPERS
# =========================================================
 
BUSINESS_TERM_TRANSLATIONS = {
    "business": "ธุรกิจ", "business model": "รูปแบบธุรกิจ", "business segment": "กลุ่มธุรกิจ",
    "revenue": "รายได้", "sales": "ยอดขาย", "earnings": "ผลประกอบการ / กำไร",
    "profit": "กำไร", "net income": "กำไรสุทธิ", "operating income": "กำไรจากการดำเนินงาน",
    "gross margin": "อัตรากำไรขั้นต้น", "operating margin": "อัตรากำไรจากการดำเนินงาน",
    "profit margin": "อัตรากำไรสุทธิ", "free cash flow": "กระแสเงินสดอิสระ", "cash flow": "กระแสเงินสด",
    "market share": "ส่วนแบ่งตลาด", "market": "ตลาด", "customer": "ลูกค้า",
    "enterprise": "องค์กร / บริษัทขนาดใหญ่", "software": "ซอฟต์แวร์", "hardware": "ฮาร์ดแวร์",
    "platform": "แพลตฟอร์ม", "cloud": "คลาวด์", "data center": "ศูนย์ข้อมูล",
    "artificial intelligence": "ปัญญาประดิษฐ์ (AI)", "semiconductor": "เซมิคอนดักเตอร์ / ชิป",
    "technology": "เทคโนโลยี", "products": "ผลิตภัณฑ์", "services": "บริการ",
    "solution": "โซลูชัน / แนวทางแก้ปัญหา", "solutions": "โซลูชัน / แนวทางแก้ปัญหา",
    "growth": "การเติบโต", "demand": "ความต้องการ", "supply": "อุปทาน", "competition": "การแข่งขัน",
    "competitive advantage": "ความได้เปรียบในการแข่งขัน", "pricing power": "ความสามารถในการตั้งราคา",
    "investment": "การลงทุน", "capital expenditure": "รายจ่ายลงทุน (CapEx)",
    "research and development": "การวิจัยและพัฒนา (R&D)", "research": "การวิจัย", "development": "การพัฒนา",
    "manufacturing": "การผลิต", "international": "ระหว่างประเทศ", "global": "ระดับโลก",
    "portfolio": "พอร์ตการลงทุน",
}
 
SECTOR_TRANSLATIONS = {
    "Technology": "เทคโนโลยี", "Financial Services": "บริการทางการเงิน", "Healthcare": "การดูแลสุขภาพ",
    "Consumer Cyclical": "สินค้าอุปโภคบริโภคตามวัฏจักร",
    "Consumer Defensive": "สินค้าอุปโภคบริโภคที่มีความต้องการค่อนข้างสม่ำเสมอ",
    "Industrials": "อุตสาหกรรม", "Communication Services": "บริการสื่อสาร", "Energy": "พลังงาน",
    "Utilities": "สาธารณูปโภค", "Real Estate": "อสังหาริมทรัพย์", "Basic Materials": "วัสดุพื้นฐาน",
}
 
 
def company_learning_terms(summary, limit=12):
    text = str(summary or "").lower()
    found = []
    for english, thai in BUSINESS_TERM_TRANSLATIONS.items():
        if english.lower() in text and english not in [x[0] for x in found]:
            found.append((english, thai))
        if len(found) >= limit:
            break
    return found
 
 
def translate_sector_name(value):
    value = str(value or "N/A")
    return SECTOR_TRANSLATIONS.get(value, value)
 
 
# =========================================================
# DETAILED STOCK ANALYSIS
# =========================================================
 
@st.cache_data(ttl=1800)
def get_detailed_stock_analysis(ticker):
    try:
        info = yf.Ticker(ticker).info
 
        def safe_num(key):
            value = info.get(key)
            try:
                return float(value) if value is not None else None
            except Exception:
                return None
 
        data = {
            "long_name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "summary": info.get("longBusinessSummary", "ไม่มีข้อมูลธุรกิจ"),
            "market_cap": safe_num("marketCap"),
            "enterprise_value": safe_num("enterpriseValue"),
            "revenue": safe_num("totalRevenue"),
            "revenue_growth": safe_num("revenueGrowth"),
            "gross_margin": safe_num("grossMargins"),
            "operating_margin": safe_num("operatingMargins"),
            "profit_margin": safe_num("profitMargins"),
            "eps": safe_num("trailingEps"),
            "forward_eps": safe_num("forwardEps"),
            "pe": safe_num("trailingPE"),
            "forward_pe": safe_num("forwardPE"),
            "peg": safe_num("pegRatio"),
            "price_to_sales": safe_num("priceToSalesTrailing12Months"),
            "earnings_growth": safe_num("earningsGrowth"),
            "earnings_quarterly_growth": safe_num("earningsQuarterlyGrowth"),
            "free_cash_flow": safe_num("freeCashflow"),
            "operating_cash_flow": safe_num("operatingCashflow"),
            "debt_to_equity": safe_num("debtToEquity"),
            "current_ratio": safe_num("currentRatio"),
            "return_on_equity": safe_num("returnOnEquity"),
            "return_on_assets": safe_num("returnOnAssets"),
            "beta": safe_num("beta"),
            "dividend_yield": safe_num("dividendYield"),
            "target_mean": safe_num("targetMeanPrice"),
            "target_high": safe_num("targetHighPrice"),
            "target_low": safe_num("targetLowPrice"),
            "fifty_two_high": safe_num("fiftyTwoWeekHigh"),
            "fifty_two_low": safe_num("fiftyTwoWeekLow"),
            "employees": safe_num("fullTimeEmployees"),
            "country": info.get("country"),
            "website": info.get("website"),
            "recommendation_key": info.get("recommendationKey"),
        }
 
        growth_score = 0
        if data["revenue_growth"] is not None:
            growth_score += 3 if data["revenue_growth"] > 0.20 else 2 if data["revenue_growth"] > 0.08 else 1 if data["revenue_growth"] > 0 else -1
        if data["earnings_growth"] is not None:
            growth_score += 3 if data["earnings_growth"] > 0.20 else 2 if data["earnings_growth"] > 0.05 else 1 if data["earnings_growth"] > 0 else -1
 
        quality_score = 0
        if data["profit_margin"] is not None:
            quality_score += 3 if data["profit_margin"] > 0.20 else 2 if data["profit_margin"] > 0.10 else 1 if data["profit_margin"] > 0 else 0
        if data["return_on_equity"] is not None:
            quality_score += 3 if data["return_on_equity"] > 0.20 else 2 if data["return_on_equity"] > 0.10 else 1 if data["return_on_equity"] > 0 else 0
 
        debt_score = 0
        if data["debt_to_equity"] is not None:
            d = data["debt_to_equity"]
            debt_score = 3 if d < 50 else 2 if d < 100 else 1 if d < 200 else -1
 
        pe = data["forward_pe"] or data["pe"]
        if pe is None:
            valuation_text = "ไม่สามารถประเมิน Valuation ได้"
        elif pe < 15:
            valuation_text = "Valuation ค่อนข้างต่ำ เมื่อเทียบกับหุ้น Growth ทั่วไป"
        elif pe < 25:
            valuation_text = "Valuation อยู่ในระดับปานกลาง"
        elif pe < 40:
            valuation_text = "Valuation ค่อนข้างสูง สะท้อนความคาดหวังการเติบโต"
        else:
            valuation_text = "Valuation สูงมาก ราคาหุ้นต้องการการเติบโตของกำไรสูง"
 
        sector = str(data["sector"]).lower()
        industry = str(data["industry"]).lower()
 
        sector_playbooks = {
            "tech": (
                ["AI Infrastructure และ Data Center", "Cloud Computing", "Enterprise AI", "Advanced Computing",
                 "Semiconductor Demand", "Automation และ Software Intelligence", "การเพิ่มงบลงทุนของ Hyperscalers"],
                ["รายได้จาก AI เติบโตเร็วกว่าคาด", "การเปิดตัวผลิตภัณฑ์ใหม่", "การเพิ่ม Data Center CapEx",
                 "การขยายตลาด Enterprise", "Margin ขยายตัวจาก Economies of Scale", "การเพิ่ม Market Share"],
                ["การแข่งขันทางเทคโนโลยี", "Semiconductor Cyclicality", "Supply Chain", "Regulation",
                 "Export Restrictions", "Valuation สูง", "การชะลอตัวของ AI CapEx"],
            ),
            "health": (
                ["Aging Population", "Precision Medicine", "Biotechnology", "Medical Technology",
                 "Chronic Disease Treatment", "AI in Healthcare"],
                ["FDA Approval", "Clinical Trial Success", "ยอดขายยาใหม่", "การขยายตลาดต่างประเทศ", "การเปิดตัวผลิตภัณฑ์ใหม่"],
                ["Clinical Trial Failure", "FDA / Regulatory Risk", "Patent Expiration", "Drug Pricing Pressure",
                 "การแข่งขันจาก Generic Drugs"],
            ),
            "financial": (
                ["ทิศทางอัตราดอกเบี้ย", "Credit Growth", "Digital Banking", "Capital Market Activity",
                 "Wealth Management", "AI ในระบบการเงิน"],
                ["Credit Growth", "Investment Banking Recovery", "Trading Revenue", "Net Interest Income", "Cost Reduction"],
                ["Credit Risk", "Interest Rate Risk", "Recession", "Regulatory Risk", "Loan Losses"],
            ),
            "consumer": (
                ["กำลังซื้อผู้บริโภค", "E-Commerce", "Brand Strength", "Pricing Power",
                 "International Expansion", "Digital Transformation"],
                ["ยอดขายเติบโต", "Same Store Sales", "E-Commerce Growth", "การขยายสาขา", "การเพิ่ม Market Share"],
                ["Consumer Spending Slowdown", "Inflation", "ต้นทุนสินค้า", "การแข่งขันด้านราคา", "Supply Chain"],
            ),
            "industrial": (
                ["Infrastructure Spending", "Automation", "Industrial Digitization", "Energy Transition",
                 "Manufacturing Investment"],
                ["คำสั่งซื้อใหม่", "Infrastructure Spending", "Automation Demand", "Margin Expansion", "Global Expansion"],
                ["Economic Slowdown", "Raw Material Cost", "Supply Chain", "Interest Rate", "Cyclical Demand"],
            ),
        }
 
        if "technology" in sector or "semiconductor" in industry:
            outlook, catalysts, risks = sector_playbooks["tech"]
        elif "health" in sector:
            outlook, catalysts, risks = sector_playbooks["health"]
        elif "financial" in sector:
            outlook, catalysts, risks = sector_playbooks["financial"]
        elif "consumer" in sector:
            outlook, catalysts, risks = sector_playbooks["consumer"]
        elif "industrial" in sector:
            outlook, catalysts, risks = sector_playbooks["industrial"]
        else:
            outlook = ["การเติบโตของตลาด", "การขยายฐานลูกค้า", "การพัฒนาผลิตภัณฑ์", "Digital Transformation",
                       "การเพิ่มประสิทธิภาพต้นทุน", "การขยายตลาดต่างประเทศ"]
            catalysts = ["Revenue Growth", "Earnings Growth", "New Products", "Market Expansion", "Margin Expansion"]
            risks = ["เศรษฐกิจมหภาค", "การแข่งขัน", "ต้นทุน", "ดอกเบี้ย", "Regulation", "Demand Slowdown"]
 
        if growth_score >= 4:
            medium_term = "แนวโน้มระยะกลางเป็นบวก หากบริษัทสามารถรักษาการเติบโตของรายได้และกำไรได้ต่อเนื่อง"
        elif growth_score >= 2:
            medium_term = "มีโอกาสเติบโต แต่ควรติดตามผลประกอบการรายไตรมาสและ Guidance อย่างใกล้ชิด"
        else:
            medium_term = "ควรรอดูการฟื้นตัวของรายได้และกำไร ก่อนเพิ่มน้ำหนักการลงทุน"
 
        if quality_score >= 4:
            long_term = "คุณภาพธุรกิจอยู่ในระดับน่าสนใจ หากบริษัทสามารถรักษา Margin, ROE และกระแสเงินสดได้ในระยะยาว"
        else:
            long_term = "การลงทุนระยะยาวควรติดตามความสามารถในการสร้างกำไรและ Free Cash Flow อย่างต่อเนื่อง"
 
        return {
            "data": data,
            "growth_score": growth_score,
            "quality_score": quality_score,
            "debt_score": debt_score,
            "valuation_text": valuation_text,
            "outlook": outlook,
            "catalysts": catalysts,
            "risks": risks,
            "medium_term": medium_term,
            "long_term": long_term,
        }
 
    except Exception as e:
        return {"error": str(e)}
 
 
def build_stock_specific_outlook(data, ticker, t_price, t_score, t_trend, t_mom1m, t_mom1w, t_vol):
    """สร้างมุมมองอนาคตแบบเจาะจงรายบริษัทจากข้อมูลพื้นฐาน + ราคา"""
    sector = str(data.get("sector") or "N/A")
    industry = str(data.get("industry") or "N/A")
    summary = str(data.get("summary") or "ไม่มีข้อมูลธุรกิจ")
    revenue_growth = data.get("revenue_growth")
    earnings_growth = data.get("earnings_growth")
    profit_margin = data.get("profit_margin")
    roe = data.get("return_on_equity")
    fcf = data.get("free_cash_flow")
    debt = data.get("debt_to_equity")
    forward_pe = data.get("forward_pe")
    pe = data.get("pe")
    target = data.get("target_mean")
    high52 = data.get("fifty_two_high")
    low52 = data.get("fifty_two_low")
 
    s = (sector + " " + industry).lower()
 
    playbook = {
        "business_drivers": ["การเติบโตของรายได้และฐานลูกค้า", "การเพิ่ม market share",
                              "การเปิดตลาดหรือผลิตภัณฑ์ใหม่", "การเพิ่มประสิทธิภาพต้นทุนและกระแสเงินสด"],
        "catalysts": ["Revenue / Earnings สูงกว่าคาด", "Margin ขยายตัว", "ผลิตภัณฑ์ใหม่หรือการขยายตลาด",
                       "Free Cash Flow แข็งแรงขึ้น"],
        "risks": ["เศรษฐกิจมหภาค", "การแข่งขัน", "ต้นทุนและ Supply Chain", "ดอกเบี้ยและ Regulation"],
    }
 
    if "semiconductor" in s:
        playbook = {
            "business_drivers": ["การเติบโตของ AI accelerator / data-center และความต้องการ compute",
                                  "การลงทุนของ Cloud/Hyperscaler และรอบ CapEx ของ Data Center",
                                  "เทคโนโลยี node/process, packaging, networking หรือ memory ตามตำแหน่งของบริษัท",
                                  "การเพิ่ม market share และการเปิดตัวผลิตภัณฑ์รุ่นใหม่"],
            "catalysts": ["ยอดสั่งซื้อชิปหรือระบบ AI เพิ่มขึ้น", "Gross Margin / Operating Margin ขยายตัว",
                          "การเปิดตัวผลิตภัณฑ์ใหม่ที่มี ASP และกำไรสูงขึ้น",
                          "การเพิ่มกำลังผลิตโดยไม่ทำให้ต้นทุนต่อหน่วยสูงขึ้น"],
            "risks": ["วัฏจักร Semiconductor และการปรับลด Inventory", "การแข่งขันด้านเทคโนโลยีและการเปลี่ยน architecture",
                      "Export controls / Geopolitical risk", "Valuation สูงเกินกว่าการเติบโตของกำไรจริง"],
        }
    elif "technology" in s or "software" in s:
        playbook = {
            "business_drivers": ["การเพิ่มจำนวนลูกค้าและรายได้ต่อหนึ่งลูกค้า", "Recurring revenue / subscription และการรักษา retention",
                                  "AI, cloud และ automation ที่เพิ่มมูลค่าต่อผลิตภัณฑ์", "Operating leverage เมื่อรายได้โตเร็วกว่าค่าใช้จ่าย"],
            "catalysts": ["Net new customers / bookings สูงกว่าคาด", "การเพิ่ม ARPU และ cross-sell",
                          "AI monetization ที่แปลงเป็นรายได้จริง", "Free Cash Flow และ Margin ดีขึ้น"],
            "risks": ["การแข่งขันและการลดราคา", "AI disruption ต่อผลิตภัณฑ์เดิม", "ลูกค้าชะลอ IT spending",
                      "Valuation ที่พึ่งพาการเติบโตระยะยาวสูง"],
        }
    elif "health" in s or "biotech" in s or "drug" in s:
        playbook = {
            "business_drivers": ["การเติบโตของผลิตภัณฑ์/ยาหลักและ pipeline", "ตลาดผู้สูงอายุและโรคเรื้อรัง",
                                  "การอนุมัติผลิตภัณฑ์และการขยาย indication", "การขยายตลาดและความสามารถในการรักษา pricing power"],
            "catalysts": ["ผลการทดลองหรือ FDA approval ที่เป็นบวก", "ยอดขายผลิตภัณฑ์หลักสูงกว่าคาด",
                          "การเพิ่ม indication / geographic expansion", "Margin และ Free Cash Flow ดีขึ้น"],
            "risks": ["Clinical / regulatory risk", "Patent expiry และการแข่งขัน Generic", "การควบคุมราคายา",
                      "Pipeline ไม่สามารถชดเชยผลิตภัณฑ์เดิมได้"],
        }
    elif "financial" in s:
        playbook = {
            "business_drivers": ["ทิศทางอัตราดอกเบี้ยและ Net Interest Margin", "การเติบโตของสินเชื่อและคุณภาพสินทรัพย์",
                                  "Investment banking / trading / wealth management", "ประสิทธิภาพต้นทุนและ digital transformation"],
            "catalysts": ["Credit growth และ fee income ดีขึ้น", "Credit losses ต่ำกว่าคาด", "ต้นทุนดำเนินงานลดลง",
                          "Capital return / buyback สนับสนุน EPS"],
            "risks": ["Credit losses และเศรษฐกิจถดถอย", "Interest-rate sensitivity", "Regulatory capital requirement",
                      "Yield curve / liquidity pressure"],
        }
    elif "consumer" in s or "retail" in s:
        playbook = {
            "business_drivers": ["กำลังซื้อและจำนวนลูกค้า", "Same-store sales / e-commerce growth",
                                  "Pricing power และความแข็งแรงของแบรนด์", "การบริหาร Inventory และ Supply Chain"],
            "catalysts": ["ยอดขายสาขาเดิมและ e-commerce สูงกว่าคาด", "Gross Margin ฟื้นตัว", "การเพิ่ม market share",
                          "การขยายสาขาหรือสินค้าใหม่"],
            "risks": ["ผู้บริโภคลดการใช้จ่าย", "เงินเฟ้อและต้นทุนสินค้า", "การแข่งขันด้านราคา", "Inventory สูงและ markdown pressure"],
        }
    elif "industrial" in s:
        playbook = {
            "business_drivers": ["คำสั่งซื้อใหม่และ backlog", "Infrastructure / manufacturing CapEx",
                                  "Automation และ productivity", "การควบคุมต้นทุนและ operating leverage"],
            "catalysts": ["Backlog เพิ่มขึ้น", "คำสั่งซื้อและ utilization สูงขึ้น", "Margin expansion", "การลงทุนโครงสร้างพื้นฐาน"],
            "risks": ["เศรษฐกิจชะลอตัวและคำสั่งซื้อลด", "ต้นทุนวัตถุดิบ", "Supply-chain disruption", "Cyclical demand"],
        }
 
    business_drivers = playbook["business_drivers"]
    catalysts = playbook["catalysts"]
    risks = list(playbook["risks"])
 
    strengths, watch = [], []
    if revenue_growth is not None:
        strengths.append(f"รายได้เติบโต {revenue_growth * 100:+.1f}%")
        watch.append("ติดตาม Revenue Growth และ Guidance รายไตรมาส")
    if earnings_growth is not None:
        strengths.append(f"กำไรเติบโต {earnings_growth * 100:+.1f}%")
        watch.append("ติดตาม EPS / Earnings Growth ว่าโตจากธุรกิจจริงหรือรายการพิเศษ")
    if profit_margin is not None:
        strengths.append(f"Net Margin {profit_margin * 100:.1f}%")
        watch.append("ติดตาม Margin ว่ารักษาระดับได้หรือไม่")
    if roe is not None:
        strengths.append(f"ROE {roe * 100:.1f}%")
    if fcf is not None:
        strengths.append(f"Free Cash Flow {format_money(fcf)}")
    if debt is not None:
        if debt < 80:
            strengths.append(f"Debt/Equity {debt:.1f} อยู่ในระดับไม่สูงมาก")
        elif debt > 150:
            risks.insert(0, f"Debt/Equity สูงที่ {debt:.1f} ต้องติดตามภาระหนี้และดอกเบี้ย")
 
    chosen_pe = forward_pe if forward_pe is not None else pe
    if chosen_pe is None:
        valuation_signal = "ประเมินไม่ได้"
    elif chosen_pe < 15:
        valuation_signal = "ค่อนข้างถูกเมื่อเทียบกับหุ้นที่มีกำไร"
    elif chosen_pe < 25:
        valuation_signal = "ระดับปานกลาง"
    elif chosen_pe < 40:
        valuation_signal = "ค่อนข้างสูง ต้องมีการเติบโตของกำไรสนับสนุน"
    else:
        valuation_signal = "สูงมาก มีความคาดหวังการเติบโตสูง"
 
    target_upside = (target / t_price - 1) * 100 if (target is not None and t_price > 0) else None
    range_position = (
        max(0, min(100, (t_price - low52) / (high52 - low52) * 100))
        if (low52 is not None and high52 is not None and high52 > low52)
        else None
    )
 
    if not strengths:
        strengths.append("ข้อมูลพื้นฐานบางรายการจากแหล่งข้อมูลไม่ครบ จึงควรตรวจสอบงบล่าสุดเพิ่มเติม")
    if not watch:
        watch.append("ติดตามงบการเงิน, Guidance และกระแสเงินสดรายไตรมาส")
 
    if t_score >= 70 and t_mom1m > 0:
        near_term = "ภาพระยะสั้นถึงกลางเป็นบวก: Quant Score สูงและโมเมนตัม 1 เดือนเป็นบวก แต่ไม่ควรไล่ราคาเมื่อ Valuation สูงมาก"
    elif t_score >= 45:
        near_term = "ภาพระยะสั้นถึงกลางเป็นกลาง: ควรรอการยืนยันจาก Momentum, Earnings และ Guidance ก่อนเพิ่มน้ำหนัก"
    else:
        near_term = "ภาพระยะสั้นถึงกลางค่อนข้างอ่อนแอ: ควรให้ความสำคัญกับการรักษาเงินต้นและรอสัญญาณ Trend กลับตัว"
 
    if revenue_growth is not None and revenue_growth > 0.15 and (earnings_growth is None or earnings_growth > 0):
        long_term = ("พื้นฐานมีโอกาสสนับสนุนการเติบโตระยะยาว เพราะรายได้ยังขยายตัวและกำไรไม่ได้สวนทาง "
                     "อย่างไรก็ตามต้องพิสูจน์ว่าการเติบโตสามารถแปลงเป็น Free Cash Flow ได้ต่อเนื่อง")
    elif profit_margin is not None and profit_margin > 0.15 and fcf is not None and fcf > 0:
        long_term = ("คุณภาพธุรกิจน่าสนใจจาก Margin และกระแสเงินสด แต่การเติบโตในอนาคตต้องติดตามว่าบริษัทสามารถ"
                     "เพิ่มรายได้โดยไม่ต้องใช้เงินลงทุนสูงเกินไปหรือไม่")
    else:
        long_term = "แนวโน้มระยะยาวยังขึ้นกับการเติบโตของรายได้ กำไร และ Free Cash Flow มากกว่าการเคลื่อนไหวของราคาหุ้นระยะสั้น"
 
    return {
        "business_drivers": business_drivers,
        "catalysts": catalysts,
        "risks": risks,
        "strengths": strengths,
        "watch": watch,
        "valuation_signal": valuation_signal,
        "target_upside": target_upside,
        "range_position": range_position,
        "near_term": near_term,
        "long_term": long_term,
        "sector_industry": f"{sector} / {industry}",
        "company_summary": summary,
        "profile": {
            "country": data.get("country"),
            "website": data.get("website"),
            "employees": data.get("employees"),
            "market_cap": data.get("market_cap"),
            "enterprise_value": data.get("enterprise_value"),
            "recommendation_key": data.get("recommendation_key"),
            "beta": data.get("beta"),
            "current_ratio": data.get("current_ratio"),
            "gross_margin": data.get("gross_margin"),
            "operating_margin": data.get("operating_margin"),
            "roe": roe,
            "roa": data.get("return_on_assets"),
            "peg": data.get("peg"),
        },
    }
 
 
# =========================================================
# PORTFOLIO RISK ANALYTICS (NEW)
# =========================================================
 
@st.cache_data(ttl=1800)
def fetch_price_history_matrix(tickers: tuple, period: str = "1y") -> pd.DataFrame:
    """Return a DataFrame of adjusted close prices, one column per ticker."""
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(list(tickers), period=period, auto_adjust=True, progress=False, group_by="ticker", threads=True)
    except Exception:
        return pd.DataFrame()
 
    series = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if t in raw.columns.get_level_values(0):
                series[t] = raw[t]["Close"]
    else:
        if len(tickers) == 1:
            series[tickers[0]] = raw["Close"]
 
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).dropna(how="all")
 
 
def compute_portfolio_risk_metrics(df_res: pd.DataFrame):
    """Compute volatility, Sharpe, max drawdown, beta vs benchmark, and HHI
    concentration for the current portfolio, using real daily returns."""
    tickers = tuple(sorted(df_res["Ticker"].unique().tolist()))
    if not tickers:
        return None
 
    price_matrix = fetch_price_history_matrix(tickers + (BENCHMARK_TICKER,))
    if price_matrix.empty or BENCHMARK_TICKER not in price_matrix.columns:
        return None
 
    weights = df_res.set_index("Ticker")["Weight in Portfolio (%)"] / 100.0
    weights = weights.reindex(tickers).fillna(0)
 
    held_prices = price_matrix[[t for t in tickers if t in price_matrix.columns]].dropna(how="any")
    if held_prices.empty or held_prices.shape[0] < 20:
        return None
 
    daily_returns = held_prices.pct_change().dropna()
    aligned_weights = weights.reindex(held_prices.columns).fillna(0)
    aligned_weights = aligned_weights / aligned_weights.sum() if aligned_weights.sum() > 0 else aligned_weights
 
    portfolio_daily_returns = (daily_returns * aligned_weights).sum(axis=1)
 
    ann_vol = float(portfolio_daily_returns.std() * np.sqrt(252) * 100)
    ann_return = float(portfolio_daily_returns.mean() * 252)
    sharpe = (ann_return - RISK_FREE_RATE) / (ann_vol / 100) if ann_vol > 0 else None
 
    cum = (1 + portfolio_daily_returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum / running_max - 1)
    max_drawdown = float(drawdown.min() * 100)
 
    benchmark_returns = price_matrix[BENCHMARK_TICKER].reindex(held_prices.index).pct_change().dropna()
    common_idx = portfolio_daily_returns.index.intersection(benchmark_returns.index)
    beta = None
    if len(common_idx) > 20:
        cov = np.cov(portfolio_daily_returns.loc[common_idx], benchmark_returns.loc[common_idx])
        if cov[1, 1] != 0:
            beta = float(cov[0, 1] / cov[1, 1])
 
    hhi = float(((aligned_weights) ** 2).sum() * 10000)  # 0 (diversified) to 10000 (single stock)
 
    correlation_matrix = daily_returns.corr()
 
    portfolio_cum_norm = 100 * cum / cum.iloc[0]
    bench_cum = (1 + benchmark_returns.reindex(cum.index).fillna(0)).cumprod()
    benchmark_cum_norm = 100 * bench_cum / bench_cum.iloc[0] if not bench_cum.empty else None
 
    return {
        "ann_vol": ann_vol,
        "ann_return": ann_return * 100,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "beta": beta,
        "hhi": hhi,
        "correlation_matrix": correlation_matrix,
        "portfolio_cum_norm": portfolio_cum_norm,
        "benchmark_cum_norm": benchmark_cum_norm,
    }
 
 
def hhi_read_out(hhi: float) -> str:
    if hhi >= 4000:
        return "🔴 กระจุกตัวสูง — พอร์ตพึ่งพาหุ้นไม่กี่ตัวมากเกินไป ความเสี่ยงเฉพาะตัวหุ้นสูง"
    if hhi >= 2000:
        return "🟡 กระจุกตัวปานกลาง — มีการกระจายความเสี่ยงอยู่บ้าง แต่ยังพึ่งพาหุ้นบางตัวค่อนข้างมาก"
    return "🟢 กระจายความเสี่ยงดี — น้ำหนักหุ้นในพอร์ตค่อนข้างสมดุล"
 
 
# =========================================================
# OPENAI CLIENT
# =========================================================
 
@st.cache_resource(show_spinner=False)
def get_openai_client(api_key: str):
    if openai is None:
        return None
    return openai.OpenAI(api_key=api_key)
 
 
def resolve_api_key(secret_name: str, sidebar_label: str, widget_key: str) -> Optional[str]:
    """Prefer st.secrets (safer for deployed apps); fall back to a sidebar input."""
    secret_val = st.secrets.get(secret_name) if hasattr(st, "secrets") else None
    if secret_val:
        return secret_val
    return st.sidebar.text_input(sidebar_label, type="password", key=widget_key) or None
 
 
# =========================================================
# SIDEBAR SCANNER
# =========================================================
 
st.sidebar.header("🔎 ตั้งค่าแหล่งข้อมูล & สแกนหุ้น")
 
scan_mode = st.sidebar.radio(
    "เลือกแหล่งข้อมูลหุ้น",
    ["รายชื่อคัดสรร (~85 ตัว, เร็ว)", "ทั้งตลาดหุ้นสหรัฐฯ (NASDAQ + NYSE + AMEX)"],
)
 
 
# =========================================================
# LOAD SCANNER DATA
# =========================================================
 
if scan_mode.startswith("รายชื่อคัดสรร"):
    symbols = get_curated_symbols()
    with st.spinner("กำลังประมวลผลโมเดล Quant..."):
        df = fetch_stock_data_and_simons_logic(symbols)
else:
    with st.spinner("กำลังดึงรายชื่อหลักทรัพย์ทั้งหมด..."):
        full_list_df = get_full_us_market_symbols()
 
    st.sidebar.success(f"พบหลักทรัพย์ทั้งหมด {len(full_list_df):,} ตัว")
 
    exclude_etf = st.sidebar.checkbox("ไม่รวมกองทุน ETF", value=True)
    universe_df = full_list_df[~full_list_df["ETF"]] if exclude_etf else full_list_df
 
    max_scan = st.sidebar.number_input(
        "จำนวนหุ้นสูงสุดที่จะสแกน", min_value=50, max_value=int(len(universe_df)),
        value=min(300, int(len(universe_df))), step=50,
    )
    start_offset = st.sidebar.number_input(
        "เริ่มสแกนจากลำดับที่", min_value=0, max_value=max(0, int(len(universe_df)) - 1),
        value=0, step=max_scan,
    )
    run_scan = st.sidebar.button("🚀 เริ่มสแกนตลาดเต็มรูปแบบ", use_container_width=True)
 
    if run_scan:
        target_symbols = universe_df["Symbol"].iloc[start_offset:start_offset + max_scan].tolist()
        st.session_state.full_scan_df = fetch_stock_data_batched(target_symbols)
        st.session_state.full_scan_range = (start_offset, start_offset + len(target_symbols))
 
    if "full_scan_df" in st.session_state:
        df = st.session_state.full_scan_df
        rng = st.session_state.get("full_scan_range", (0, len(df)))
        st.sidebar.caption(f"ผลสแกนล่าสุด: {rng[0]:,}-{rng[1]:,}")
    else:
        st.info("👈 กรุณาคลิก 'เริ่มสแกนตลาด' ที่ Sidebar")
        st.stop()
 
if df.empty:
    st.error("ไม่สามารถดึงข้อมูลตลาดได้ กรุณารีเฟรชหน้าเว็บ")
    st.stop()
 
 
# =========================================================
# MAIN TABS
# =========================================================
 
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 สแกนหุ้นตลาด", "🔬 วิเคราะห์หุ้นรายตัว", "💼 พอร์ตของฉัน", "🤖 AI Portfolio Advisor"]
)
 
 
# =========================================================
# TAB 1 — SCANNER
# =========================================================
 
with tab1:
    st.subheader("📈 Quantitative Stock Screener")
 
    col1, col2 = st.columns(2)
    with col1:
        signal_filter = st.selectbox(
            "กรองตามคำแนะนำ",
            ["ทั้งหมด", "🟢 ซื้อสะสม (Strong Buy)", "🟡 ถือ / เฝ้าสังเกต (Hold)", "🔴 ควรขาย / หลีกเลี่ยง (Sell / Avoid)"],
        )
    with col2:
        search_query = st.text_input("🔍 ค้นหา Ticker", "").upper()
 
    df_filtered = df.copy()
    if signal_filter != "ทั้งหมด":
        df_filtered = df_filtered[df_filtered["Simons Signal"] == signal_filter]
    if search_query:
        df_filtered = df_filtered[df_filtered["Ticker"].str.contains(search_query)]
 
    k1, k2, k3 = st.columns(3)
    k1.metric("จำนวนหุ้นทั้งหมดในผล", f"{len(df):,}")
    k2.metric("ตรงเงื่อนไขตัวกรอง", f"{len(df_filtered):,}")
    k3.metric("Avg Score (ที่กรองแล้ว)", f"{df_filtered['Score'].mean():.0f}" if not df_filtered.empty else "N/A")
 
    st.dataframe(
        df_filtered.sort_values(by="Score", ascending=False),
        column_config={
            "Price (USD)": st.column_config.NumberColumn(format="$%.2f"),
            "Momentum 1M (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Momentum 1W (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Volatility (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Score": st.column_config.NumberColumn(format="%d คะแนน"),
        },
        use_container_width=True,
        hide_index=True,
    )
 
    st.download_button(
        "⬇️ ดาวน์โหลดผลสแกน CSV",
        data=df_filtered.to_csv(index=False).encode("utf-8-sig"),
        file_name="scanner_results.csv",
        mime="text/csv",
    )
 
 
# =========================================================
# TAB 2 — SINGLE STOCK DEEP DIVE
# =========================================================
 
with tab2:
    st.subheader("🔬 วิเคราะห์หุ้นรายตัวแบบละเอียด")
    st.write(
        "กรอก Ticker เพื่อดูข้อมูลบริษัท พื้นฐาน การเติบโต Valuation "
        "Momentum ความเสี่ยง และแนวโน้มธุรกิจในอนาคตแบบเจาะจงรายหุ้น"
    )
 
    inner_search_ticker = st.text_input("🏷️ Ticker หุ้น", "NVDA", key="detail_ticker").upper().strip()
 
    if not inner_search_ticker:
        st.info("กรุณากรอก Ticker")
    else:
        match_dd = df[df["Ticker"] == inner_search_ticker]
 
        if match_dd.empty:
            try:
                with st.spinner(f"กำลังดึงข้อมูล {inner_search_ticker}..."):
                    hist_dd = robust_download(inner_search_ticker, period="6mo", progress=False, auto_adjust=True)
                    if not hist_dd.empty:
                        cs_dd = hist_dd["Close"].iloc[:, 0] if isinstance(hist_dd.columns, pd.MultiIndex) else hist_dd["Close"]
                        row_dd = compute_simons_row(inner_search_ticker, cs_dd)
                        if row_dd:
                            match_dd = pd.DataFrame([row_dd])
            except Exception as e:
                st.warning(f"ไม่สามารถดึงราคาของ {inner_search_ticker}: {e}")
 
        if match_dd.empty:
            st.error(f"ไม่พบข้อมูลสำหรับ {inner_search_ticker}")
        else:
            r = match_dd.iloc[0]
            t_ticker = str(r["Ticker"])
            t_price = float(r["Price (USD)"])
            t_score = int(r["Score"])
            t_trend = str(r["Trend"])
            t_mom1m = float(r["Momentum 1M (%)"])
            t_mom1w = float(r["Momentum 1W (%)"])
            t_vol = float(r["Volatility (%)"])
            t_sma50 = float(r["SMA 50"])
 
            quote = get_market_quote_snapshot(t_ticker)
            if quote.get("price") is not None:
                t_price = quote["price"]
 
            with st.spinner(f"กำลังวิเคราะห์ข้อมูลพื้นฐานของ {t_ticker}..."):
                analysis = get_detailed_stock_analysis(t_ticker)
 
            if "error" in analysis:
                st.error(f"ไม่สามารถสร้างรายงานได้: {analysis['error']}")
            else:
                data = analysis["data"]
                outlook = build_stock_specific_outlook(data, t_ticker, t_price, t_score, t_trend, t_mom1m, t_mom1w, t_vol)
 
                st.markdown("---")
                st.title(f"🏢 {t_ticker}")
                st.caption(f"{data['long_name']} | {data['sector']} | {data['industry']}")
 
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("ราคาล่าสุด", f"${t_price:,.2f}")
                c2.metric("Quant Score", f"{t_score}/100")
                c3.metric("1M Momentum", f"{t_mom1m:+.2f}%")
                c4.metric("1W Momentum", f"{t_mom1w:+.2f}%")
                c5.metric("Volatility", f"{t_vol:.2f}%")
 
                st.markdown("#### 🕒 สถานะราคา (Market Data Freshness)")
                q1, q2, q3 = st.columns(3)
                q1.metric("สถานะ", quote.get("status", "N/A"))
                q2.metric("เวลา", quote.get("timestamp") or "N/A")
                age = quote.get("age_minutes")
                q3.metric("อายุ", f"{age:.1f} นาที" if age is not None else "N/A")
                st.caption(
                    "หมายเหตุ: yfinance/Yahoo Finance ไม่รับประกันข้อมูล Real-time ระดับโบรกเกอร์ "
                    "สถานะสีเขียวหมายถึงที่เข้าถึงได้ ไม่ได้ยืนยันว่าเป็น Real-time 100%"
                )
 
                st.markdown("---")
                st.subheader("1️⃣ บริษัททำธุรกิจอะไร? (Business Profile)")
                st.write(data["summary"])
 
                st.markdown("#### 📚 คำศัพท์จากข้อมูลบริษัท (Business Vocabulary)")
                learning_terms = company_learning_terms(data["summary"], limit=12)
                if learning_terms:
                    vocab_rows = [{"English": en, "คำแปลไทย": th} for en, th in learning_terms]
                    st.dataframe(pd.DataFrame(vocab_rows), use_container_width=True, hide_index=True)
                else:
                    st.caption("ยังไม่พบคำศัพท์ที่ตรงกับรายการฝึกภาษาในคำอธิบายบริษัท")
 
                st.markdown("#### 🇹🇭🇬🇧 สรุปข้อมูลบริษัทแบบไทย + English")
                st.write(f"บริษัท: **{data['long_name']}** | Company: **{data['long_name']}**")
                st.write(f"กลุ่มธุรกิจ (Sector): **{data['sector']}** — {translate_sector_name(data['sector'])}")
                st.write(f"อุตสาหกรรม (Industry): **{data['industry']}**")
 
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Sector", data["sector"])
                p2.metric("Industry", data["industry"])
                p3.metric("Market Cap", format_money(data["market_cap"]))
                p4.metric("Enterprise Value", format_money(data["enterprise_value"]))
 
                profile = outlook["profile"]
                extra = []
                if profile["country"]:
                    extra.append(f"ประเทศ: {profile['country']}")
                if profile["employees"] is not None:
                    extra.append(f"พนักงาน: {int(profile['employees']):,} คน")
                if profile["recommendation_key"]:
                    extra.append(f"Consensus: {profile['recommendation_key']}")
                if profile["beta"] is not None:
                    extra.append(f"Beta: {profile['beta']:.2f}")
                if extra:
                    st.caption(" | ".join(extra))
 
                st.subheader("2️⃣ แนวโน้มธุรกิจในอนาคตแบบเจาะจงรายหุ้น (Future Business Outlook)")
                st.write(
                    f"หุ้น {t_ticker} อยู่ในกลุ่ม {outlook['sector_industry']} "
                    "ดังนั้นการประเมินอนาคตควรเชื่อมโยง 3 เรื่องเข้าด้วยกัน: "
                    "การเติบโตของตลาด, ความสามารถของบริษัทในการรักษา Margin/Market Share "
                    "และราคาที่นักลงทุนยอมจ่ายให้กับกำไรในอนาคต"
                )
 
                oc1, oc2 = st.columns(2)
                with oc1:
                    st.markdown("#### ⚙️ เครื่องยนต์ธุรกิจที่ต้องจับตา")
                    for item in outlook["business_drivers"]:
                        st.markdown(f"- {item}")
                with oc2:
                    st.markdown("#### 💪 จุดแข็งที่เห็นจากข้อมูล")
                    for item in outlook["strengths"]:
                        st.markdown(f"- {item}")
 
                st.markdown("#### 📅 แนวโน้มระยะใกล้–กลาง")
                st.info(outlook["near_term"])
                st.markdown("#### 🔭 แนวโน้มระยะยาว")
                st.info(outlook["long_term"])
 
                st.subheader("3️⃣ ข้อมูลการเงินและปัจจัยพื้นฐาน (Financial & Fundamental Snapshot)")
                f1, f2, f3, f4 = st.columns(4)
                f1.metric("Revenue", format_money(data["revenue"]))
                f2.metric("Revenue Growth", pct_text(data["revenue_growth"]))
                f3.metric("Earnings Growth", pct_text(data["earnings_growth"]))
                f4.metric("Free Cash Flow", format_money(data["free_cash_flow"]))
 
                f5, f6, f7, f8 = st.columns(4)
                f5.metric("Gross Margin", pct_text(data["gross_margin"]))
                f6.metric("Operating Margin", pct_text(data["operating_margin"]))
                f7.metric("Net Margin", pct_text(data["profit_margin"]))
                f8.metric("ROE", pct_text(data["return_on_equity"]))
 
                f9, f10, f11, f12 = st.columns(4)
                f9.metric("ROA", pct_text(data["return_on_assets"]))
                f10.metric("Debt / Equity", metric_or_na(data["debt_to_equity"]))
                f11.metric("Current Ratio", metric_or_na(data["current_ratio"]))
                f12.metric("EPS", f"${data['eps']:.2f}" if data["eps"] is not None else "N/A")
 
                st.subheader("4️⃣ Growth Engine")
                st.write("ดูว่าการเติบโตของบริษัทเกิดจาก Revenue, Earnings และ Cash Flow หรือเป็นเพียงการขยาย Valuation")
                g1, g2, g3 = st.columns(3)
                g1.metric("Growth Score", f"{analysis['growth_score']}/6")
                g2.metric("Quality Score", f"{analysis['quality_score']}/6")
                g3.metric("Debt Score", f"{analysis['debt_score']}/3")
 
                growth_notes = []
                if data["revenue_growth"] is not None:
                    growth_notes.append(f"Revenue Growth {data['revenue_growth']*100:+.2f}%")
                if data["earnings_growth"] is not None:
                    growth_notes.append(f"Earnings Growth {data['earnings_growth']*100:+.2f}%")
                if data["free_cash_flow"] is not None:
                    growth_notes.append(f"Free Cash Flow {format_money(data['free_cash_flow'])}")
                st.write(" | ".join(growth_notes) if growth_notes else "ข้อมูล Growth ไม่ครบ")
 
                st.subheader("5️⃣ Valuation")
                v1, v2, v3, v4 = st.columns(4)
                v1.metric("P/E", metric_or_na(data["pe"], "x"))
                v2.metric("Forward P/E", metric_or_na(data["forward_pe"], "x"))
                v3.metric("PEG", metric_or_na(data["peg"]))
                v4.metric("P/S", metric_or_na(data["price_to_sales"], "x"))
                st.info(f"มุมมอง Valuation: {analysis['valuation_text']} ({outlook['valuation_signal']})")
 
                st.subheader("6️⃣ ราคาเป้าหมายจากนักวิเคราะห์ (Analyst Target Price)")
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Target Low", f"${data['target_low']:,.2f}" if data["target_low"] is not None else "N/A")
                a2.metric("Target Mean", f"${data['target_mean']:,.2f}" if data["target_mean"] is not None else "N/A")
                a3.metric("Target High", f"${data['target_high']:,.2f}" if data["target_high"] is not None else "N/A")
                a4.metric("Upside to Mean", f"{outlook['target_upside']:+.2f}%" if outlook["target_upside"] is not None else "N/A")
 
                oc3, oc4 = st.columns(2)
                with oc3:
                    st.subheader("7️⃣ ปัจจัยกระตุ้นราคาหุ้น (Catalysts)")
                    for item in outlook["catalysts"]:
                        st.markdown(f"- **{item}**")
                with oc4:
                    st.subheader("8️⃣ ⚠️ ความเสี่ยงเฉพาะหุ้น")
                    for item in outlook["risks"]:
                        st.markdown(f"- **{item}**")
 
                st.subheader("9️⃣ การวิเคราะห์ทางเทคนิคและโมเมนตัม (Technical & Momentum)")
                tc1, tc2, tc3, tc4, tc5 = st.columns(5)
                tc1.metric("1 Week", f"{t_mom1w:+.2f}%")
                tc2.metric("1 Month", f"{t_mom1m:+.2f}%")
                tc3.metric("SMA 50", f"${t_sma50:,.2f}")
                tc4.metric("Trend", t_trend)
                tc5.metric("Quant Signal", str(r["Simons Signal"]))
 
                if t_price > t_sma50:
                    st.success("ราคาปัจจุบันอยู่เหนือ SMA 50 — โครงสร้าง Momentum ระยะกลางเป็นบวก")
                else:
                    st.warning("ราคาปัจจุบันต่ำกว่า SMA 50 — โครงสร้าง Momentum ระยะกลางยังอ่อนแอ")
 
                st.subheader("🔟 โปรไฟล์ความเสี่ยง (Risk Profile)")
                rp1, rp2, rp3 = st.columns(3)
                rp1.metric("Annualized Volatility", f"{t_vol:.2f}%")
                rp2.metric("Beta", metric_or_na(data["beta"]))
                rp3.metric("Debt / Equity", metric_or_na(data["debt_to_equity"]))
 
                if t_vol >= 60:
                    st.error("ความผันผวนสูงมาก: ควรลด Position Size และหลีกเลี่ยงการใช้เงินก้อนใหญ่ในจังหวะเดียว")
                elif t_vol >= 40:
                    st.warning("ความผันผวนค่อนข้างสูง: เหมาะกับการทยอยซื้อและกำหนดขนาด Position ให้เหมาะสม")
                else:
                    st.success("ความผันผวนอยู่ในระดับที่ต่ำกว่าเกณฑ์ 40% ของโมเดล Quant")
 
                st.subheader("1️⃣1️⃣ ตำแหน่งราคาในกรอบ 52 สัปดาห์")
                low52 = data["fifty_two_low"]
                high52 = data["fifty_two_high"]
                if outlook["range_position"] is not None:
                    st.progress(outlook["range_position"] / 100)
                    st.caption(f"52W Low: ${low52:,.2f} | Current: ${t_price:,.2f} | 52W High: ${high52:,.2f}")
                    st.write(f"ราคาปัจจุบันอยู่ที่ประมาณ **{outlook['range_position']:.1f}%** ของกรอบ 52 สัปดาห์")
                else:
                    st.info("ไม่มีข้อมูล 52-week range ที่เพียงพอ")
 
                st.subheader("1️⃣2️⃣ Bull / Base / Bear Scenario")
                bull, base, bear = st.columns(3)
                with bull:
                    st.success("**Bull Case**\n\n- Revenue/Earnings สูงกว่าคาด\n- Margin ขยายตัว\n- Market Share เพิ่ม\n- Valuation ยังได้รับ Premium")
                with base:
                    st.info("**Base Case**\n\n- Revenue โตตามอุตสาหกรรม\n- Margin ทรงตัว\n- Earnings โตปานกลาง\n- Valuation เคลื่อนไหวใกล้ระดับเดิม")
                with bear:
                    st.error("**Bear Case**\n\n- Revenue/Earnings ชะลอตัว\n- Margin หดตัว\n- คู่แข่งกดดัน\n- Valuation ถูก De-rate")
 
                st.subheader("1️⃣3️⃣ Final Investment Verdict")
                if t_score >= 70:
                    st.success(
                        f"✅ STRONG BUY / ACCUMULATE — {t_ticker} ได้ Quant Score {t_score}/100 และ Momentum เป็นบวก "
                        "เหมาะกับการพิจารณาทยอยสะสม โดยต้องตรวจสอบ Valuation และงบล่าสุดประกอบ"
                    )
                elif t_score >= 45:
                    st.warning(
                        f"🟡 HOLD / WAIT & SEE — {t_ticker} ได้ Quant Score {t_score}/100 "
                        "สัญญาณยังไม่แข็งแรงพอสำหรับการเพิ่มน้ำหนักแบบ aggressive"
                    )
                else:
                    st.error(
                        f"🔻 AVOID / REDUCE RISK — {t_ticker} ได้ Quant Score {t_score}/100 "
                        "โครงสร้าง Quant ยังอ่อนแอ ควรรอสัญญาณ Trend ฟื้นตัว"
                    )
 
                st.subheader("1️⃣4️⃣ สิ่งที่ควรติดตามในงบไตรมาสถัดไป")
                for item in outlook["watch"]:
                    st.markdown(f"- {item}")
                st.markdown("- การเปลี่ยนแปลงของ Guidance และประมาณการ EPS")
                st.markdown("- Free Cash Flow และระดับหนี้")
                st.markdown("- การตอบสนองของราคาหุ้นต่อผลประกอบการ")
 
                st.subheader(f"1️⃣5️⃣ ราคาย้อนหลัง 6 เดือน — {t_ticker}")
                try:
                    hist_chart = robust_download(t_ticker, period="6mo", progress=False, auto_adjust=True)
                    if not hist_chart.empty:
                        close_c = hist_chart["Close"].iloc[:, 0] if isinstance(hist_chart.columns, pd.MultiIndex) else hist_chart["Close"]
                        sma_line = close_c.rolling(50).mean()
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=close_c.index, y=close_c.values, name="Close", line=dict(color=heading_color, width=2)))
                        fig.add_trace(go.Scatter(x=sma_line.index, y=sma_line.values, name="SMA 50", line=dict(color="#f0883e", width=1.5, dash="dot")))
                        fig.update_layout(
                            template=plot_template, height=380, margin=dict(l=10, r=10, t=30, b=10),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            legend=dict(orientation="h", y=1.1),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"ไม่สามารถสร้างกราฟได้: {e}")
 
                st.caption(
                    "หมายเหตุ: รายงานนี้ใช้ข้อมูลจาก Yahoo Finance และโมเดล Quant "
                    "ข้อมูลตลาดและตัวชี้วัดสามารถเปลี่ยนแปลงได้ ไม่ควรใช้เป็นคำแนะนำการลงทุนเพียงแหล่งเดียว"
                )
 
 
# =========================================================
# TAB 3 — PORTFOLIO
# =========================================================
 
with tab3:
    st.subheader("💼 พอร์ตการลงทุนของคุณ")
    st.write("กรอกข้อมูลพอร์ต หรืออัปโหลดใบเสร็จ เพื่อให้ AI อ่านข้อมูลหุ้น")
 
    with st.expander("🧾 AI OCR — อ่านใบเสร็จซื้อขายหุ้น", expanded=False):
        uploaded_receipt = st.file_uploader("เลือกรูปภาพ", type=["png", "jpg", "jpeg"])
        ai_key_receipt = resolve_api_key("OPENAI_API_KEY", "🔑 OpenAI API Key สำหรับ OCR", "receipt_key")
 
        if uploaded_receipt is not None:
            image = Image.open(uploaded_receipt)
            st.image(image, caption="ใบเสร็จ", width=300)
 
            if st.button("🔍 สแกนใบเสร็จ"):
                if not ai_key_receipt:
                    st.error("กรุณาใส่ OpenAI API Key")
                elif openai is None:
                    st.error("ไม่พบไลบรารี openai ในสภาพแวดล้อมนี้")
                else:
                    with st.spinner("AI กำลังอ่านใบเสร็จ..."):
                        try:
                            buffered = io.BytesIO()
                            image.convert("RGB").save(buffered, format="JPEG")
                            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
 
                            client = get_openai_client(ai_key_receipt)
                            response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {
                                        "role": "system",
                                        "content": (
                                            "อ่านใบเสร็จซื้อขายหุ้น และตอบ JSON เท่านั้น\n\n"
                                            '{"Ticker":"AAPL","Shares":10,"Buy Price (USD)":175.5}'
                                        ),
                                    },
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": "อ่าน Ticker จำนวนหุ้น และราคาซื้อ"},
                                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}},
                                        ],
                                    },
                                ],
                                max_tokens=150,
                            )
 
                            res_text = response.choices[0].message.content.strip()
                            if res_text.startswith("```json"):
                                res_text = res_text[7:-3].strip()
                            elif res_text.startswith("```"):
                                res_text = res_text[3:-3].strip()
 
                            extracted = json.loads(res_text)
                            new_ticker = str(extracted.get("Ticker", "")).upper()
                            new_shares = float(extracted.get("Shares", 0))
                            new_price = float(extracted.get("Buy Price (USD)", 0))
 
                            if new_ticker and new_shares > 0 and new_price > 0:
                                if "portfolio_input" not in st.session_state:
                                    st.session_state["portfolio_input"] = load_portfolio_from_disk()
 
                                new_row = pd.DataFrame(
                                    [{"Ticker": new_ticker, "Shares": new_shares, "Buy Price (USD)": new_price, "Asset Class": "Growth"}]
                                )
                                st.session_state["portfolio_input"] = pd.concat(
                                    [st.session_state["portfolio_input"], new_row], ignore_index=True
                                )
                                save_portfolio_to_disk(st.session_state["portfolio_input"])
                                st.success(f"เพิ่ม {new_ticker} เรียบร้อยแล้ว")
                                st.rerun()
                            else:
                                st.error("AI อ่านข้อมูลไม่ครบ")
 
                        except Exception as e:
                            st.error(f"เกิดข้อผิดพลาด: {e}")
 
    if "portfolio_input" not in st.session_state:
        st.session_state["portfolio_input"] = load_portfolio_from_disk()
 
    edited_portfolio = st.data_editor(
        st.session_state["portfolio_input"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Asset Class": st.column_config.SelectboxColumn("Asset Class", options=["Growth", "Defensive"], default="Growth")
        },
    )
 
    try:
        changed = not edited_portfolio.equals(st.session_state["portfolio_input"])
    except Exception:
        changed = True
 
    if changed:
        st.session_state["portfolio_input"] = edited_portfolio
        save_portfolio_to_disk(edited_portfolio)
 
    if st.button("♻️ ล้างพอร์ตเป็นค่าเริ่มต้น"):
        st.session_state["portfolio_input"] = DEFAULT_PORTFOLIO.copy()
        save_portfolio_to_disk(DEFAULT_PORTFOLIO)
        st.rerun()
 
    # -----------------------------------------------------
    # CALCULATE PORTFOLIO
    # -----------------------------------------------------
 
    df_res = pd.DataFrame()
 
    if not edited_portfolio.empty:
        clean_rows = []
        for _, row in edited_portfolio.iterrows():
            ticker = str(row["Ticker"]).strip().upper()
            shares = float(row["Shares"]) if pd.notna(row["Shares"]) else 0
            buy_price = float(row["Buy Price (USD)"]) if pd.notna(row["Buy Price (USD)"]) else 0
            asset_class = str(row.get("Asset Class", "Growth"))
            if ticker and shares > 0 and buy_price > 0:
                clean_rows.append(
                    {"Ticker": ticker, "Shares": shares, "Buy Price (USD)": buy_price, "Asset Class": asset_class}
                )
 
        if clean_rows:
            df_clean = pd.DataFrame(clean_rows)
            df_clean["Cost Basis ($)"] = df_clean["Shares"] * df_clean["Buy Price (USD)"]
            agg = (
                df_clean.groupby("Ticker")
                .agg(Shares=("Shares", "sum"), CostBasis=("Cost Basis ($)", "sum"), AssetClass=("Asset Class", "first"))
                .reset_index()
            )
            agg["Buy Price (USD)"] = agg["CostBasis"] / agg["Shares"]
        else:
            agg = pd.DataFrame()
 
        portfolio_results = []
        for _, row in agg.iterrows():
            ticker = row["Ticker"]
            shares = float(row["Shares"])
            buy_price = float(row["Buy Price (USD)"])
            asset_class = row["AssetClass"]
 
            current_price, signal, score, trend = buy_price, "ถือประเมินสถานะ", 50, "ไม่ทราบ"
 
            match_row = df[df["Ticker"] == ticker]
            if not match_row.empty:
                current_price = float(match_row.iloc[0]["Price (USD)"])
                signal = match_row.iloc[0]["Simons Signal"]
                score = int(match_row.iloc[0]["Score"])
                trend = match_row.iloc[0]["Trend"]
            else:
                try:
                    hist = robust_download(ticker, period="6mo", progress=False, auto_adjust=True)
                    if not hist.empty:
                        close_series = hist["Close"].iloc[:, 0] if isinstance(hist.columns, pd.MultiIndex) else hist["Close"]
                        current_price = float(close_series.iloc[-1])
                except Exception:
                    current_price = buy_price
 
            invested_value = shares * buy_price
            current_value = shares * current_price
            profit_loss = current_value - invested_value
            profit_loss_pct = (current_price - buy_price) / buy_price * 100 if buy_price > 0 else 0
 
            portfolio_results.append(
                {
                    "Ticker": ticker, "Shares": shares, "Buy Price (USD)": buy_price,
                    "Current Price (USD)": current_price, "Invested Value ($)": invested_value,
                    "Current Value ($)": current_value, "Profit/Loss ($)": profit_loss,
                    "Profit/Loss (%)": profit_loss_pct, "Simons Signal": signal, "Score": score,
                    "Trend": trend, "Asset Class": asset_class,
                }
            )
 
        if portfolio_results:
            df_res = pd.DataFrame(portfolio_results)
            total_invested = df_res["Invested Value ($)"].sum()
            total_current = df_res["Current Value ($)"].sum()
            total_pl = total_current - total_invested
            total_pl_pct = total_pl / total_invested * 100 if total_invested > 0 else 0
            df_res["Weight in Portfolio (%)"] = (
                df_res["Current Value ($)"] / total_current * 100 if total_current > 0 else 0
            )
 
            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("💵 เงินลงทุน", f"${total_invested:,.2f}")
            m2.metric("📊 มูลค่าปัจจุบัน", f"${total_current:,.2f}")
            m3.metric("กำไร/ขาดทุน", f"${total_pl:,.2f}", f"{total_pl_pct:.2f}%")
 
            # =============================================
            # ACTION SUMMARY
            # =============================================
 
            st.subheader("⚡ Quantitative Action Summary")
            buy_candidates = df_res[df_res["Score"] >= 70]
            sell_candidates = df_res[df_res["Score"] < 45]
 
            act1, act2 = st.columns(2)
            with act1:
                st.markdown("### 🟢 ซื้อเพิ่ม")
                if not buy_candidates.empty:
                    for _, x in buy_candidates.iterrows():
                        st.success(f"**{x['Ticker']}** — Score: {x['Score']}/100 — {x['Simons Signal']}")
                else:
                    st.info("ยังไม่มีหุ้นที่เข้าเกณฑ์ซื้อเพิ่ม")
            with act2:
                st.markdown("### 🔴 ลดความเสี่ยง")
                if not sell_candidates.empty:
                    for _, x in sell_candidates.iterrows():
                        st.error(f"**{x['Ticker']}** — Score: {x['Score']}/100 — {x['Simons Signal']}")
                else:
                    st.success("ไม่มีหุ้นที่เข้าเกณฑ์ขาย")
 
            # =============================================
            # NEW: RISK & DIVERSIFICATION ANALYTICS
            # =============================================
 
            st.markdown("---")
            st.subheader("🧠 Risk & Diversification Analytics")
            st.caption("คำนวณจากผลตอบแทนรายวันจริงย้อนหลัง ~1 ปี เทียบกับ SPY เป็นเบนช์มาร์ก")
 
            with st.spinner("กำลังคำนวณความเสี่ยงของพอร์ต..."):
                risk_metrics = compute_portfolio_risk_metrics(df_res)
 
            if risk_metrics is None:
                st.info("ข้อมูลราคาย้อนหลังไม่เพียงพอสำหรับคำนวณ Risk Analytics (ลองอีกครั้งหรือเพิ่มหุ้นที่มีประวัติราคายาวขึ้น)")
            else:
                rk1, rk2, rk3, rk4, rk5 = st.columns(5)
                rk1.metric("Ann. Volatility", f"{risk_metrics['ann_vol']:.1f}%")
                rk2.metric("Ann. Return", f"{risk_metrics['ann_return']:.1f}%")
                rk3.metric("Sharpe Ratio", f"{risk_metrics['sharpe']:.2f}" if risk_metrics["sharpe"] is not None else "N/A")
                rk4.metric("Max Drawdown", f"{risk_metrics['max_drawdown']:.1f}%")
                rk5.metric(f"Beta vs {BENCHMARK_TICKER}", f"{risk_metrics['beta']:.2f}" if risk_metrics["beta"] is not None else "N/A")
 
                st.markdown(f"**Concentration (HHI):** {risk_metrics['hhi']:.0f} / 10,000 — {hhi_read_out(risk_metrics['hhi'])}")
 
                rc1, rc2 = st.columns(2)
                with rc1:
                    if risk_metrics["benchmark_cum_norm"] is not None:
                        perf_fig = go.Figure()
                        perf_fig.add_trace(go.Scatter(
                            x=risk_metrics["portfolio_cum_norm"].index, y=risk_metrics["portfolio_cum_norm"].values,
                            name="พอร์ตของคุณ", line=dict(color=up_color, width=2),
                        ))
                        perf_fig.add_trace(go.Scatter(
                            x=risk_metrics["benchmark_cum_norm"].index, y=risk_metrics["benchmark_cum_norm"].values,
                            name=BENCHMARK_TICKER, line=dict(color="#8b949e", width=2, dash="dot"),
                        ))
                        perf_fig.update_layout(
                            title="ผลตอบแทนสะสม (Rebased = 100)", template=plot_template, height=340,
                            margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.15),
                        )
                        st.plotly_chart(perf_fig, use_container_width=True)
                with rc2:
                    corr = risk_metrics["correlation_matrix"]
                    if corr.shape[0] > 1:
                        heat_fig = px.imshow(
                            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                            title="Correlation ระหว่างหุ้นในพอร์ต",
                        )
                        heat_fig.update_layout(template=plot_template, height=340, margin=dict(l=10, r=10, t=40, b=10))
                        st.plotly_chart(heat_fig, use_container_width=True)
                    else:
                        st.info("มีหุ้นเพียงตัวเดียวในพอร์ต — ไม่สามารถคำนวณ Correlation ได้")
 
                st.caption(
                    "Sharpe Ratio คำนวณด้วยอัตราดอกเบี้ยปลอดความเสี่ยงสมมติที่ "
                    f"{RISK_FREE_RATE*100:.1f}% ต่อปี ตัวเลขทั้งหมดเป็นข้อมูลย้อนหลัง ไม่ใช่การรับประกันผลตอบแทนในอนาคต"
                )
 
            # =============================================
            # SECTOR ALLOCATION (NEW)
            # =============================================
 
            st.markdown("---")
            st.subheader("🥧 Sector Allocation")
            with st.spinner("กำลังดึงข้อมูล Sector ของแต่ละหุ้น..."):
                sector_map = {t: get_sector_only(t) for t in df_res["Ticker"].unique()}
            df_res["Sector"] = df_res["Ticker"].map(sector_map)
            sector_alloc = df_res.groupby("Sector")["Current Value ($)"].sum().reset_index()
            if not sector_alloc.empty:
                donut = px.pie(
                    sector_alloc, values="Current Value ($)", names="Sector", hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                donut.update_layout(template=plot_template, height=360, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(donut, use_container_width=True)
 
            # =============================================
            # TABLE
            # =============================================
 
            st.subheader("📋 สรุปพอร์ต")
            st.dataframe(
                df_res[
                    ["Ticker", "Shares", "Buy Price (USD)", "Current Price (USD)", "Profit/Loss ($)",
                     "Profit/Loss (%)", "Weight in Portfolio (%)", "Asset Class", "Sector", "Simons Signal", "Score"]
                ],
                column_config={
                    "Buy Price (USD)": st.column_config.NumberColumn(format="$%.2f"),
                    "Current Price (USD)": st.column_config.NumberColumn(format="$%.2f"),
                    "Profit/Loss ($)": st.column_config.NumberColumn(format="$%.2f"),
                    "Profit/Loss (%)": st.column_config.NumberColumn(format="%.2f%%"),
                    "Weight in Portfolio (%)": st.column_config.NumberColumn(format="%.1f%%"),
                    "Score": st.column_config.NumberColumn(format="%d คะแนน"),
                },
                use_container_width=True,
                hide_index=True,
            )
 
            st.download_button(
                "⬇️ ดาวน์โหลด Portfolio CSV",
                data=df_res.to_csv(index=False).encode("utf-8-sig"),
                file_name="portfolio_report.csv",
                mime="text/csv",
            )
 
 
# =========================================================
# TAB 4 — AI CHAT ADVISOR
# =========================================================
 
with tab4:
    st.subheader("🤖 AI Portfolio Advisor")
 
    openai_api_key = resolve_api_key("OPENAI_API_KEY", "🔑 OpenAI API Key สำหรับ AI Chat", "chat_key")
 
    if openai_api_key and openai is not None:
        portfolio_context = "ไม่มีข้อมูลพอร์ต"
        if "df_res" in locals() and not df_res.empty:
            lines = []
            for _, r in df_res.iterrows():
                lines.append(
                    f"หุ้น {r['Ticker']}: ถือ {r['Shares']} หุ้น, ต้นทุน ${r['Buy Price (USD)']:.2f}, "
                    f"ราคาปัจจุบัน ${r['Current Price (USD)']:.2f}, P/L {r['Profit/Loss (%)']:.2f}%, "
                    f"Weight {r['Weight in Portfolio (%)']:.1f}%, Asset Class {r['Asset Class']}, "
                    f"Quant Score {r['Score']}/100, Signal {r['Simons Signal']}"
                )
            portfolio_context = "\n".join(lines)
 
        system_instruction = f"""
        คุณคือผู้ช่วยวิเคราะห์การลงทุนเชิง Quantitative
 
        ใช้ข้อมูล:
        - Statistics
        - Momentum
        - Risk
        - Valuation
        - Portfolio Weight
        - Business Fundamentals
 
        ห้ามอ้างว่าคุณเป็น Jim Simons ตัวจริง
        คุณให้ข้อมูลประกอบการตัดสินใจเท่านั้น ไม่ใช่คำแนะนำการลงทุนที่รับประกันผลตอบแทน
 
        เน้น:
        1. ความน่าจะเป็น
        2. Risk Management
        3. Position Sizing
        4. Diversification
        5. Long-term Business Quality
 
        ข้อมูลพอร์ต:
 
        {portfolio_context}
 
        ตอบเป็นภาษาไทย
        """
 
        if "messages" not in st.session_state or st.session_state.get("last_system_prompt") != system_instruction:
            st.session_state.messages = [{"role": "system", "content": system_instruction}]
            st.session_state["last_system_prompt"] = system_instruction
 
        for message in st.session_state.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
 
        prompt = st.chat_input("ถาม AI เกี่ยวกับหุ้นหรือพอร์ต...")
 
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
 
            with st.chat_message("assistant"):
                with st.spinner("AI กำลังวิเคราะห์..."):
                    try:
                        client = get_openai_client(openai_api_key)
                        response = client.chat.completions.create(model="gpt-4o-mini", messages=st.session_state.messages)
                        reply = response.choices[0].message.content
                        st.markdown(reply)
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาด: {e}")
    elif openai is None:
        st.error("ไม่พบไลบรารี openai ในสภาพแวดล้อมนี้ — โปรดติดตั้งด้วย `pip install openai`")
    else:
        st.info("👈 กรุณาใส่ OpenAI API Key ที่ Sidebar (หรือกำหนดใน st.secrets['OPENAI_API_KEY']) เพื่อเปิด AI Chat")