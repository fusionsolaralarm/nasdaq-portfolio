import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import openai
import json
import os
from PIL import Image
import base64
import io


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SIMON Quant & Portfolio AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# THEME
# =========================================================

st.sidebar.header(" ตั้งค่าการแสดงผล")

theme_mode = st.sidebar.selectbox(
    "เลือกธีมหน้าจอ",
    [
        "โหมดกลางคืน (Dark Mode)",
        "โหมดกลางวัน (Light Mode)"
    ]
)

if theme_mode == "โหมดกลางคืน (Dark Mode)":
    bg_color = "#0e1117"
    card_bg = "#161b22"
    text_color = "#c9d1d9"
    heading_color = "#58a6ff"
    border_color = "#30363d"
else:
    bg_color = "#ffffff"
    card_bg = "#f6f8fa"
    text_color = "#24292e"
    heading_color = "#0366d6"
    border_color = "#e1e4e8"


# =========================================================
# CSS
# =========================================================

st.markdown(
    f"""
    <style>

    .main {{
        background-color: {bg_color};
    }}

    .stock-card {{
        background-color: {card_bg};
        padding: 20px;
        border-radius: 12px;
        border: 1px solid {border_color};
        margin-bottom: 15px;
        color: {text_color};
    }}

    .article-box {{
        background-color: {card_bg};
        padding: 20px;
        border-radius: 12px;
        border: 1px solid {border_color};
        margin-top: 15px;
        margin-bottom: 15px;
        color: {text_color};
        line-height: 1.7;
    }}

    .small-card {{
        background-color: {card_bg};
        padding: 15px;
        border-radius: 10px;
        border: 1px solid {border_color};
        margin-bottom: 10px;
        color: {text_color};
    }}

    .verdict-box-yes {{
        background-color: rgba(35, 134, 54, 0.15);
        border: 1px solid #238636;
        padding: 20px;
        border-radius: 10px;
        margin-top: 15px;
        color: {text_color};
    }}

    .verdict-box-no {{
        background-color: rgba(218, 54, 51, 0.15);
        border: 1px solid #da3633;
        padding: 20px;
        border-radius: 10px;
        margin-top: 15px;
        color: {text_color};
    }}

    .action-box-buy {{
        background-color: rgba(35, 134, 54, 0.15);
        border: 1px solid #238636;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: {text_color};
    }}

    .action-box-sell {{
        background-color: rgba(218, 54, 51, 0.15);
        border: 1px solid #da3633;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: {text_color};
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FILE
# =========================================================

PORTFOLIO_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "portfolio_data.json"
)


# =========================================================
# PORTFOLIO FUNCTIONS
# =========================================================

def load_portfolio_from_disk():

    if os.path.exists(PORTFOLIO_FILE):

        try:

            with open(
                PORTFOLIO_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                records = json.load(f)

            loaded = pd.DataFrame(records)

            required = {
                "Ticker",
                "Shares",
                "Buy Price (USD)"
            }

            if not loaded.empty and required.issubset(
                loaded.columns
            ):

                if "Asset Class" not in loaded.columns:
                    loaded["Asset Class"] = "Growth"

                return loaded[
                    [
                        "Ticker",
                        "Shares",
                        "Buy Price (USD)",
                        "Asset Class"
                    ]
                ]

        except Exception:
            pass

    return pd.DataFrame(
        [
            {
                "Ticker": "AAPL",
                "Shares": 10.0,
                "Buy Price (USD)": 170.0,
                "Asset Class": "Growth"
            },
            {
                "Ticker": "TSLA",
                "Shares": 5.0,
                "Buy Price (USD)": 220.0,
                "Asset Class": "Growth"
            }
        ]
    )


def save_portfolio_to_disk(df):

    try:

        df.to_json(
            PORTFOLIO_FILE,
            orient="records",
            force_ascii=False
        )

    except Exception as e:

        st.sidebar.error(
            f"บันทึกพอร์ตลงดิสก์ไม่สำเร็จ: {e}"
        )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    f"""
    <h1 style="
        text-align:center;
        color:{heading_color};
    ">
        SIMON QUANT & PORTFOLIO AI ADVISOR
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <p style="
        text-align:center;
        color:{text_color};
    ">
        ระบบสแกนหุ้นสหรัฐฯ วิเคราะห์หุ้นรายตัว
        และวิเคราะห์พอร์ตด้วยข้อมูลเชิงปริมาณ
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")


# =========================================================
# SYMBOL LIST
# =========================================================

@st.cache_data
def get_curated_symbols():

    return [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
        "NFLX",
        "AMD",
        "INTC",
        "CRM",
        "ADBE",
        "ORCL",
        "IBM",
        "NOW",
        "SNOW",
        "PLTR",
        "DDOG",
        "CRWD",
        "JPM",
        "BAC",
        "WFC",
        "GS",
        "MS",
        "V",
        "MA",
        "AXP",
        "PYPL",
        "WMT",
        "COST",
        "TGT",
        "HD",
        "NKE",
        "SBUX",
        "MCD",
        "KO",
        "PEP",
        "DIS",
        "UNH",
        "JNJ",
        "PFE",
        "ABBV",
        "LLY",
        "MRK",
        "AMGN",
        "TMO",
        "XOM",
        "CVX",
        "CAT",
        "BA",
        "GE",
        "HON",
        "UPS",
        "LMT",
        "ASTS",
        "LUNR",
        "IONQ",
        "SOUN",
        "RGTI",
        "QBTS",
        "ACHR",
        "JOBY",
        "RIVN",
        "HOOD",
        "ARM",
        "SMCI",
        "RDDT",
        "DJT",
        "BBAI",
        "HLGN",
        "CLSK",
        "RIOT",
        "MARA",
        "HUT",
        "BITF",
        "OPEN",
        "LMND",
        "AVGO",
        "ACN",
        "CSCO",
        "TXN",
        "QCOM",
        "AMAT",
        "INTU",
        "BKNG",
        "ISRG"
    ]


@st.cache_data(ttl=86400)
def get_full_us_market_symbols():

    try:

        nasdaq_url = (
            "https://www.nasdaqtrader.com/"
            "dynamic/SymDir/nasdaqlisted.txt"
        )

        other_url = (
            "https://www.nasdaqtrader.com/"
            "dynamic/SymDir/otherlisted.txt"
        )

        nasdaq_df = pd.read_csv(
            nasdaq_url,
            sep="|"
        )

        nasdaq_df = nasdaq_df.iloc[:-1]

        nasdaq_df = nasdaq_df[
            nasdaq_df["Test Issue"] == "N"
        ]

        nasdaq_df = nasdaq_df.rename(
            columns={
                "Security Name": "Name"
            }
        )[
            [
                "Symbol",
                "Name",
                "ETF"
            ]
        ]

        other_df = pd.read_csv(
            other_url,
            sep="|"
        )

        other_df = other_df.iloc[:-1]

        other_df = other_df[
            other_df["Test Issue"] == "N"
        ]

        other_df = other_df.rename(
            columns={
                "ACT Symbol": "Symbol",
                "Security Name": "Name"
            }
        )[
            [
                "Symbol",
                "Name",
                "ETF"
            ]
        ]

        combined = pd.concat(
            [
                nasdaq_df,
                other_df
            ],
            ignore_index=True
        )

        combined["ETF"] = (
            combined["ETF"]
            .astype(str)
            .str.upper()
            .eq("Y")
        )

        combined["Symbol"] = (
            combined["Symbol"]
            .astype(str)
            .str.strip()
        )

        combined = combined[
            combined["Symbol"].str.match(
                r"^[A-Za-z.]+$",
                na=False
            )
        ]

        combined["Symbol"] = (
            combined["Symbol"]
            .str.replace(
                ".",
                "-",
                regex=False
            )
        )

        combined = (
            combined
            .drop_duplicates(
                subset="Symbol"
            )
            .sort_values("Symbol")
            .reset_index(drop=True)
        )

        if combined.empty:
            raise ValueError(
                "รายชื่อหุ้นว่างเปล่า"
            )

        return combined

    except Exception as e:

        st.sidebar.warning(
            "ดึงรายชื่อหุ้นทั้งหมดไม่สำเร็จ "
            f"({e}) ใช้รายชื่อคัดสรรแทน"
        )

        curated = get_curated_symbols()

        return pd.DataFrame(
            {
                "Symbol": curated,
                "Name": curated,
                "ETF": False
            }
        )


# =========================================================
# QUANT MODEL
# =========================================================

def compute_simons_row(
    ticker,
    close_series
):

    if close_series is None:
        return None

    close_series = close_series.dropna()

    if len(close_series) < 30:
        return None

    current_p = float(
        close_series.iloc[-1]
    )

    price_1m_ago = float(
        close_series.iloc[
            -min(20, len(close_series))
        ]
    )

    price_1w_ago = float(
        close_series.iloc[
            -min(5, len(close_series))
        ]
    )

    return_1m = (
        (current_p - price_1m_ago)
        / price_1m_ago
        * 100
    )

    return_1w = (
        (current_p - price_1w_ago)
        / price_1w_ago
        * 100
    )

    sma_50 = float(
        close_series
        .rolling(
            min(30, len(close_series))
        )
        .mean()
        .iloc[-1]
    )

    volatility = float(
        close_series
        .pct_change()
        .std()
        * np.sqrt(252)
        * 100
    )

    score = 50

    if current_p > sma_50:
        score += 25
    else:
        score -= 20

    if return_1m > 0:
        score += 20
    else:
        score -= 15

    if volatility < 40:
        score += 15
    else:
        score -= 10

    score = max(
        0,
        min(100, score)
    )

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
        "SMA 50": sma_50
    }


# =========================================================
# FETCH MARKET DATA
# =========================================================

@st.cache_data(ttl=1800)
def fetch_stock_data_and_simons_logic(
    ticker_list
):

    data_list = []

    for ticker in ticker_list:

        try:

            hist = yf.download(
                ticker,
                period="6mo",
                progress=False,
                auto_adjust=True
            )

            if hist.empty:
                continue

            if isinstance(
                hist.columns,
                pd.MultiIndex
            ):

                close_series = hist[
                    "Close"
                ].iloc[:, 0]

            else:

                close_series = hist["Close"]

            row = compute_simons_row(
                ticker,
                close_series
            )

            if row:
                data_list.append(row)

        except Exception:
            continue

    return pd.DataFrame(data_list)


def fetch_stock_data_batched(
    ticker_list,
    chunk_size=50
):

    all_rows = []

    total = len(ticker_list)

    progress_bar = st.progress(
        0.0,
        text="กำลังสแกนตลาด..."
    )

    for i in range(
        0,
        total,
        chunk_size
    ):

        chunk = ticker_list[
            i:i + chunk_size
        ]

        try:

            data = yf.download(
                chunk,
                period="6mo",
                group_by="ticker",
                threads=True,
                progress=False,
                auto_adjust=True
            )

        except Exception:
            data = None

        if (
            data is not None
            and not data.empty
        ):

            for ticker in chunk:

                try:

                    if (
                        len(chunk) == 1
                        or not isinstance(
                            data.columns,
                            pd.MultiIndex
                        )
                    ):

                        close_series = data[
                            "Close"
                        ]

                    else:

                        if (
                            ticker
                            not in
                            data.columns.get_level_values(
                                0
                            )
                        ):
                            continue

                        close_series = data[
                            ticker
                        ]["Close"]

                    row = compute_simons_row(
                        ticker,
                        close_series
                    )

                    if row:
                        all_rows.append(row)

                except Exception:
                    continue

        done = min(
            i + chunk_size,
            total
        )

        progress_bar.progress(
            done / total,
            text=(
                f"สแกนแล้ว {done:,}/{total:,} "
                f"ตัว | สำเร็จ {len(all_rows):,} ตัว"
            )
        )

    progress_bar.empty()

    return pd.DataFrame(all_rows)


# =========================================================
# COMPANY INFO
# =========================================================

@st.cache_data(ttl=1800)
def get_company_business_info(
    ticker_symbol
):

    try:

        ticker = yf.Ticker(
            ticker_symbol
        )

        info = ticker.info

        return (
            info.get(
                "longName",
                ticker_symbol
            ),
            info.get(
                "sector",
                "N/A"
            ),
            info.get(
                "industry",
                "N/A"
            ),
            info.get(
                "longBusinessSummary",
                "ไม่มีข้อมูลสรุปธุรกิจ"
            )
        )

    except Exception:

        return (
            ticker_symbol,
            "N/A",
            "N/A",
            "ไม่สามารถดึงข้อมูลธุรกิจได้"
        )


# =========================================================
# FORMATTING HELPERS
# =========================================================
def format_money(value):
    """แปลงตัวเลขทางการเงินเป็นรูปแบบอ่านง่าย"""
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
    if value is None:
        return "N/A"
    return f"{float(value) * 100:+.2f}%"


def metric_or_na(value, suffix=""):
    return "N/A" if value is None else f"{float(value):.2f}{suffix}"


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
    """เลือกคำศัพท์ธุรกิจที่พบในคำอธิบายบริษัทเพื่อฝึกภาษา"""
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
def get_detailed_stock_analysis(
    ticker
):

    try:

        t = yf.Ticker(ticker)

        info = t.info

        def safe_num(key):

            value = info.get(key)

            try:

                return (
                    float(value)
                    if value is not None
                    else None
                )

            except Exception:

                return None

        data = {

            "long_name":
                info.get(
                    "longName",
                    ticker
                ),

            "sector":
                info.get(
                    "sector",
                    "N/A"
                ),

            "industry":
                info.get(
                    "industry",
                    "N/A"
                ),

            "summary":
                info.get(
                    "longBusinessSummary",
                    "ไม่มีข้อมูลธุรกิจ"
                ),

            "market_cap":
                safe_num("marketCap"),

            "enterprise_value":
                safe_num("enterpriseValue"),

            "revenue":
                safe_num("totalRevenue"),

            "revenue_growth":
                safe_num("revenueGrowth"),

            "gross_margin":
                safe_num("grossMargins"),

            "operating_margin":
                safe_num("operatingMargins"),

            "profit_margin":
                safe_num("profitMargins"),

            "eps":
                safe_num("trailingEps"),

            "forward_eps":
                safe_num("forwardEps"),

            "pe":
                safe_num("trailingPE"),

            "forward_pe":
                safe_num("forwardPE"),

            "peg":
                safe_num("pegRatio"),

            "price_to_sales":
                safe_num(
                    "priceToSalesTrailing12Months"
                ),

            "earnings_growth":
                safe_num(
                    "earningsGrowth"
                ),

            "earnings_quarterly_growth":
                safe_num(
                    "earningsQuarterlyGrowth"
                ),

            "free_cash_flow":
                safe_num(
                    "freeCashflow"
                ),

            "operating_cash_flow":
                safe_num(
                    "operatingCashflow"
                ),

            "debt_to_equity":
                safe_num(
                    "debtToEquity"
                ),

            "current_ratio":
                safe_num(
                    "currentRatio"
                ),

            "return_on_equity":
                safe_num(
                    "returnOnEquity"
                ),

            "return_on_assets":
                safe_num(
                    "returnOnAssets"
                ),

            "beta":
                safe_num("beta"),

            "dividend_yield":
                safe_num(
                    "dividendYield"
                ),

            "target_mean":
                safe_num(
                    "targetMeanPrice"
                ),

            "target_high":
                safe_num(
                    "targetHighPrice"
                ),

            "target_low":
                safe_num(
                    "targetLowPrice"
                ),

            "fifty_two_high":
                safe_num(
                    "fiftyTwoWeekHigh"
                ),

            "fifty_two_low":
                safe_num(
                    "fiftyTwoWeekLow"
                ),

            "employees":
                safe_num(
                    "fullTimeEmployees"
                )
        }

        # -------------------------------------------------
        # GROWTH SCORE
        # -------------------------------------------------

        growth_score = 0

        if data["revenue_growth"] is not None:

            if data["revenue_growth"] > 0.20:
                growth_score += 3

            elif data["revenue_growth"] > 0.08:
                growth_score += 2

            elif data["revenue_growth"] > 0:
                growth_score += 1

            else:
                growth_score -= 1

        if data["earnings_growth"] is not None:

            if data["earnings_growth"] > 0.20:
                growth_score += 3

            elif data["earnings_growth"] > 0.05:
                growth_score += 2

            elif data["earnings_growth"] > 0:
                growth_score += 1

            else:
                growth_score -= 1

        # -------------------------------------------------
        # QUALITY SCORE
        # -------------------------------------------------

        quality_score = 0

        if data["profit_margin"] is not None:

            if data["profit_margin"] > 0.20:
                quality_score += 3

            elif data["profit_margin"] > 0.10:
                quality_score += 2

            elif data["profit_margin"] > 0:
                quality_score += 1

        if data["return_on_equity"] is not None:

            if data["return_on_equity"] > 0.20:
                quality_score += 3

            elif data["return_on_equity"] > 0.10:
                quality_score += 2

            elif data["return_on_equity"] > 0:
                quality_score += 1

        # -------------------------------------------------
        # DEBT SCORE
        # -------------------------------------------------

        debt_score = 0

        if data["debt_to_equity"] is not None:

            if data["debt_to_equity"] < 50:
                debt_score = 3

            elif data["debt_to_equity"] < 100:
                debt_score = 2

            elif data["debt_to_equity"] < 200:
                debt_score = 1

            else:
                debt_score = -1

        # -------------------------------------------------
        # VALUATION
        # -------------------------------------------------

        valuation_text = (
            "ไม่สามารถประเมิน Valuation ได้"
        )

        pe = (
            data["forward_pe"]
            or data["pe"]
        )

        if pe is not None:

            if pe < 15:

                valuation_text = (
                    "Valuation ค่อนข้างต่ำ "
                    "เมื่อเทียบกับหุ้น Growth ทั่วไป"
                )

            elif pe < 25:

                valuation_text = (
                    "Valuation อยู่ในระดับปานกลาง"
                )

            elif pe < 40:

                valuation_text = (
                    "Valuation ค่อนข้างสูง "
                    "สะท้อนความคาดหวังการเติบโต"
                )

            else:

                valuation_text = (
                    "Valuation สูงมาก "
                    "ราคาหุ้นต้องการการเติบโตของกำไรสูง"
                )

        # -------------------------------------------------
        # INDUSTRY OUTLOOK
        # -------------------------------------------------

        sector = str(
            data["sector"]
        ).lower()

        industry = str(
            data["industry"]
        ).lower()

        outlook = []

        catalysts = []

        risks = []

        # Technology / Semiconductor

        if (
            "technology" in sector
            or "semiconductor" in industry
            or "semiconductors" in industry
        ):

            outlook = [

                "AI Infrastructure และ Data Center",

                "Cloud Computing",

                "Enterprise AI",

                "Advanced Computing",

                "Semiconductor Demand",

                "Automation และ Software Intelligence",

                "การเพิ่มงบลงทุนของ Hyperscalers"

            ]

            catalysts = [

                "รายได้จาก AI เติบโตเร็วกว่าคาด",

                "การเปิดตัวผลิตภัณฑ์ใหม่",

                "การเพิ่ม Data Center CapEx",

                "การขยายตลาด Enterprise",

                "Margin ขยายตัวจาก Economies of Scale",

                "การเพิ่ม Market Share"

            ]

            risks = [

                "การแข่งขันทางเทคโนโลยี",

                "Semiconductor Cyclicality",

                "Supply Chain",

                "Regulation",

                "Export Restrictions",

                "Valuation สูง",

                "การชะลอตัวของ AI CapEx"

            ]

        # Healthcare

        elif "health" in sector:

            outlook = [

                "Aging Population",

                "Precision Medicine",

                "Biotechnology",

                "Medical Technology",

                "Chronic Disease Treatment",

                "AI in Healthcare"

            ]

            catalysts = [

                "FDA Approval",

                "Clinical Trial Success",

                "ยอดขายยาใหม่",

                "การขยายตลาดต่างประเทศ",

                "การเปิดตัวผลิตภัณฑ์ใหม่"

            ]

            risks = [

                "Clinical Trial Failure",

                "FDA / Regulatory Risk",

                "Patent Expiration",

                "Drug Pricing Pressure",

                "การแข่งขันจาก Generic Drugs"

            ]

        # Financial

        elif "financial" in sector:

            outlook = [

                "ทิศทางอัตราดอกเบี้ย",

                "Credit Growth",

                "Digital Banking",

                "Capital Market Activity",

                "Wealth Management",

                "AI ในระบบการเงิน"

            ]

            catalysts = [

                "Credit Growth",

                "Investment Banking Recovery",

                "Trading Revenue",

                "Net Interest Income",

                "Cost Reduction"

            ]

            risks = [

                "Credit Risk",

                "Interest Rate Risk",

                "Recession",

                "Regulatory Risk",

                "Loan Losses"

            ]

        # Consumer

        elif "consumer" in sector:

            outlook = [

                "กำลังซื้อผู้บริโภค",

                "E-Commerce",

                "Brand Strength",

                "Pricing Power",

                "International Expansion",

                "Digital Transformation"

            ]

            catalysts = [

                "ยอดขายเติบโต",

                "Same Store Sales",

                "E-Commerce Growth",

                "การขยายสาขา",

                "การเพิ่ม Market Share"

            ]

            risks = [

                "Consumer Spending Slowdown",

                "Inflation",

                "ต้นทุนสินค้า",

                "การแข่งขันด้านราคา",

                "Supply Chain"

            ]

        # Industrial

        elif "industrial" in sector:

            outlook = [

                "Infrastructure Spending",

                "Automation",

                "Industrial Digitization",

                "Energy Transition",

                "Manufacturing Investment"

            ]

            catalysts = [

                "คำสั่งซื้อใหม่",

                "Infrastructure Spending",

                "Automation Demand",

                "Margin Expansion",

                "Global Expansion"

            ]

            risks = [

                "Economic Slowdown",

                "Raw Material Cost",

                "Supply Chain",

                "Interest Rate",

                "Cyclical Demand"

            ]

        else:

            outlook = [

                "การเติบโตของตลาด",

                "การขยายฐานลูกค้า",

                "การพัฒนาผลิตภัณฑ์",

                "Digital Transformation",

                "การเพิ่มประสิทธิภาพต้นทุน",

                "การขยายตลาดต่างประเทศ"

            ]

            catalysts = [

                "Revenue Growth",

                "Earnings Growth",

                "New Products",

                "Market Expansion",

                "Margin Expansion"

            ]

            risks = [

                "เศรษฐกิจมหภาค",

                "การแข่งขัน",

                "ต้นทุน",

                "ดอกเบี้ย",

                "Regulation",

                "Demand Slowdown"

            ]

        # -------------------------------------------------
        # MEDIUM / LONG TERM
        # -------------------------------------------------

        if growth_score >= 4:

            medium_term = (
                "แนวโน้มระยะกลางเป็นบวก "
                "หากบริษัทสามารถรักษาการเติบโตของรายได้ "
                "และกำไรได้ต่อเนื่อง"
            )

        elif growth_score >= 2:

            medium_term = (
                "มีโอกาสเติบโต แต่ควรติดตาม "
                "ผลประกอบการรายไตรมาสและ Guidance "
                "อย่างใกล้ชิด"
            )

        else:

            medium_term = (
                "ควรรอดูการฟื้นตัวของรายได้และกำไร "
                "ก่อนเพิ่มน้ำหนักการลงทุน"
            )

        if quality_score >= 4:

            long_term = (
                "คุณภาพธุรกิจอยู่ในระดับน่าสนใจ "
                "หากบริษัทสามารถรักษา Margin, ROE "
                "และกระแสเงินสดได้ในระยะยาว"
            )

        else:

            long_term = (
                "การลงทุนระยะยาวควรติดตาม "
                "ความสามารถในการสร้างกำไร "
                "และ Free Cash Flow อย่างต่อเนื่อง"
            )

        return {

            "data": data,

            "growth_score":
                growth_score,

            "quality_score":
                quality_score,

            "debt_score":
                debt_score,

            "valuation_text":
                valuation_text,

            "outlook":
                outlook,

            "catalysts":
                catalysts,

            "risks":
                risks,

            "medium_term":
                medium_term,

            "long_term":
                long_term,

        }

    except Exception as e:

        return {
            "error": str(e)
        }


# =========================================================
# SIDEBAR SCANNER
# =========================================================

st.sidebar.header(
    " ตั้งค่าแหล่งข้อมูล & สแกนหุ้น"
)

scan_mode = st.sidebar.radio(
    "เลือกแหล่งข้อมูลหุ้น",
    [
        "รายชื่อคัดสรร (~85 ตัว, เร็ว)",
        "ทั้งตลาดหุ้นสหรัฐฯ (NASDAQ + NYSE + AMEX)"
    ]
)


# =========================================================
# LOAD SCANNER DATA
# =========================================================

if scan_mode.startswith(
    "รายชื่อคัดสรร"
):

    symbols = get_curated_symbols()

    with st.spinner(
        "กำลังประมวลผลโมเดล Quant..."
    ):

        df = fetch_stock_data_and_simons_logic(
            symbols
        )

else:

    with st.spinner(
        "กำลังดึงรายชื่อหลักทรัพย์ทั้งหมด..."
    ):

        full_list_df = (
            get_full_us_market_symbols()
        )

    st.sidebar.success(
        f"พบหลักทรัพย์ทั้งหมด "
        f"{len(full_list_df):,} ตัว"
    )

    exclude_etf = st.sidebar.checkbox(
        "ไม่รวมกองทุน ETF",
        value=True
    )

    universe_df = (
        full_list_df[
            ~full_list_df["ETF"]
        ]
        if exclude_etf
        else full_list_df
    )

    max_scan = st.sidebar.number_input(
        "จำนวนหุ้นสูงสุดที่จะสแกน",
        min_value=50,
        max_value=int(
            len(universe_df)
        ),
        value=min(
            300,
            int(len(universe_df))
        ),
        step=50
    )

    start_offset = st.sidebar.number_input(
        "เริ่มสแกนจากลำดับที่",
        min_value=0,
        max_value=max(
            0,
            int(len(universe_df)) - 1
        ),
        value=0,
        step=max_scan
    )

    run_scan = st.sidebar.button(
        " เริ่มสแกนตลาดเต็มรูปแบบ",
        use_container_width=True
    )

    if run_scan:

        target_symbols = (
            universe_df["Symbol"]
            .iloc[
                start_offset:
                start_offset + max_scan
            ]
            .tolist()
        )

        st.session_state.full_scan_df = (
            fetch_stock_data_batched(
                target_symbols
            )
        )

        st.session_state.full_scan_range = (
            start_offset,
            start_offset + len(
                target_symbols
            )
        )

    if (
        "full_scan_df"
        in st.session_state
    ):

        df = (
            st.session_state
            .full_scan_df
        )

        rng = (
            st.session_state.get(
                "full_scan_range",
                (0, len(df))
            )
        )

        st.sidebar.caption(
            f"ผลสแกนล่าสุด: "
            f"{rng[0]:,}-{rng[1]:,}"
        )

    else:

        st.info(
            " กรุณาคลิก "
            "เริ่มสแกนตลาดที่ Sidebar"
        )

        st.stop()


if df.empty:

    st.error(
        "ไม่สามารถดึงข้อมูลตลาดได้ "
        "กรุณารีเฟรชหน้าเว็บ"
    )

    st.stop()


# =========================================================
# MAIN TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        " สแกนหุ้นตลาด",
        " วิเคราะห์หุ้นรายตัว",
        " พอร์ตของฉัน",
        " AI Portfolio Advisor"
    ]
)


# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.subheader(
        " Quantitative Stock Screener"
    )

    col1, col2 = st.columns(2)

    with col1:

        signal_filter = st.selectbox(
            "กรองตามคำแนะนำ",
            [
                "ทั้งหมด",
                "🟢 ซื้อสะสม (Strong Buy)",
                "🟡 ถือ / เฝ้าสังเกต (Hold)",
                "🔴 ควรขาย / หลีกเลี่ยง (Sell / Avoid)"
            ]
        )

    with col2:

        search_query = st.text_input(
            " ค้นหา Ticker",
            ""
        ).upper()

    df_filtered = df.copy()

    if signal_filter != "ทั้งหมด":

        df_filtered = df_filtered[
            df_filtered[
                "Simons Signal"
            ] == signal_filter
        ]

    if search_query:

        df_filtered = df_filtered[
            df_filtered[
                "Ticker"
            ].str.contains(
                search_query
            )
        ]

    st.dataframe(
        df_filtered.sort_values(
            by="Score",
            ascending=False
        ),
        column_config={

            "Price (USD)":
                st.column_config.NumberColumn(
                    format="$%.2f"
                ),

            "Momentum 1M (%)":
                st.column_config.NumberColumn(
                    format="%.2f%%"
                ),

            "Momentum 1W (%)":
                st.column_config.NumberColumn(
                    format="%.2f%%"
                ),

            "Volatility (%)":
                st.column_config.NumberColumn(
                    format="%.2f%%"
                ),

            "Score":
                st.column_config.NumberColumn(
                    format="%d คะแนน"
                )
        },
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# TAB 2
# =========================================================

def build_stock_specific_outlook(data, ticker, t_price, t_score, t_trend, t_mom1m, t_mom1w, t_vol):
    """สร้างมุมมองอนาคตแบบเจาะจงรายบริษัทจากข้อมูลพื้นฐาน + ราคา"""
    sector = str(data.get("sector") or "N/A")
    industry = str(data.get("industry") or "N/A")
    summary = str(data.get("summary") or "ไม่มีข้อมูลธุรกิจ")
    revenue_growth = data.get("revenue_growth")
    earnings_growth = data.get("earnings_growth")
    profit_margin = data.get("profit_margin")
    gross_margin = data.get("gross_margin")
    operating_margin = data.get("operating_margin")
    roe = data.get("return_on_equity")
    roa = data.get("return_on_assets")
    fcf = data.get("free_cash_flow")
    debt = data.get("debt_to_equity")
    forward_pe = data.get("forward_pe")
    pe = data.get("pe")
    peg = data.get("peg")
    target = data.get("target_mean")
    target_low = data.get("target_low")
    target_high = data.get("target_high")
    high52 = data.get("fifty_two_high")
    low52 = data.get("fifty_two_low")
    beta = data.get("beta")
    current_ratio = data.get("current_ratio")
    market_cap = data.get("market_cap")
    enterprise_value = data.get("enterprise_value")
    employees = data.get("employees")
    recommendation_key = data.get("recommendation_key")
    country = data.get("country")
    website = data.get("website")

    s = (sector + " " + industry).lower()
    catalysts = []
    risks = []
    business_drivers = []

    if "semiconductor" in s or "semiconductors" in s:
        business_drivers = [
            "การเติบโตของ AI accelerator / data-center และความต้องการ compute",
            "การลงทุนของ Cloud/Hyperscaler และรอบ CapEx ของ Data Center",
            "เทคโนโลยี node/process, packaging, networking หรือ memory ตามตำแหน่งของบริษัท",
            "การเพิ่ม market share และการเปิดตัวผลิตภัณฑ์รุ่นใหม่",
        ]
        catalysts = [
            "ยอดสั่งซื้อชิปหรือระบบ AI เพิ่มขึ้น",
            "Gross Margin / Operating Margin ขยายตัว",
            "การเปิดตัวผลิตภัณฑ์ใหม่ที่มี ASP และกำไรสูงขึ้น",
            "การเพิ่มกำลังผลิตโดยไม่ทำให้ต้นทุนต่อหน่วยสูงขึ้น",
        ]
        risks = [
            "วัฏจักร Semiconductor และการปรับลด Inventory",
            "การแข่งขันด้านเทคโนโลยีและการเปลี่ยน architecture",
            "Export controls / Geopolitical risk",
            "Valuation สูงเกินกว่าการเติบโตของกำไรจริง",
        ]
    elif "technology" in s or "software" in s:
        business_drivers = [
            "การเพิ่มจำนวนลูกค้าและรายได้ต่อหนึ่งลูกค้า",
            "Recurring revenue / subscription และการรักษา retention",
            "AI, cloud และ automation ที่เพิ่มมูลค่าต่อผลิตภัณฑ์",
            "Operating leverage เมื่อรายได้โตเร็วกว่าค่าใช้จ่าย",
        ]
        catalysts = [
            "Net new customers / bookings สูงกว่าคาด",
            "การเพิ่ม ARPU และ cross-sell",
            "AI monetization ที่แปลงเป็นรายได้จริง",
            "Free Cash Flow และ Margin ดีขึ้น",
        ]
        risks = [
            "การแข่งขันและการลดราคา",
            "AI disruption ต่อผลิตภัณฑ์เดิม",
            "ลูกค้าชะลอ IT spending",
            "Valuation ที่พึ่งพาการเติบโตระยะยาวสูง",
        ]
    elif "health" in s or "biotech" in s or "drug" in s:
        business_drivers = [
            "การเติบโตของผลิตภัณฑ์/ยาหลักและ pipeline",
            "ตลาดผู้สูงอายุและโรคเรื้อรัง",
            "การอนุมัติผลิตภัณฑ์และการขยาย indication",
            "การขยายตลาดและความสามารถในการรักษา pricing power",
        ]
        catalysts = [
            "ผลการทดลองหรือ FDA approval ที่เป็นบวก",
            "ยอดขายผลิตภัณฑ์หลักสูงกว่าคาด",
            "การเพิ่ม indication / geographic expansion",
            "Margin และ Free Cash Flow ดีขึ้น",
        ]
        risks = [
            "Clinical / regulatory risk",
            "Patent expiry และการแข่งขัน Generic",
            "การควบคุมราคายา",
            "Pipeline ไม่สามารถชดเชยผลิตภัณฑ์เดิมได้",
        ]
    elif "financial" in s:
        business_drivers = [
            "ทิศทางอัตราดอกเบี้ยและ Net Interest Margin",
            "การเติบโตของสินเชื่อและคุณภาพสินทรัพย์",
            "Investment banking / trading / wealth management",
            "ประสิทธิภาพต้นทุนและ digital transformation",
        ]
        catalysts = [
            "Credit growth และ fee income ดีขึ้น",
            "Credit losses ต่ำกว่าคาด",
            "ต้นทุนดำเนินงานลดลง",
            "Capital return / buyback สนับสนุน EPS",
        ]
        risks = [
            "Credit losses และเศรษฐกิจถดถอย",
            "Interest-rate sensitivity",
            "Regulatory capital requirement",
            "Yield curve / liquidity pressure",
        ]
    elif "consumer" in s or "retail" in s:
        business_drivers = [
            "กำลังซื้อและจำนวนลูกค้า",
            "Same-store sales / e-commerce growth",
            "Pricing power และความแข็งแรงของแบรนด์",
            "การบริหาร Inventory และ Supply Chain",
        ]
        catalysts = [
            "ยอดขายสาขาเดิมและ e-commerce สูงกว่าคาด",
            "Gross Margin ฟื้นตัว",
            "การเพิ่ม market share",
            "การขยายสาขาหรือสินค้าใหม่",
        ]
        risks = [
            "ผู้บริโภคลดการใช้จ่าย",
            "เงินเฟ้อและต้นทุนสินค้า",
            "การแข่งขันด้านราคา",
            "Inventory สูงและ markdown pressure",
        ]
    elif "industrial" in s:
        business_drivers = [
            "คำสั่งซื้อใหม่และ backlog",
            "Infrastructure / manufacturing CapEx",
            "Automation และ productivity",
            "การควบคุมต้นทุนและ operating leverage",
        ]
        catalysts = [
            "Backlog เพิ่มขึ้น",
            "คำสั่งซื้อและ utilization สูงขึ้น",
            "Margin expansion",
            "การลงทุนโครงสร้างพื้นฐาน",
        ]
        risks = [
            "เศรษฐกิจชะลอตัวและคำสั่งซื้อลด",
            "ต้นทุนวัตถุดิบ",
            "Supply-chain disruption",
            "Cyclical demand",
        ]
    else:
        business_drivers = [
            "การเติบโตของรายได้และฐานลูกค้า",
            "การเพิ่ม market share",
            "การเปิดตลาดหรือผลิตภัณฑ์ใหม่",
            "การเพิ่มประสิทธิภาพต้นทุนและกระแสเงินสด",
        ]
        catalysts = [
            "Revenue / Earnings สูงกว่าคาด",
            "Margin ขยายตัว",
            "ผลิตภัณฑ์ใหม่หรือการขยายตลาด",
            "Free Cash Flow แข็งแรงขึ้น",
        ]
        risks = [
            "เศรษฐกิจมหภาค",
            "การแข่งขัน",
            "ต้นทุนและ Supply Chain",
            "ดอกเบี้ยและ Regulation",
        ]

    strengths = []
    watch = []

    if revenue_growth is not None:
        strengths.append(f"รายได้เติบโต {revenue_growth*100:+.1f}%")
        watch.append("ติดตาม Revenue Growth และ Guidance รายไตรมาส")
    if earnings_growth is not None:
        strengths.append(f"กำไรเติบโต {earnings_growth*100:+.1f}%")
        watch.append("ติดตาม EPS / Earnings Growth ว่าโตจากธุรกิจจริงหรือรายการพิเศษ")
    if profit_margin is not None:
        strengths.append(f"Net Margin {profit_margin*100:.1f}%")
        watch.append("ติดตาม Margin ว่ารักษาระดับได้หรือไม่")
    if roe is not None:
        strengths.append(f"ROE {roe*100:.1f}%")
    if fcf is not None:
        strengths.append(f"Free Cash Flow {format_money(fcf)}")
    if debt is not None:
        if debt < 80:
            strengths.append(f"Debt/Equity {debt:.1f} อยู่ในระดับไม่สูงมาก")
        elif debt > 150:
            risks.insert(0, f"Debt/Equity สูงที่ {debt:.1f} ต้องติดตามภาระหนี้และดอกเบี้ย")
    if t_price > 0 and t_mom1m > 0 and t_price > data.get("fiftyTwoWeekLow", t_price):
        pass

    valuation_signal = "ประเมินไม่ได้"
    chosen_pe = forward_pe if forward_pe is not None else pe
    if chosen_pe is not None:
        if chosen_pe < 15:
            valuation_signal = "ค่อนข้างถูกเมื่อเทียบกับหุ้นที่มีกำไร"
        elif chosen_pe < 25:
            valuation_signal = "ระดับปานกลาง"
        elif chosen_pe < 40:
            valuation_signal = "ค่อนข้างสูง ต้องมีการเติบโตของกำไรสนับสนุน"
        else:
            valuation_signal = "สูงมาก มีความคาดหวังการเติบโตสูง"

    target_upside = None
    if target is not None and t_price > 0:
        target_upside = (target / t_price - 1) * 100

    range_position = None
    if low52 is not None and high52 is not None and high52 > low52:
        range_position = max(0, min(100, (t_price - low52) / (high52 - low52) * 100))

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
        long_term = "พื้นฐานมีโอกาสสนับสนุนการเติบโตระยะยาว เพราะรายได้ยังขยายตัวและกำไรไม่ได้สวนทาง อย่างไรก็ตามต้องพิสูจน์ว่าการเติบโตสามารถแปลงเป็น Free Cash Flow ได้ต่อเนื่อง"
    elif profit_margin is not None and profit_margin > 0.15 and fcf is not None and fcf > 0:
        long_term = "คุณภาพธุรกิจน่าสนใจจาก Margin และกระแสเงินสด แต่การเติบโตในอนาคตต้องติดตามว่าบริษัทสามารถเพิ่มรายได้โดยไม่ต้องใช้เงินลงทุนสูงเกินไปหรือไม่"
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
            "country": country,
            "website": website,
            "employees": employees,
            "market_cap": market_cap,
            "enterprise_value": enterprise_value,
            "recommendation_key": recommendation_key,
            "beta": beta,
            "current_ratio": current_ratio,
            "gross_margin": gross_margin,
            "operating_margin": operating_margin,
            "roe": roe,
            "roa": roa,
            "peg": peg,
        },
    }


with tab2:
    st.subheader(" วิเคราะห์หุ้นรายตัวแบบละเอียด")
    st.write(
        "กรอก Ticker เพื่อดูข้อมูลบริษัท พื้นฐาน การเติบโต Valuation "
        "Momentum ความเสี่ยง และแนวโน้มธุรกิจในอนาคตแบบเจาะจงรายหุ้น"
    )

    inner_search_ticker = st.text_input(
        " Ticker หุ้น",
        "NVDA",
        key="detail_ticker"
    ).upper().strip()

    if not inner_search_ticker:
        st.info("กรุณากรอก Ticker")
    else:
        match_dd = df[df["Ticker"] == inner_search_ticker]

        if match_dd.empty:
            try:
                with st.spinner(f"กำลังดึงข้อมูล {inner_search_ticker}..."):
                    hist_dd = yf.download(
                        inner_search_ticker,
                        period="6mo",
                        progress=False,
                        auto_adjust=True
                    )
                    if not hist_dd.empty:
                        if isinstance(hist_dd.columns, pd.MultiIndex):
                            cs_dd = hist_dd["Close"].iloc[:, 0]
                        else:
                            cs_dd = hist_dd["Close"]
                        row_dd = compute_simons_row(inner_search_ticker, cs_dd)
                        if row_dd:
                            match_dd = pd.DataFrame([row_dd])
            except Exception as e:
                st.warning(f"ไม่สามารถดึงข้อมูลราคาของ {inner_search_ticker}: {e}")

        if match_dd.empty:
            st.error(f"ไม่พบข้อมูลสำหรับ {inner_search_ticker}")
        else:
            r = match_dd.iloc[0]
            t_ticker = str(r["Ticker"])
            t_price = float(r["Price (USD)"])
            t_score = int(r["Score"])
            t_signal = str(r["Simons Signal"])
            t_trend = str(r["Trend"])
            t_mom1m = float(r["Momentum 1M (%)"])
            t_mom1w = float(r["Momentum 1W (%)"])
            t_vol = float(r["Volatility (%)"])
            t_sma50 = float(r["SMA 50"])

            with st.spinner(f"กำลังวิเคราะห์ข้อมูลพื้นฐานของ {t_ticker}..."):
                analysis = get_detailed_stock_analysis(t_ticker)

            if "error" in analysis:
                st.error(f"ไม่สามารถสร้างรายงานได้: {analysis['error']}")
            else:
                data = analysis["data"]
                outlook = build_stock_specific_outlook(
                    data, t_ticker, t_price, t_score, t_trend, t_mom1m, t_mom1w, t_vol
                )

                st.markdown("---")
                st.title(f" {t_ticker}")
                st.caption(
                    f"{data['long_name']} | {data['sector']} | {data['industry']}"
                )

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("ราคาปัจจุบัน", f"${t_price:,.2f}")
                c2.metric("Quant Score", f"{t_score}/100")
                c3.metric("1M Momentum", f"{t_mom1m:+.2f}%")
                c4.metric("1W Momentum", f"{t_mom1w:+.2f}%")
                c5.metric("Volatility", f"{t_vol:.2f}%")

                # 1 Business profile
                st.markdown("---")
                st.subheader("1. บริษัททำธุรกิจอะไร? (Business Profile)")
                st.write(data["summary"])

                st.markdown("#### คำศัพท์จากข้อมูลบริษัท (Business Vocabulary)")
                learning_terms = company_learning_terms(data["summary"], limit=12)
                if learning_terms:
                    vocab_rows = [{"English": en, "คำแปลไทย": th} for en, th in learning_terms]
                    st.dataframe(pd.DataFrame(vocab_rows), use_container_width=True, hide_index=True)
                else:
                    st.caption("ยังไม่พบคำศัพท์ที่ตรงกับรายการฝึกภาษาในคำอธิบายบริษัท")

                st.markdown("#### สรุปข้อมูลบริษัทแบบไทย + English")
                st.write(f"บริษัท: **{data['long_name']}** | Company: **{data['long_name']}**")
                st.write(f"กลุ่มธุรกิจ (Sector): **{data['sector']}** — {translate_sector_name(data['sector'])}")
                st.write(f"อุตสาหกรรม (Industry): **{data['industry']}** — ใช้คำศัพท์นี้เพื่อระบุประเภทอุตสาหกรรมหลักของบริษัท")

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

                # 2 Business model and future
                st.subheader("2. แนวโน้มธุรกิจในอนาคตแบบเจาะจงรายหุ้น (Future Business Outlook)")
                st.write(
                    f"หุ้น {t_ticker} อยู่ในกลุ่ม {outlook['sector_industry']} "
                    "ดังนั้นการประเมินอนาคตควรเชื่อมโยง 3 เรื่องเข้าด้วยกัน: "
                    "การเติบโตของตลาด, ความสามารถของบริษัทในการรักษา Margin/Market Share "
                    "และราคาที่นักลงทุนยอมจ่ายให้กับกำไรในอนาคต"
                )

                st.markdown("#### เครื่องยนต์ธุรกิจที่ต้องจับตา")
                for item in outlook["business_drivers"]:
                    st.markdown(f"- {item}")

                st.markdown("#### จุดแข็งที่เห็นจากข้อมูลล่าสุด")
                for item in outlook["strengths"]:
                    st.markdown(f"- {item}")

                st.markdown("#### แนวโน้มระยะใกล้–กลาง")
                st.info(outlook["near_term"])

                st.markdown("#### แนวโน้มระยะยาว")
                st.info(outlook["long_term"])

                # 3 Financial snapshot
                st.subheader("3. ข้อมูลการเงินและปัจจัยพื้นฐาน (Financial & Fundamental Snapshot)")
                f1, f2, f3, f4 = st.columns(4)
                f1.metric("Revenue", format_money(data["revenue"]))
                f2.metric(
                    "Revenue Growth",
                    pct_text(data["revenue_growth"])
                )
                f3.metric(
                    "Earnings Growth",
                    pct_text(data["earnings_growth"])
                )
                f4.metric(
                    "Free Cash Flow",
                    format_money(data["free_cash_flow"])
                )

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

                # 4 Growth engine
                st.subheader("4.  Growth Engine")
                st.write(
                    "ดูว่าการเติบโตของบริษัทเกิดจาก Revenue, Earnings และ Cash Flow "
                    "หรือเป็นเพียงการขยาย Valuation"
                )
                g1, g2, g3 = st.columns(3)
                g1.metric("Growth Score", f"{analysis['growth_score']}/6")
                g2.metric("Quality Score", f"{analysis['quality_score']}/6")
                g3.metric("Debt Score", f"{analysis['debt_score']}/3")

                growth_notes = []
                if data["revenue_growth"] is not None:
                    growth_notes.append(
                        f"Revenue Growth {data['revenue_growth']*100:+.2f}%"
                    )
                if data["earnings_growth"] is not None:
                    growth_notes.append(
                        f"Earnings Growth {data['earnings_growth']*100:+.2f}%"
                    )
                if data["free_cash_flow"] is not None:
                    growth_notes.append(
                        f"Free Cash Flow {format_money(data['free_cash_flow'])}"
                    )
                st.write(" | ".join(growth_notes) if growth_notes else "ข้อมูล Growth ไม่ครบ")

                # 5 Valuation
                st.subheader("5.  Valuation")
                v1, v2, v3, v4 = st.columns(4)
                v1.metric("P/E", metric_or_na(data["pe"], "x"))
                v2.metric("Forward P/E", metric_or_na(data["forward_pe"], "x"))
                v3.metric("PEG", metric_or_na(data["peg"]))
                v4.metric("P/S", metric_or_na(data["price_to_sales"], "x"))
                st.info(
                    f"มุมมอง Valuation: {analysis['valuation_text']} "
                    f"({outlook['valuation_signal']})"
                )

                # 6 Analyst targets
                st.subheader("6. ราคาเป้าหมายจากนักวิเคราะห์ (Analyst Target Price)")
                a1, a2, a3, a4 = st.columns(4)
                a1.metric(
                    "Target Low",
                    f"${data['target_low']:,.2f}" if data["target_low"] is not None else "N/A"
                )
                a2.metric(
                    "Target Mean",
                    f"${data['target_mean']:,.2f}" if data["target_mean"] is not None else "N/A"
                )
                a3.metric(
                    "Target High",
                    f"${data['target_high']:,.2f}" if data["target_high"] is not None else "N/A"
                )
                if outlook["target_upside"] is not None:
                    a4.metric("Upside to Mean", f"{outlook['target_upside']:+.2f}%")
                else:
                    a4.metric("Upside to Mean", "N/A")

                # 7 Catalysts and risks
                st.subheader("7. ปัจจัยกระตุ้นราคาหุ้น (Catalysts)")
                for item in outlook["catalysts"]:
                    st.markdown(f"- **{item}**")

                st.subheader("8.  ความเสี่ยงเฉพาะหุ้น")
                for item in outlook["risks"]:
                    st.markdown(f"- **{item}**")

                # 9 Technical
                st.subheader("9. การวิเคราะห์ทางเทคนิคและโมเมนตัม (Technical & Momentum)")
                tc1, tc2, tc3, tc4, tc5 = st.columns(5)
                tc1.metric("1 Week", f"{t_mom1w:+.2f}%")
                tc2.metric("1 Month", f"{t_mom1m:+.2f}%")
                tc3.metric("SMA 50", f"${t_sma50:,.2f}")
                tc4.metric("Trend", t_trend)
                tc5.metric("Quant Signal", t_signal)

                if t_price > t_sma50:
                    st.success("ราคาปัจจุบันอยู่เหนือ SMA 50 — โครงสร้าง Momentum ระยะกลางเป็นบวก")
                else:
                    st.warning("ราคาปัจจุบันต่ำกว่า SMA 50 — โครงสร้าง Momentum ระยะกลางยังอ่อนแอ")

                # 10 Risk profile
                st.subheader("10. โปรไฟล์ความเสี่ยง (Risk Profile)")
                r1, r2, r3 = st.columns(3)
                r1.metric("Annualized Volatility", f"{t_vol:.2f}%")
                r2.metric("Beta", metric_or_na(data["beta"]))
                r3.metric("Debt / Equity", metric_or_na(data["debt_to_equity"]))

                if t_vol >= 60:
                    st.error("ความผันผวนสูงมาก: ควรลด Position Size และหลีกเลี่ยงการใช้เงินก้อนใหญ่ในจังหวะเดียว")
                elif t_vol >= 40:
                    st.warning("ความผันผวนค่อนข้างสูง: เหมาะกับการทยอยซื้อและกำหนดขนาด Position ให้เหมาะสม")
                else:
                    st.success("ความผันผวนอยู่ในระดับที่ต่ำกว่าเกณฑ์ 40% ของโมเดล Quant")

                # 11 52-week position
                st.subheader("11.  ตำแหน่งราคาในกรอบ 52 สัปดาห์")
                low52 = data["fifty_two_low"]
                high52 = data["fifty_two_high"]
                if outlook["range_position"] is not None:
                    st.progress(outlook["range_position"] / 100)
                    st.caption(
                        f"52W Low: ${low52:,.2f} | Current: ${t_price:,.2f} | "
                        f"52W High: ${high52:,.2f}"
                    )
                    st.write(
                        f"ราคาปัจจุบันอยู่ที่ประมาณ **{outlook['range_position']:.1f}%** "
                        "ของกรอบ 52 สัปดาห์"
                    )
                else:
                    st.info("ไม่มีข้อมูล 52-week range ที่เพียงพอ")

                # 12 Bull/Base/Bear
                st.subheader("12.  Bull / Base / Bear Scenario")
                bull, base, bear = st.columns(3)
                with bull:
                    st.success(
                        "Bull Case\n\n"
                        "• Revenue/Earnings สูงกว่าคาด\n\n"
                        "• Margin ขยายตัว\n\n"
                        "• Market Share เพิ่ม\n\n"
                        "• Valuation ยังได้รับ Premium"
                    )
                with base:
                    st.info(
                        "Base Case\n\n"
                        "• Revenue โตตามอุตสาหกรรม\n\n"
                        "• Margin ทรงตัว\n\n"
                        "• Earnings โตปานกลาง\n\n"
                        "• Valuation เคลื่อนไหวใกล้ระดับเดิม"
                    )
                with bear:
                    st.error(
                        "Bear Case\n\n"
                        "• Revenue/Earnings ชะลอตัว\n\n"
                        "• Margin หดตัว\n\n"
                        "• คู่แข่งกดดัน\n\n"
                        "• Valuation ถูก De-rate"
                    )

                # 13 Final verdict
                st.subheader("13.  Final Investment Verdict")
                if t_score >= 70:
                    st.success(
                        f" STRONG BUY / ACCUMULATE — {t_ticker} ได้ Quant Score "
                        f"{t_score}/100 และ Momentum เป็นบวก เหมาะกับการพิจารณาทยอยสะสม "
                        "โดยต้องตรวจสอบ Valuation และงบล่าสุดประกอบ"
                    )
                elif t_score >= 45:
                    st.warning(
                        f" HOLD / WAIT & SEE — {t_ticker} ได้ Quant Score "
                        f"{t_score}/100 สัญญาณยังไม่แข็งแรงพอสำหรับการเพิ่มน้ำหนักแบบ aggressive"
                    )
                else:
                    st.error(
                        f" AVOID / REDUCE RISK — {t_ticker} ได้ Quant Score "
                        f"{t_score}/100 โครงสร้าง Quant ยังอ่อนแอ ควรรอสัญญาณ Trend ฟื้นตัว"
                    )

                # 14 What to monitor
                st.subheader("14.  สิ่งที่ควรติดตามในงบไตรมาสถัดไป")
                for item in outlook["watch"]:
                    st.markdown(f"- {item}")
                st.markdown("- การเปลี่ยนแปลงของ Guidance และประมาณการ EPS")
                st.markdown("- Free Cash Flow และระดับหนี้")
                st.markdown("- การตอบสนองของราคาหุ้นต่อผลประกอบการ")

                # 15 Price chart
                st.subheader(f"15.  ราคาย้อนหลัง 6 เดือน — {t_ticker}")
                try:
                    hist_chart = yf.download(
                        t_ticker,
                        period="6mo",
                        progress=False,
                        auto_adjust=True
                    )
                    if not hist_chart.empty:
                        if isinstance(hist_chart.columns, pd.MultiIndex):
                            close_c = hist_chart["Close"].iloc[:, 0]
                        else:
                            close_c = hist_chart["Close"]
                        st.line_chart(close_c)
                except Exception as e:
                    st.warning(f"ไม่สามารถสร้างกราฟได้: {e}")

                st.caption(
                    "หมายเหตุ: รายงานนี้ใช้ข้อมูลจาก Yahoo Finance และโมเดล Quant "
                    "ข้อมูลตลาดและตัวชี้วัดสามารถเปลี่ยนแปลงได้ ไม่ควรใช้เป็นคำแนะนำการลงทุนเพียงแหล่งเดียว"
                )


# =========================================================
# TAB 3 PORTFOLIO
# =========================================================

with tab3:

    st.subheader(
        " พอร์ตการลงทุนของคุณ"
    )

    st.write(
        "กรอกข้อมูลพอร์ต หรืออัปโหลดใบเสร็จ "
        "เพื่อให้ AI อ่านข้อมูลหุ้น"
    )

    with st.expander(
        " AI OCR — อ่านใบเสร็จซื้อขายหุ้น",
        expanded=False
    ):

        uploaded_receipt = st.file_uploader(
            "เลือกรูปภาพ",
            type=[
                "png",
                "jpg",
                "jpeg"
            ]
        )

        ai_key_receipt = st.sidebar.text_input(
            " OpenAI API Key สำหรับ OCR",
            type="password",
            key="receipt_key"
        )

        if uploaded_receipt is not None:

            image = Image.open(
                uploaded_receipt
            )

            st.image(
                image,
                caption="ใบเสร็จ",
                width=300
            )

            if st.button(
                " สแกนใบเสร็จ"
            ):

                if not ai_key_receipt:

                    st.error(
                        "กรุณาใส่ OpenAI API Key"
                    )

                else:

                    with st.spinner(
                        "AI กำลังอ่านใบเสร็จ..."
                    ):

                        try:

                            buffered = io.BytesIO()

                            image.save(
                                buffered,
                                format="JPEG"
                            )

                            img_str = (
                                base64.b64encode(
                                    buffered.getvalue()
                                )
                                .decode("utf-8")
                            )

                            client = openai.OpenAI(
                                api_key=ai_key_receipt
                            )

                            response = (
                                client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[
                                        {
                                            "role": "system",
                                            "content":
                                                """
                                                อ่านใบเสร็จซื้อขายหุ้น
                                                และตอบ JSON เท่านั้น

                                                {
                                                  "Ticker":"AAPL",
                                                  "Shares":10,
                                                  "Buy Price (USD)":175.5
                                                }
                                                """
                                        },
                                        {
                                            "role": "user",
                                            "content": [
                                                {
                                                    "type": "text",
                                                    "text":
                                                        "อ่าน Ticker จำนวนหุ้น และราคาซื้อ"
                                                },
                                                {
                                                    "type": "image_url",
                                                    "image_url": {
                                                        "url":
                                                            f"data:image/jpeg;base64,{img_str}"
                                                    }
                                                }
                                            ]
                                        }
                                    ],
                                    max_tokens=150
                                )
                            )

                            res_text = (
                                response
                                .choices[0]
                                .message
                                .content
                                .strip()
                            )

                            if res_text.startswith(
                                "```json"
                            ):

                                res_text = (
                                    res_text[7:-3]
                                    .strip()
                                )

                            elif res_text.startswith(
                                "```"
                            ):

                                res_text = (
                                    res_text[3:-3]
                                    .strip()
                                )

                            extracted = json.loads(
                                res_text
                            )

                            new_ticker = (
                                str(
                                    extracted.get(
                                        "Ticker",
                                        ""
                                    )
                                )
                                .upper()
                            )

                            new_shares = float(
                                extracted.get(
                                    "Shares",
                                    0
                                )
                            )

                            new_price = float(
                                extracted.get(
                                    "Buy Price (USD)",
                                    0
                                )
                            )

                            if (
                                new_ticker
                                and new_shares > 0
                                and new_price > 0
                            ):

                                if (
                                    "portfolio_input"
                                    not in
                                    st.session_state
                                ):

                                    st.session_state[
                                        "portfolio_input"
                                    ] = (
                                        load_portfolio_from_disk()
                                    )

                                new_row = pd.DataFrame(
                                    [
                                        {
                                            "Ticker":
                                                new_ticker,
                                            "Shares":
                                                new_shares,
                                            "Buy Price (USD)":
                                                new_price,
                                            "Asset Class":
                                                "Growth"
                                        }
                                    ]
                                )

                                st.session_state[
                                    "portfolio_input"
                                ] = pd.concat(
                                    [
                                        st.session_state[
                                            "portfolio_input"
                                        ],
                                        new_row
                                    ],
                                    ignore_index=True
                                )

                                save_portfolio_to_disk(
                                    st.session_state[
                                        "portfolio_input"
                                    ]
                                )

                                st.success(
                                    f"เพิ่ม {new_ticker} "
                                    f"เรียบร้อยแล้ว"
                                )

                                st.rerun()

                            else:

                                st.error(
                                    "AI อ่านข้อมูลไม่ครบ"
                                )

                        except Exception as e:

                            st.error(
                                f"เกิดข้อผิดพลาด: {e}"
                            )

    if (
        "portfolio_input"
        not in st.session_state
    ):

        st.session_state[
            "portfolio_input"
        ] = load_portfolio_from_disk()

    edited_portfolio = st.data_editor(
        st.session_state[
            "portfolio_input"
        ],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Asset Class":
                st.column_config.SelectboxColumn(
                    "Asset Class",
                    options=[
                        "Growth",
                        "Defensive"
                    ],
                    default="Growth"
                )
        }
    )

    try:

        changed = not edited_portfolio.equals(
            st.session_state[
                "portfolio_input"
            ]
        )

    except Exception:

        changed = True

    if changed:

        st.session_state[
            "portfolio_input"
        ] = edited_portfolio

        save_portfolio_to_disk(
            edited_portfolio
        )

    if st.button(
        " ล้างพอร์ตเป็นค่าเริ่มต้น"
    ):

        default_df = pd.DataFrame(
            [
                {
                    "Ticker": "AAPL",
                    "Shares": 10.0,
                    "Buy Price (USD)": 170.0,
                    "Asset Class": "Growth"
                },
                {
                    "Ticker": "TSLA",
                    "Shares": 5.0,
                    "Buy Price (USD)": 220.0,
                    "Asset Class": "Growth"
                }
            ]
        )

        st.session_state[
            "portfolio_input"
        ] = default_df

        save_portfolio_to_disk(
            default_df
        )

        st.rerun()

    # -----------------------------------------------------
    # CALCULATE PORTFOLIO
    # -----------------------------------------------------

    df_res = pd.DataFrame()

    if not edited_portfolio.empty:

        clean_rows = []

        for _, row in (
            edited_portfolio.iterrows()
        ):

            ticker = str(
                row["Ticker"]
            ).strip().upper()

            shares = float(
                row["Shares"]
            ) if pd.notna(
                row["Shares"]
            ) else 0

            buy_price = float(
                row["Buy Price (USD)"]
            ) if pd.notna(
                row["Buy Price (USD)"]
            ) else 0

            asset_class = str(
                row.get(
                    "Asset Class",
                    "Growth"
                )
            )

            if (
                ticker
                and shares > 0
                and buy_price > 0
            ):

                clean_rows.append(
                    {
                        "Ticker":
                            ticker,
                        "Shares":
                            shares,
                        "Buy Price (USD)":
                            buy_price,
                        "Asset Class":
                            asset_class
                    }
                )

        if clean_rows:

            df_clean = pd.DataFrame(
                clean_rows
            )

            df_clean[
                "Cost Basis ($)"
            ] = (
                df_clean["Shares"]
                * df_clean["Buy Price (USD)"]
            )

            agg = (
                df_clean
                .groupby("Ticker")
                .agg(
                    Shares=(
                        "Shares",
                        "sum"
                    ),
                    CostBasis=(
                        "Cost Basis ($)",
                        "sum"
                    ),
                    AssetClass=(
                        "Asset Class",
                        "first"
                    )
                )
                .reset_index()
            )

            agg[
                "Buy Price (USD)"
            ] = (
                agg["CostBasis"]
                / agg["Shares"]
            )

        else:

            agg = pd.DataFrame()

        portfolio_results = []

        for _, row in agg.iterrows():

            ticker = row["Ticker"]

            shares = float(
                row["Shares"]
            )

            buy_price = float(
                row["Buy Price (USD)"]
            )

            asset_class = row[
                "AssetClass"
            ]

            current_price = 0

            signal = "ถือประเมินสถานะ"

            score = 50

            trend = "ไม่ทราบ"

            volatility = 0

            sma50 = 0

            mom1m = 0

            match_row = df[
                df["Ticker"]
                == ticker
            ]

            if not match_row.empty:

                current_price = float(
                    match_row.iloc[0][
                        "Price (USD)"
                    ]
                )

                signal = (
                    match_row.iloc[0][
                        "Simons Signal"
                    ]
                )

                score = int(
                    match_row.iloc[0][
                        "Score"
                    ]
                )

                trend = (
                    match_row.iloc[0][
                        "Trend"
                    ]
                )

                volatility = float(
                    match_row.iloc[0][
                        "Volatility (%)"
                    ]
                )

                sma50 = float(
                    match_row.iloc[0][
                        "SMA 50"
                    ]
                )

                mom1m = float(
                    match_row.iloc[0][
                        "Momentum 1M (%)"
                    ]
                )

            else:

                try:

                    hist = yf.download(
                        ticker,
                        period="6mo",
                        progress=False,
                        auto_adjust=True
                    )

                    if not hist.empty:

                        if isinstance(
                            hist.columns,
                            pd.MultiIndex
                        ):

                            close_series = (
                                hist[
                                    "Close"
                                ]
                                .iloc[:, 0]
                            )

                        else:

                            close_series = (
                                hist[
                                    "Close"
                                ]
                            )

                        current_price = float(
                            close_series.iloc[-1]
                        )

                except Exception:

                    current_price = buy_price

            invested_value = (
                shares
                * buy_price
            )

            current_value = (
                shares
                * current_price
            )

            profit_loss = (
                current_value
                - invested_value
            )

            profit_loss_pct = (
                (
                    current_price
                    - buy_price
                )
                / buy_price
                * 100
                if buy_price > 0
                else 0
            )

            portfolio_results.append(
                {
                    "Ticker":
                        ticker,
                    "Shares":
                        shares,
                    "Buy Price (USD)":
                        buy_price,
                    "Current Price (USD)":
                        current_price,
                    "Invested Value ($)":
                        invested_value,
                    "Current Value ($)":
                        current_value,
                    "Profit/Loss ($)":
                        profit_loss,
                    "Profit/Loss (%)":
                        profit_loss_pct,
                    "Simons Signal":
                        signal,
                    "Score":
                        score,
                    "Trend":
                        trend,
                    "Asset Class":
                        asset_class
                }
            )

        if portfolio_results:

            df_res = pd.DataFrame(
                portfolio_results
            )

            total_invested = (
                df_res[
                    "Invested Value ($)"
                ].sum()
            )

            total_current = (
                df_res[
                    "Current Value ($)"
                ].sum()
            )

            total_pl = (
                total_current
                - total_invested
            )

            total_pl_pct = (
                total_pl
                / total_invested
                * 100
                if total_invested > 0
                else 0
            )

            df_res[
                "Weight in Portfolio (%)"
            ] = (
                df_res[
                    "Current Value ($)"
                ]
                / total_current
                * 100
                if total_current > 0
                else 0
            )

            # =============================================
            # PORTFOLIO METRICS
            # =============================================

            st.markdown("---")

            m1, m2, m3 = st.columns(3)

            m1.metric(
                " เงินลงทุน",
                f"${total_invested:,.2f}"
            )

            m2.metric(
                " มูลค่าปัจจุบัน",
                f"${total_current:,.2f}"
            )

            m3.metric(
                "กำไร/ขาดทุน",
                f"${total_pl:,.2f}",
                f"{total_pl_pct:.2f}%"
            )

            # =============================================
            # ACTION
            # =============================================

            st.subheader(
                " Quantitative Action Summary"
            )

            buy_candidates = df_res[
                df_res["Score"] >= 70
            ]

            sell_candidates = df_res[
                df_res["Score"] < 45
            ]

            act1, act2 = st.columns(2)

            with act1:

                st.markdown(
                    "###  ซื้อเพิ่ม"
                )

                if not buy_candidates.empty:

                    for _, x in (
                        buy_candidates.iterrows()
                    ):

                        st.success(
                            f"""
                            {x['Ticker']}
                            
                            Score: {x['Score']}/100
                            
                            สัญญาณ:
                            {x['Simons Signal']}
                            """
                        )

                else:

                    st.info(
                        "ยังไม่มีหุ้นที่เข้าเกณฑ์ซื้อเพิ่ม"
                    )

            with act2:

                st.markdown(
                    "###  ลดความเสี่ยง"
                )

                if not sell_candidates.empty:

                    for _, x in (
                        sell_candidates.iterrows()
                    ):

                        st.error(
                            f"""
                            {x['Ticker']}
                            
                            Score: {x['Score']}/100
                            
                            สัญญาณ:
                            {x['Simons Signal']}
                            """
                        )

                else:

                    st.success(
                        "ไม่มีหุ้นที่เข้าเกณฑ์ขาย"
                    )

            # =============================================
            # TABLE
            # =============================================

            st.subheader(
                " สรุปพอร์ต"
            )

            st.dataframe(
                df_res[
                    [
                        "Ticker",
                        "Shares",
                        "Buy Price (USD)",
                        "Current Price (USD)",
                        "Profit/Loss ($)",
                        "Profit/Loss (%)",
                        "Weight in Portfolio (%)",
                        "Asset Class",
                        "Simons Signal",
                        "Score"
                    ]
                ],
                column_config={

                    "Buy Price (USD)":
                        st.column_config.NumberColumn(
                            format="$%.2f"
                        ),

                    "Current Price (USD)":
                        st.column_config.NumberColumn(
                            format="$%.2f"
                        ),

                    "Profit/Loss ($)":
                        st.column_config.NumberColumn(
                            format="$%.2f"
                        ),

                    "Profit/Loss (%)":
                        st.column_config.NumberColumn(
                            format="%.2f%%"
                        ),

                    "Weight in Portfolio (%)":
                        st.column_config.NumberColumn(
                            format="%.1f%%"
                        ),

                    "Score":
                        st.column_config.NumberColumn(
                            format="%d คะแนน"
                        )
                },
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                " ดาวน์โหลด Portfolio CSV",
                data=df_res.to_csv(
                    index=False
                ).encode(
                    "utf-8-sig"
                ),
                file_name="portfolio_report.csv",
                mime="text/csv"
            )


# =========================================================
# TAB 4 AI CHAT
# =========================================================

with tab4:

    st.subheader(
        " AI Portfolio Advisor"
    )

    openai_api_key = st.sidebar.text_input(
        " OpenAI API Key สำหรับ AI Chat",
        type="password",
        key="chat_key"
    )

    if openai_api_key:

        portfolio_context = (
            "ไม่มีข้อมูลพอร์ต"
        )

        if (
            "df_res" in locals()
            and not df_res.empty
        ):

            lines = []

            for _, r in df_res.iterrows():

                lines.append(
                    f"""
                    หุ้น {r['Ticker']}:
                    ถือ {r['Shares']} หุ้น,
                    ต้นทุน ${r['Buy Price (USD)']:.2f},
                    ราคาปัจจุบัน ${r['Current Price (USD)']:.2f},
                    P/L {r['Profit/Loss (%)']:.2f}%,
                    Weight {r['Weight in Portfolio (%)']:.1f}%,
                    Asset Class {r['Asset Class']},
                    Quant Score {r['Score']}/100,
                    Signal {r['Simons Signal']}
                    """
                )

            portfolio_context = "\n".join(
                lines
            )

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

        if (
            "messages" not in st.session_state
            or st.session_state.get(
                "last_system_prompt"
            ) != system_instruction
        ):

            st.session_state.messages = [
                {
                    "role":
                        "system",
                    "content":
                        system_instruction
                }
            ]

            st.session_state[
                "last_system_prompt"
            ] = system_instruction

        for message in (
            st.session_state.messages
        ):

            if message["role"] != "system":

                with st.chat_message(
                    message["role"]
                ):

                    st.markdown(
                        message["content"]
                    )

        prompt = st.chat_input(
            "ถาม AI เกี่ยวกับหุ้นหรือพอร์ต..."
        )

        if prompt:

            st.session_state.messages.append(
                {
                    "role":
                        "user",
                    "content":
                        prompt
                }
            )

            with st.chat_message("user"):

                st.markdown(
                    prompt
                )

            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "AI กำลังวิเคราะห์..."
                ):

                    try:

                        client = openai.OpenAI(
                            api_key=openai_api_key
                        )

                        response = (
                            client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=
                                    st.session_state.messages
                            )
                        )

                        reply = (
                            response
                            .choices[0]
                            .message
                            .content
                        )

                        st.markdown(
                            reply
                        )

                        st.session_state.messages.append(
                            {
                                "role":
                                    "assistant",
                                "content":
                                    reply
                            }
                        )

                    except Exception as e:

                        st.error(
                            f"เกิดข้อผิดพลาด: {e}"
                        )

    else:

        st.info(
            " กรุณาใส่ OpenAI API Key "
            "ที่ Sidebar เพื่อเปิด AI Chat"
        )