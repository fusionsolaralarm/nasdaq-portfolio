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
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# THEME
# =========================================================

st.sidebar.header("🎨 ตั้งค่าการแสดงผล")

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
        signal = "ซื้อสะสม (Strong Buy)"
    elif score >= 45:
        signal = "ถือ / เฝ้าสังเกต (Hold)"
    else:
        signal = "ควรขาย / หลีกเลี่ยง (Sell / Avoid)"

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

        def fmt_money(value):

            if value is None:
                return "N/A"

            if abs(value) >= 1e12:
                return (
                    f"${value / 1e12:.2f}T"
                )

            if abs(value) >= 1e9:
                return (
                    f"${value / 1e9:.2f}B"
                )

            if abs(value) >= 1e6:
                return (
                    f"${value / 1e6:.2f}M"
                )

            return f"${value:,.0f}"

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

            "fmt_money":
                fmt_money
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# =========================================================
# SIDEBAR SCANNER
# =========================================================

st.sidebar.header(
    "⚙️ ตั้งค่าแหล่งข้อมูล & สแกนหุ้น"
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
        "🚀 เริ่มสแกนตลาดเต็มรูปแบบ",
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
            "👈 กรุณาคลิก "
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
        "📊 สแกนหุ้นตลาด",
        "📝 วิเคราะห์หุ้นรายตัว",
        "💼 พอร์ตของฉัน",
        "🤖 AI Portfolio Advisor"
    ]
)


# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.subheader(
        "📊 Quantitative Stock Screener"
    )

    col1, col2 = st.columns(2)

    with col1:

        signal_filter = st.selectbox(
            "กรองตามคำแนะนำ",
            [
                "ทั้งหมด",
                "ซื้อสะสม (Strong Buy)",
                "ถือ / เฝ้าสังเกต (Hold)",
                "ควรขาย / หลีกเลี่ยง (Sell / Avoid)"
            ]
        )

    with col2:

        search_query = st.text_input(
            "🔍 ค้นหา Ticker",
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

with tab2:

    st.subheader(
        "📝 วิเคราะห์หุ้นรายตัวแบบละเอียด"
    )

    st.write(
        "กรอก Ticker เพื่อดูธุรกิจ "
        "พื้นฐาน การเติบโต Valuation "
        "Technical Risk และแนวโน้มอนาคต"
    )

    inner_search_ticker = st.text_input(
        "🔍 Ticker หุ้น",
        "NVDA"
    ).upper().strip()

    if not inner_search_ticker:

        st.info(
            "กรุณากรอก Ticker"
        )

    else:

        match_dd = df[
            df["Ticker"]
            == inner_search_ticker
        ]

        if match_dd.empty:

            try:

                with st.spinner(
                    f"กำลังดึงข้อมูล "
                    f"{inner_search_ticker}..."
                ):

                    hist_dd = yf.download(
                        inner_search_ticker,
                        period="6mo",
                        progress=False,
                        auto_adjust=True
                    )

                    if not hist_dd.empty:

                        if isinstance(
                            hist_dd.columns,
                            pd.MultiIndex
                        ):

                            cs_dd = (
                                hist_dd[
                                    "Close"
                                ]
                                .iloc[:, 0]
                            )

                        else:

                            cs_dd = (
                                hist_dd[
                                    "Close"
                                ]
                            )

                        row_dd = (
                            compute_simons_row(
                                inner_search_ticker,
                                cs_dd
                            )
                        )

                        if row_dd:

                            match_dd = pd.DataFrame(
                                [row_dd]
                            )

            except Exception:
                pass

        if match_dd.empty:

            st.error(
                f"ไม่พบข้อมูล "
                f"{inner_search_ticker}"
            )

        else:

            r = match_dd.iloc[0]

            t_ticker = r["Ticker"]
            t_price = float(
                r["Price (USD)"]
            )
            t_score = int(
                r["Score"]
            )
            t_signal = r[
                "Simons Signal"
            ]
            t_trend = r[
                "Trend"
            ]
            t_mom1m = float(
                r["Momentum 1M (%)"]
            )
            t_mom1w = float(
                r["Momentum 1W (%)"]
            )
            t_vol = float(
                r["Volatility (%)"]
            )
            t_sma50 = float(
                r["SMA 50"]
            )

            analysis = (
                get_detailed_stock_analysis(
                    t_ticker
                )
            )

            if "error" in analysis:

                st.error(
                    "ไม่สามารถสร้างรายงาน "
                    f"ได้: {analysis['error']}"
                )

            else:

                data = analysis["data"]

                fmt_money = (
                    analysis["fmt_money"]
                )

                # =========================================
                # HEADER
                # =========================================

                st.markdown("---")

                st.title(
                    f"📊 {t_ticker}"
                )

                st.caption(
                    f"{data['long_name']} | "
                    f"{data['sector']} | "
                    f"{data['industry']}"
                )

                # =========================================
                # TOP METRICS
                # =========================================

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "ราคาปัจจุบัน",
                    f"${t_price:,.2f}"
                )

                c2.metric(
                    "Quant Score",
                    f"{t_score}/100"
                )

                c3.metric(
                    "Momentum 1M",
                    f"{t_mom1m:+.2f}%"
                )

                c4.metric(
                    "Volatility",
                    f"{t_vol:.2f}%"
                )

                # =========================================
                # BUSINESS
                # =========================================

                st.markdown("---")

                st.subheader(
                    "1. 🏢 บริษัททำธุรกิจอะไร?"
                )

                st.write(
                    data["summary"]
                )

                b1, b2, b3 = st.columns(3)

                b1.metric(
                    "Sector",
                    data["sector"]
                )

                b2.metric(
                    "Industry",
                    data["industry"]
                )

                b3.metric(
                    "Market Cap",
                    fmt_money(
                        data["market_cap"]
                    )
                )

                # =========================================
                # FINANCIAL
                # =========================================

                st.subheader(
                    "2. 💰 Financial Snapshot"
                )

                f1, f2, f3, f4 = st.columns(4)

                f1.metric(
                    "Revenue",
                    fmt_money(
                        data["revenue"]
                    )
                )

                f2.metric(
                    "Revenue Growth",
                    (
                        f"{data['revenue_growth'] * 100:.2f}%"
                        if data["revenue_growth"]
                        is not None
                        else "N/A"
                    )
                )

                f3.metric(
                    "Profit Margin",
                    (
                        f"{data['profit_margin'] * 100:.2f}%"
                        if data["profit_margin"]
                        is not None
                        else "N/A"
                    )
                )

                f4.metric(
                    "EPS",
                    (
                        f"${data['eps']:.2f}"
                        if data["eps"]
                        is not None
                        else "N/A"
                    )
                )

                f5, f6, f7, f8 = st.columns(4)

                f5.metric(
                    "Forward P/E",
                    (
                        f"{data['forward_pe']:.2f}x"
                        if data["forward_pe"]
                        is not None
                        else "N/A"
                    )
                )

                f6.metric(
                    "ROE",
                    (
                        f"{data['return_on_equity'] * 100:.2f}%"
                        if data["return_on_equity"]
                        is not None
                        else "N/A"
                    )
                )

                f7.metric(
                    "Debt / Equity",
                    (
                        f"{data['debt_to_equity']:.1f}"
                        if data["debt_to_equity"]
                        is not None
                        else "N/A"
                    )
                )

                f8.metric(
                    "Free Cash Flow",
                    fmt_money(
                        data["free_cash_flow"]
                    )
                )

                # =========================================
                # BUSINESS OUTLOOK
                # =========================================

                st.subheader(
                    "3. 🔮 แนวโน้มธุรกิจในอนาคต"
                )

                st.write(
                    f"""
                    การประเมิน {t_ticker}
                    ไม่ควรดูเฉพาะราคาหุ้น แต่ควรดูว่า
                    บริษัทสามารถเพิ่มรายได้ กำไร
                    กระแสเงินสด และ Market Share
                    ได้หรือไม่
                    """
                )

                st.markdown(
                    "### ปัจจัยที่จะสนับสนุนการเติบโต"
                )

                for item in analysis["outlook"]:

                    st.markdown(
                        f"- {item}"
                    )

                st.markdown(
                    "### มุมมองระยะกลาง"
                )

                st.info(
                    analysis["medium_term"]
                )

                st.markdown(
                    "### มุมมองระยะยาว"
                )

                st.info(
                    analysis["long_term"]
                )

                # =========================================
                # GROWTH ENGINE
                # =========================================

                st.subheader(
                    "4. 🚀 Growth Engine"
                )

                g1, g2, g3 = st.columns(3)

                if data["revenue_growth"] is not None:

                    g1.metric(
                        "Revenue Growth",
                        f"{data['revenue_growth'] * 100:+.2f}%"
                    )

                else:

                    g1.metric(
                        "Revenue Growth",
                        "N/A"
                    )

                if data["earnings_growth"] is not None:

                    g2.metric(
                        "Earnings Growth",
                        f"{data['earnings_growth'] * 100:+.2f}%"
                    )

                else:

                    g2.metric(
                        "Earnings Growth",
                        "N/A"
                    )

                g3.metric(
                    "Free Cash Flow",
                    fmt_money(
                        data["free_cash_flow"]
                    )
                )

                st.write(
                    f"""
                    **Growth Score:** 
                    {analysis['growth_score']}
                    
                    คะแนนนี้ใช้เพื่อประเมินเบื้องต้นว่า
                    Revenue และ Earnings
                    มีแนวโน้มเติบโตหรือไม่
                    """
                )

                # =========================================
                # PROFITABILITY
                # =========================================

                st.subheader(
                    "5. 📈 คุณภาพของธุรกิจและกำไร"
                )

                q1, q2, q3, q4 = st.columns(4)

                q1.metric(
                    "Gross Margin",
                    (
                        f"{data['gross_margin'] * 100:.2f}%"
                        if data["gross_margin"]
                        is not None
                        else "N/A"
                    )
                )

                q2.metric(
                    "Operating Margin",
                    (
                        f"{data['operating_margin'] * 100:.2f}%"
                        if data["operating_margin"]
                        is not None
                        else "N/A"
                    )
                )

                q3.metric(
                    "Profit Margin",
                    (
                        f"{data['profit_margin'] * 100:.2f}%"
                        if data["profit_margin"]
                        is not None
                        else "N/A"
                    )
                )

                q4.metric(
                    "ROA",
                    (
                        f"{data['return_on_assets'] * 100:.2f}%"
                        if data["return_on_assets"]
                        is not None
                        else "N/A"
                    )
                )

                st.write(
                    f"""
                    **Quality Score:** 
                    {analysis['quality_score']}
                    
                    คะแนนสูงหมายถึงบริษัทมี
                    Margin และ ROE
                    ที่ค่อนข้างแข็งแรง
                    """
                )

                # =========================================
                # CATALYST
                # =========================================

                st.subheader(
                    "6. ⚡ Catalyst"
                )

                st.write(
                    "ปัจจัยที่อาจทำให้รายได้ กำไร "
                    "หรือ Valuation ของบริษัทดีขึ้น:"
                )

                for item in analysis[
                    "catalysts"
                ]:

                    st.markdown(
                        f"- **{item}**"
                    )

                # =========================================
                # RISK
                # =========================================

                st.subheader(
                    "7. ⚠️ ความเสี่ยง"
                )

                for item in analysis["risks"]:

                    st.markdown(
                        f"- **{item}**"
                    )

                # =========================================
                # VALUATION
                # =========================================

                st.subheader(
                    "8. 💵 Valuation"
                )

                v1, v2, v3, v4 = st.columns(4)

                v1.metric(
                    "P/E",
                    (
                        f"{data['pe']:.2f}x"
                        if data["pe"]
                        is not None
                        else "N/A"
                    )
                )

                v2.metric(
                    "Forward P/E",
                    (
                        f"{data['forward_pe']:.2f}x"
                        if data["forward_pe"]
                        is not None
                        else "N/A"
                    )
                )

                v3.metric(
                    "PEG",
                    (
                        f"{data['peg']:.2f}"
                        if data["peg"]
                        is not None
                        else "N/A"
                    )
                )

                v4.metric(
                    "P/S",
                    (
                        f"{data['price_to_sales']:.2f}x"
                        if data["price_to_sales"]
                        is not None
                        else "N/A"
                    )
                )

                st.info(
                    analysis["valuation_text"]
                )

                # =========================================
                # ANALYST TARGET
                # =========================================

                st.subheader(
                    "9. 🎯 Analyst Target Price"
                )

                a1, a2, a3 = st.columns(3)

                a1.metric(
                    "Target Low",
                    (
                        f"${data['target_low']:,.2f}"
                        if data["target_low"]
                        is not None
                        else "N/A"
                    )
                )

                a2.metric(
                    "Target Mean",
                    (
                        f"${data['target_mean']:,.2f}"
                        if data["target_mean"]
                        is not None
                        else "N/A"
                    )
                )

                a3.metric(
                    "Target High",
                    (
                        f"${data['target_high']:,.2f}"
                        if data["target_high"]
                        is not None
                        else "N/A"
                    )
                )

                # =========================================
                # TECHNICAL
                # =========================================

                st.subheader(
                    "10. 📈 Technical & Momentum"
                )

                tc1, tc2, tc3, tc4 = st.columns(4)

                tc1.metric(
                    "1 Week",
                    f"{t_mom1w:+.2f}%"
                )

                tc2.metric(
                    "1 Month",
                    f"{t_mom1m:+.2f}%"
                )

                tc3.metric(
                    "SMA 50",
                    f"${t_sma50:,.2f}"
                )

                tc4.metric(
                    "Trend",
                    t_trend
                )

                if t_price > t_sma50:

                    st.success(
                        "ราคาปัจจุบันอยู่เหนือ "
                        "SMA 50 — Momentum ระยะกลางเป็นบวก"
                    )

                else:

                    st.warning(
                        "ราคาปัจจุบันต่ำกว่า "
                        "SMA 50 — Momentum ระยะกลางอ่อนตัว"
                    )

                # =========================================
                # 52 WEEK
                # =========================================

                st.subheader(
                    "11. 📊 52 Week Range"
                )

                low52 = data[
                    "fifty_two_low"
                ]

                high52 = data[
                    "fifty_two_high"
                ]

                if (
                    low52 is not None
                    and high52 is not None
                    and high52 > low52
                ):

                    position = (
                        (
                            t_price - low52
                        )
                        /
                        (
                            high52 - low52
                        )
                        * 100
                    )

                    position = max(
                        0,
                        min(
                            100,
                            position
                        )
                    )

                    st.progress(
                        position / 100
                    )

                    st.caption(
                        f"52W Low: "
                        f"${low52:,.2f} | "
                        f"Current: "
                        f"${t_price:,.2f} | "
                        f"52W High: "
                        f"${high52:,.2f}"
                    )

                    st.write(
                        f"ราคาปัจจุบันอยู่ที่ "
                        f"**{position:.1f}%** "
                        f"ของกรอบราคา 52 สัปดาห์"
                    )

                # =========================================
                # BULL BASE BEAR
                # =========================================

                st.subheader(
                    "12. 🧠 Bull / Base / Bear Scenario"
                )

                bull, base, bear = st.columns(3)

                with bull:

                    st.success(
                        """
                        **Bull Case**

                        • Revenue Growth สูง  
                        • Earnings Growth สูง  
                        • Margin ขยายตัว  
                        • Market Share เพิ่ม  
                        • อุตสาหกรรมเติบโตแรง  
                        • นักลงทุนยอมให้ Premium Valuation
                        """
                    )

                with base:

                    st.info(
                        """
                        **Base Case**

                        • Revenue โตตามอุตสาหกรรม  
                        • Margin ทรงตัว  
                        • Earnings โตปานกลาง  
                        • Valuation ไม่เปลี่ยนมาก  
                        • ราคาหุ้นเคลื่อนไหวตามพื้นฐาน
                        """
                    )

                with bear:

                    st.error(
                        """
                        **Bear Case**

                        • Revenue Growth ชะลอตัว  
                        • Earnings ลดลง  
                        • Margin หดตัว  
                        • คู่แข่งเพิ่มขึ้น  
                        • Valuation ลดลง  
                        • ราคาหุ้นเกิดแรงขาย
                        """
                    )

                # =========================================
                # FINAL VERDICT
                # =========================================

                st.subheader(
                    "13. 🎯 Final Investment Verdict"
                )

                if t_score >= 70:

                    st.success(
                        f"""
                        🟢 **STRONG BUY / ACCUMULATE**

                        {t_ticker} ได้ Quant Score
                        **{t_score}/100**

                        Momentum และโครงสร้างราคามีสัญญาณเชิงบวก
                        เหมาะสำหรับพิจารณาทยอยสะสม

                        อย่างไรก็ตามควรตรวจสอบ Valuation
                        และผลประกอบการล่าสุดก่อนเพิ่มน้ำหนัก
                        """
                    )

                elif t_score >= 45:

                    st.warning(
                        f"""
                        🟡 **HOLD / WAIT & SEE**

                        {t_ticker} ได้ Quant Score
                        **{t_score}/100**

                        ยังไม่มีสัญญาณที่แข็งแรงเพียงพอ
                        สำหรับการเพิ่มน้ำหนักอย่าง aggressive

                        หากมีหุ้นอยู่แล้วสามารถถือและติดตาม
                        ผลประกอบการต่อได้
                        """
                    )

                else:

                    st.error(
                        f"""
                        🔴 **AVOID / REDUCE RISK**

                        {t_ticker} ได้ Quant Score
                        **{t_score}/100**

                        โครงสร้างทาง Quant ยังอ่อนแอ
                        ควรเน้นการควบคุมความเสี่ยง
                        และรอสัญญาณฟื้นตัว
                        """

                    )

                # =========================================
                # PRICE CHART
                # =========================================

                st.subheader(
                    f"14. 📉 ราคาย้อนหลัง 6 เดือน — {t_ticker}"
                )

                try:

                    hist_chart = yf.download(
                        t_ticker,
                        period="6mo",
                        progress=False,
                        auto_adjust=True
                    )

                    if not hist_chart.empty:

                        if isinstance(
                            hist_chart.columns,
                            pd.MultiIndex
                        ):

                            close_c = (
                                hist_chart[
                                    "Close"
                                ]
                                .iloc[:, 0]
                            )

                        else:

                            close_c = (
                                hist_chart[
                                    "Close"
                                ]
                            )

                        st.line_chart(
                            close_c
                        )

                except Exception:

                    st.warning(
                        "ไม่สามารถสร้างกราฟได้"
                    )

                st.caption(
                    "หมายเหตุ: รายงานนี้ใช้ข้อมูลจาก "
                    "Yahoo Finance และโมเดลเชิงปริมาณ "
                    "ข้อมูลทางการเงินและราคาอาจเปลี่ยนแปลงได้ "
                    "ไม่ควรใช้เป็นคำแนะนำการลงทุนเพียงแหล่งเดียว"
                )


# =========================================================
# TAB 3 PORTFOLIO
# =========================================================

with tab3:

    st.subheader(
        "💼 พอร์ตการลงทุนของคุณ"
    )

    st.write(
        "กรอกข้อมูลพอร์ต หรืออัปโหลดใบเสร็จ "
        "เพื่อให้ AI อ่านข้อมูลหุ้น"
    )

    with st.expander(
        "📷 AI OCR — อ่านใบเสร็จซื้อขายหุ้น",
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
            "🔑 OpenAI API Key สำหรับ OCR",
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
                "🚀 สแกนใบเสร็จ"
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
        "🗑️ ล้างพอร์ตเป็นค่าเริ่มต้น"
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
                "💰 เงินลงทุน",
                f"${total_invested:,.2f}"
            )

            m2.metric(
                "📈 มูลค่าปัจจุบัน",
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
                "💡 Quantitative Action Summary"
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
                    "### 🟢 ซื้อเพิ่ม"
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
                    "### 🔴 ลดความเสี่ยง"
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
                "📋 สรุปพอร์ต"
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
                "📥 ดาวน์โหลด Portfolio CSV",
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
        "🤖 AI Portfolio Advisor"
    )

    openai_api_key = st.sidebar.text_input(
        "🔑 OpenAI API Key สำหรับ AI Chat",
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
            "💡 กรุณาใส่ OpenAI API Key "
            "ที่ Sidebar เพื่อเปิด AI Chat"
        )