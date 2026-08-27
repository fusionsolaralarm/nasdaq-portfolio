import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from google import genai
from google.genai import types
import json
import os

st.set_page_config(
    page_title="Simon Screener & Portfolio AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

PORTFOLIO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio_data.json")
DEFAULT_PORTFOLIO = pd.DataFrame([
    {"Ticker": "AAPL", "Shares": 10.0, "Buy Price (USD)": 170.0, "Asset Class": "Growth"},
    {"Ticker": "TSLA", "Shares": 5.0, "Buy Price (USD)": 220.0, "Asset Class": "Growth"}
])


# =====================================================================
# PORTFOLIO PERSISTENCE
# =====================================================================

def load_portfolio_from_disk():
    """โหลดพอร์ตที่เคยบันทึกไว้จากไฟล์บนดิสก์ ถ้าไม่เคยมีให้คืนค่าตัวอย่างเริ่มต้น"""
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                records = json.load(f)
            loaded = pd.DataFrame(records)
            if not loaded.empty and {'Ticker', 'Shares', 'Buy Price (USD)'}.issubset(loaded.columns):
                if 'Asset Class' not in loaded.columns:
                    loaded['Asset Class'] = 'Growth'
                return loaded[['Ticker', 'Shares', 'Buy Price (USD)', 'Asset Class']]
        except Exception:
            pass
    return DEFAULT_PORTFOLIO.copy()


def save_portfolio_to_disk(df):
    """บันทึกพอร์ตปัจจุบันลงไฟล์ ครั้งถัดไปที่เปิดแอปจะโหลดข้อมูลเดิมกลับมาให้อัตโนมัติ"""
    try:
        df.to_json(PORTFOLIO_FILE, orient="records", force_ascii=False)
    except Exception as e:
        st.sidebar.error(f"บันทึกพอร์ตลงดิสก์ไม่สำเร็จ: {e}")


# =====================================================================
# HEADER
# =====================================================================

st.title("Simon Screener - Quant & Portfolio AI Advisor")
st.caption("ระบบสแกนหุ้นสหรัฐฯ วิเคราะห์หุ้นรายตัว และผู้ช่วยวิเคราะห์พอร์ตการลงทุนส่วนตัวด้วยโมเดลสถิติเชิงปริมาณ")
st.divider()


# =====================================================================
# SYMBOL LISTS
# =====================================================================

@st.cache_data
def get_curated_symbols():
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX", "AMD", "INTC",
        "CRM", "ADBE", "ORCL", "IBM", "NOW", "SNOW", "PLTR", "DDOG", "CRWD",
        "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "PYPL",
        "WMT", "COST", "TGT", "HD", "NKE", "SBUX", "MCD", "KO", "PEP", "DIS",
        "UNH", "JNJ", "PFE", "ABBV", "LLY", "MRK", "AMGN", "TMO",
        "XOM", "CVX", "CAT", "BA", "GE", "HON", "UPS", "LMT",
        "ASTS", "LUNR", "IONQ", "SOUN", "RGTI", "QBTS", "ACHR", "JOBY",
        "RIVN", "HOOD", "ARM", "SMCI", "RDDT", "DJT", "BBAI",
        "HLGN", "CLSK", "RIOT", "MARA", "HUT", "BITF", "OPEN", "LMND",
        "AVGO", "ACN", "CSCO", "TXN", "QCOM", "AMAT", "INTU", "BKNG", "ISRG"
    ]


@st.cache_data(ttl=86400)
def get_full_us_market_symbols():
    """ดึงรายชื่อหุ้น/หลักทรัพย์ที่จดทะเบียนทั้งหมดในตลาดสหรัฐฯ (NASDAQ + NYSE + AMEX)
    จากไฟล์ทางการของ Nasdaq Trader คืนค่าเป็น DataFrame คอลัมน์ Symbol, Name, ETF
    ถ้าดึงไม่สำเร็จจะ fallback ไปใช้รายชื่อคัดสรรแทน"""
    try:
        nasdaq_url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
        other_url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

        nasdaq_df = pd.read_csv(nasdaq_url, sep='|')
        nasdaq_df = nasdaq_df.iloc[:-1]
        nasdaq_df = nasdaq_df[nasdaq_df['Test Issue'] == 'N']
        nasdaq_df = nasdaq_df.rename(columns={'Security Name': 'Name'})[['Symbol', 'Name', 'ETF']]

        other_df = pd.read_csv(other_url, sep='|')
        other_df = other_df.iloc[:-1]
        other_df = other_df[other_df['Test Issue'] == 'N']
        other_df = other_df.rename(columns={'ACT Symbol': 'Symbol', 'Security Name': 'Name'})[['Symbol', 'Name', 'ETF']]

        combined = pd.concat([nasdaq_df, other_df], ignore_index=True)
        combined['ETF'] = combined['ETF'].astype(str).str.upper() == 'Y'
        combined['Symbol'] = combined['Symbol'].astype(str).str.strip()
        combined = combined[combined['Symbol'].str.match(r'^[A-Za-z.]+$', na=False)]
        combined['Symbol'] = combined['Symbol'].str.replace('.', '-', regex=False)
        combined = combined.drop_duplicates(subset='Symbol').sort_values('Symbol').reset_index(drop=True)

        if combined.empty:
            raise ValueError("รายชื่อที่ได้ว่างเปล่า")
        return combined
    except Exception as e:
        st.sidebar.warning(f"ดึงรายชื่อหุ้นทั้งตลาดไม่สำเร็จ ({e}) — ใช้รายชื่อคัดสรรแทนชั่วคราว")
        curated = get_curated_symbols()
        return pd.DataFrame({'Symbol': curated, 'Name': curated, 'ETF': False})


# =====================================================================
# QUANT MODEL
# =====================================================================

def compute_simons_row(ticker, close_series):
    """คำนวณคะแนนและคำแนะนำสไตล์ Simons จากราคาปิดย้อนหลังของหุ้นหนึ่งตัว"""
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
        recommendation = "ซื้อสะสม (Strong Buy)"
    elif score >= 45:
        recommendation = "ถือ / เฝ้าสังเกต (Hold)"
    else:
        recommendation = "ควรขาย / หลีกเลี่ยง (Sell / Avoid)"

    if return_1w > 0.5:
        trend = "ขาขึ้น"
    elif return_1w < -0.5:
        trend = "ขาลง"
    else:
        trend = "ไซด์เวย์"

    return {
        'Ticker': ticker,
        'Simons Signal': recommendation,
        'Score': score,
        'Price (USD)': current_p,
        'Momentum 1M (%)': return_1m,
        'Momentum 1W (%)': return_1w,
        'Trend': trend,
        'Volatility (%)': volatility,
        'SMA 50': sma_50
    }


@st.cache_data(ttl=1800)
def fetch_stock_data_and_simons_logic(ticker_list):
    """โหมดคัดสรร: ดึงทีละตัว เหมาะกับลิสต์สั้น ๆ เท่านั้น"""
    data_list = []
    for ticker in ticker_list:
        try:
            hist = yf.download(ticker, period="6mo", progress=False)
            if hist.empty:
                continue
            close_series = hist['Close'].iloc[:, 0] if isinstance(hist.columns, pd.MultiIndex) else hist['Close']
            row = compute_simons_row(ticker, close_series)
            if row:
                data_list.append(row)
        except Exception:
            continue
    return pd.DataFrame(data_list)


def fetch_stock_data_batched(ticker_list, chunk_size=50):
    """โหมดสแกนเต็มตลาด: ดึงเป็นชุดพร้อม progress bar"""
    all_rows = []
    total = len(ticker_list)
    progress_bar = st.progress(0.0, text="เริ่มสแกน...")
    for i in range(0, total, chunk_size):
        chunk = ticker_list[i:i + chunk_size]
        try:
            data = yf.download(chunk, period="6mo", group_by='ticker', threads=True, progress=False, auto_adjust=True)
        except Exception:
            data = None

        if data is not None and not data.empty:
            for ticker in chunk:
                try:
                    if len(chunk) == 1 or not isinstance(data.columns, pd.MultiIndex):
                        close_series = data['Close']
                    else:
                        if ticker not in data.columns.get_level_values(0):
                            continue
                        close_series = data[ticker]['Close']
                    row = compute_simons_row(ticker, close_series)
                    if row:
                        all_rows.append(row)
                except Exception:
                    continue

        done = min(i + chunk_size, total)
        progress_bar.progress(done / total, text=f"สแกนแล้ว {done:,}/{total:,} ตัว (พบข้อมูลใช้ได้ {len(all_rows):,} ตัว)")

    progress_bar.empty()
    return pd.DataFrame(all_rows)


# =====================================================================
# COMPANY RESEARCH HELPERS (used by the deep-dive tab)
# =====================================================================

@st.cache_data(ttl=1800)
def get_company_fundamentals(ticker_symbol):
    """ดึงข้อมูลบริษัทและปัจจัยพื้นฐานจาก Yahoo Finance คืนค่าเป็น dict เสมอ"""
    try:
        info = yf.Ticker(ticker_symbol).info
    except Exception:
        info = {}

    def num(key):
        v = info.get(key)
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    return {
        'long_name': info.get('longName', ticker_symbol),
        'sector': info.get('sector', 'N/A'),
        'industry': info.get('industry', 'N/A'),
        'summary': info.get('longBusinessSummary', 'ไม่มีข้อมูลสรุปธุรกิจ'),
        'market_cap': num('marketCap'),
        'trailing_pe': num('trailingPE'),
        'forward_pe': num('forwardPE'),
        'peg': num('pegRatio'),
        'profit_margins': num('profitMargins'),
        'gross_margins': num('grossMargins'),
        'revenue_growth': num('revenueGrowth'),
        'earnings_growth': num('earningsGrowth'),
        'debt_to_equity': num('debtToEquity'),
        'return_on_equity': num('returnOnEquity'),
        'free_cash_flow': num('freeCashflow'),
        'target_mean_price': num('targetMeanPrice'),
        'target_low_price': num('targetLowPrice'),
        'target_high_price': num('targetHighPrice'),
        'recommendation_key': info.get('recommendationKey'),
        'dividend_yield': num('dividendYield'),
        'beta': num('beta'),
        'fifty_two_high': num('fiftyTwoWeekHigh'),
        'fifty_two_low': num('fiftyTwoWeekLow'),
    }


def format_money(value):
    if value is None:
        return "ไม่มีข้อมูล"
    try:
        value = float(value)
    except Exception:
        return "ไม่มีข้อมูล"
    if abs(value) >= 1e12:
        return f"${value / 1e12:.2f}T"
    if abs(value) >= 1e9:
        return f"${value / 1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:.2f}M"
    return f"${value:,.0f}"


def format_pct(value):
    return "ไม่มีข้อมูล" if value is None else f"{value * 100:+.2f}%"


def format_num(value, suffix=""):
    return "ไม่มีข้อมูล" if value is None else f"{value:.2f}{suffix}"


def get_sector_outlook(sector, industry, ticker):
    """คืนค่าย่อหน้าแนวโน้มธุรกิจในอนาคต แยกตามกลุ่มอุตสาหกรรม"""
    s = str(sector).lower()
    i = str(industry).lower()

    if "technology" in s or "semiconductor" in i:
        return (
            f"บริษัท {ticker} อยู่ในกลุ่มเทคโนโลยี/เซมิคอนดักเตอร์ ซึ่งเป็นหัวใจของโครงสร้างพื้นฐาน AI ในปัจจุบัน "
            "แนวโน้มได้แรงหนุนจากงบลงทุนของกลุ่ม Cloud Hyperscalers และการนำ AI ไปใช้ในระดับองค์กร "
            "ความเสี่ยงคือวัฏจักรความต้องการชิปและมาตรการควบคุมการส่งออกเทคโนโลยี"
        )
    if "financial" in s:
        return (
            f"บริษัท {ticker} อยู่ในกลุ่มการเงิน ผลประกอบการผูกกับทิศทางดอกเบี้ยนโยบายและคุณภาพสินเชื่อ "
            "แนวโน้มขึ้นอยู่กับวัฏจักรดอกเบี้ยและการขยายสู่บริการการเงินดิจิทัล ความเสี่ยงคือหนี้เสียในภาวะเศรษฐกิจถดถอย"
        )
    if "energy" in s:
        return (
            f"บริษัท {ticker} อยู่ในกลุ่มพลังงาน ผลประกอบการผันผวนตามราคาน้ำมัน/ก๊าซและนโยบาย OPEC+ "
            "แนวโน้มขึ้นอยู่กับสมดุลอุปสงค์-อุปทานพลังงานโลก ความเสี่ยงคือกฎระเบียบสิ่งแวดล้อมและราคาโภคภัณฑ์ที่ควบคุมไม่ได้"
        )
    if "industrial" in s:
        return (
            f"บริษัท {ticker} อยู่ในกลุ่มอุตสาหกรรมการผลิต เติบโตตามวัฏจักรการลงทุนภาครัฐและเอกชน "
            "แนวโน้มได้แรงหนุนจากการนำระบบอัตโนมัติมาใช้และการย้ายฐานการผลิตกลับประเทศ ความเสี่ยงคือต้นทุนวัตถุดิบและห่วงโซ่อุปทาน"
        )
    if "communication" in s:
        return (
            f"บริษัท {ticker} อยู่ในกลุ่มสื่อสาร/สื่อดิจิทัล รายได้ผูกกับปริมาณการใช้งานข้อมูลและโฆษณาดิจิทัล "
            "แนวโน้มขึ้นอยู่กับการขยายฐานผู้ใช้และการลงทุนโครงข่าย ความเสี่ยงคือการแข่งขันและกฎระเบียบความเป็นส่วนตัว"
        )
    if "utilities" in s:
        return (
            f"บริษัท {ticker} อยู่ในกลุ่มสาธารณูปโภค รายได้ค่อนข้างสม่ำเสมอแต่อ่อนไหวต่ออัตราดอกเบี้ยเนื่องจากใช้เงินทุนสูง "
            "เหมาะกับนักลงทุนที่เน้นเงินปันผลมากกว่าการเติบโตสูง"
        )
    if "real estate" in s:
        return (
            f"บริษัท {ticker} อยู่ในกลุ่มอสังหาริมทรัพย์/REIT ไวต่อทิศทางอัตราดอกเบี้ยและต้นทุนการกู้ยืมอย่างมาก "
            "แนวโน้มขึ้นอยู่กับอัตราการเช่าและวัฏจักรดอกเบี้ย"
        )
    if "consumer" in s:
        return (
            f"บริษัท {ticker} อยู่ในกลุ่มสินค้าอุปโภคบริโภค เชื่อมโยงกับกำลังซื้อผู้บริโภคและเงินเฟ้อ "
            "แนวโน้มขึ้นอยู่กับอำนาจการตั้งราคาและการปรับตัวสู่ช่องทางดิจิทัล"
        )
    if "health" in s:
        return (
            f"บริษัท {ticker} อยู่ในกลุ่มเทคโนโลยีชีวภาพและสุขภาพ ขับเคลื่อนด้วยสังคมผู้สูงอายุและนวัตกรรมการแพทย์เฉพาะบุคคล "
            "ความเสี่ยงคือกฎระเบียบ การควบคุมราคายา และผลการทดลองทางคลินิก"
        )
    if "material" in s:
        return (
            f"บริษัท {ticker} อยู่ในกลุ่มวัตถุดิบพื้นฐาน ราคาสินค้าโภคภัณฑ์และความต้องการจากภาคอุตสาหกรรมโลกเป็นตัวขับเคลื่อนหลัก "
            "มีความผันผวนสูงตามวัฏจักรสินค้าโภคภัณฑ์"
        )
    return (
        f"บริษัท {ticker} ในกลุ่มอุตสาหกรรม {sector} มีทิศทางเติบโตสอดคล้องกับภาพรวมเศรษฐกิจมหภาค "
        "แนวโน้มขึ้นอยู่กับการบริหารต้นทุนห่วงโซ่อุปทานและการปรับตัวรับความผันผวนของอัตราแลกเปลี่ยนและดอกเบี้ยโลก"
    )


# =====================================================================
# SIDEBAR — SCAN SCOPE
# =====================================================================

st.sidebar.header("ตั้งค่าแหล่งข้อมูล & สแกนหุ้น")
scan_mode = st.sidebar.radio(
    "เลือกแหล่งหุ้น",
    ["รายชื่อคัดสรร (~85 ตัว, เร็ว)", "ทั้งตลาดหุ้นสหรัฐฯ (NASDAQ + NYSE + AMEX)"]
)

if scan_mode.startswith("รายชื่อคัดสรร"):
    symbols = get_curated_symbols()
    with st.spinner("กำลังดึงราคาล่าสุดและคำนวณโมเดลตลาดสหรัฐฯ..."):
        df = fetch_stock_data_and_simons_logic(symbols)
else:
    with st.spinner("กำลังดึงรายชื่อหุ้นทั้งหมดที่จดทะเบียนในตลาดสหรัฐฯ..."):
        full_list_df = get_full_us_market_symbols()

    st.sidebar.success(f"พบหุ้น/หลักทรัพย์ทั้งหมด {len(full_list_df):,} ตัว")
    exclude_etf = st.sidebar.checkbox("ไม่รวมกองทุน ETF (แนะนำ)", value=True)
    universe_df = full_list_df[~full_list_df['ETF']] if exclude_etf else full_list_df

    st.sidebar.warning(
        "ตลาดสหรัฐฯ มีหลักทรัพย์จดทะเบียนหลักพันถึงหลักหมื่นตัว การสแกนทุกตัวผ่าน Yahoo Finance "
        "อาจใช้เวลานานและเสี่ยงโดน rate-limit แนะนำจำกัดจำนวนต่อรอบแล้วขยับ offset สแกนทีละล็อต"
    )
    max_scan = st.sidebar.number_input(
        "จำนวนหุ้นสูงสุดที่จะสแกนในรอบนี้",
        min_value=50, max_value=int(len(universe_df)),
        value=min(300, int(len(universe_df))), step=50
    )
    start_offset = st.sidebar.number_input(
        "เริ่มสแกนจากลำดับที่ (offset)",
        min_value=0, max_value=max(0, int(len(universe_df)) - 1), value=0, step=max_scan
    )
    run_scan = st.sidebar.button("เริ่มสแกน", use_container_width=True)

    if run_scan:
        target_symbols = universe_df['Symbol'].iloc[start_offset:start_offset + max_scan].tolist()
        st.session_state.full_scan_df = fetch_stock_data_batched(target_symbols)
        st.session_state.full_scan_range = (start_offset, start_offset + len(target_symbols))

    if "full_scan_df" in st.session_state:
        df = st.session_state.full_scan_df
        rng = st.session_state.get("full_scan_range", (0, len(df)))
        st.sidebar.caption(f"ผลสแกนล่าสุด: ลำดับที่ {rng[0]:,}-{rng[1]:,} จากทั้งหมด {len(universe_df):,} ตัว")
    else:
        st.info("เลือกโหมด 'ทั้งตลาดหุ้นสหรัฐฯ' แล้วกดปุ่ม 'เริ่มสแกน' ที่แถบด้านซ้ายเพื่อเริ่มดึงข้อมูล")
        st.stop()

if df.empty:
    st.error("ไม่สามารถดึงข้อมูลได้ในขณะนี้ กรุณารีเฟรชหน้าเว็บใหม่ หรือลองลดจำนวนหุ้นที่สแกนต่อรอบ")
    st.stop()


# =====================================================================
# MAIN TABS
# =====================================================================

tab1, tab2, tab3, tab4 = st.tabs(
    ["สแกนหุ้นตลาด", "วิเคราะห์หุ้นรายตัว", "พอร์ตของฉัน", "คุยกับ AI Jim Simons"]
)

# ---------------------------------------------------------------------
# TAB 1 — MARKET SCANNER
# ---------------------------------------------------------------------
with tab1:
    st.subheader("ตารางสแกนหุ้นสหรัฐฯ เชิงปริมาณ")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        signal_filter = st.selectbox(
            "กรองตามคำแนะนำโมเดล",
            ["ทั้งหมด", "ซื้อสะสม (Strong Buy)", "ถือ / เฝ้าสังเกต (Hold)", "ควรขาย / หลีกเลี่ยง (Sell / Avoid)"]
        )
    with col_f2:
        search_query = st.text_input("ค้นหา Ticker หุ้น", "").upper()

    df_filtered = df if signal_filter == "ทั้งหมด" else df[df['Simons Signal'] == signal_filter]
    if search_query:
        df_filtered = df_filtered[df_filtered['Ticker'].str.contains(search_query)]

    st.dataframe(
        df_filtered.sort_values(by='Score', ascending=False),
        column_config={
            "Price (USD)": st.column_config.NumberColumn(format="$%.2f"),
            "Momentum 1M (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Momentum 1W (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Volatility (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Score": st.column_config.NumberColumn(format="%d คะแนน"),
        },
        use_container_width=True,
        hide_index=True
    )

# ---------------------------------------------------------------------
# TAB 2 — SINGLE STOCK DEEP-DIVE
# ---------------------------------------------------------------------
with tab2:
    st.subheader("วิเคราะห์หุ้นรายตัวเชิงลึก")
    st.write("พิมพ์ Ticker เพื่อดูภาพรวมธุรกิจ ปัจจัยพื้นฐาน แนวโน้มอนาคต และคำแนะนำการซื้อเข้าพอร์ต")

    inner_search_ticker = st.text_input("Ticker หุ้นที่ต้องการวิเคราะห์ (เช่น NVDA, TSLA, AAPL, PLTR)", "NVDA").upper().strip()

    if not inner_search_ticker:
        st.info("กรุณากรอก Ticker หุ้นที่ต้องการวิเคราะห์")
    else:
        match_dd = df[df['Ticker'] == inner_search_ticker]
        if match_dd.empty:
            try:
                with st.spinner(f"กำลังดึงข้อมูลราคาของ {inner_search_ticker}..."):
                    hist_dd = yf.download(inner_search_ticker, period="6mo", progress=False)
                    if not hist_dd.empty:
                        cs_dd = hist_dd['Close'].iloc[:, 0] if isinstance(hist_dd.columns, pd.MultiIndex) else hist_dd['Close']
                        row_dd = compute_simons_row(inner_search_ticker, cs_dd)
                        if row_dd:
                            match_dd = pd.DataFrame([row_dd])
            except Exception:
                pass

        if match_dd.empty:
            st.error(f"ไม่พบข้อมูลสำหรับ Ticker: {inner_search_ticker} กรุณาตรวจสอบความถูกต้องของชื่อหุ้นอีกครั้ง")
        else:
            r = match_dd.iloc[0]
            t_ticker = r['Ticker']
            t_price = float(r['Price (USD)'])
            t_score = int(r['Score'])
            t_signal = r['Simons Signal']
            t_trend = r['Trend']
            t_mom1m = float(r['Momentum 1M (%)'])
            t_mom1w = float(r['Momentum 1W (%)'])
            t_vol = float(r['Volatility (%)'])
            t_sma50 = float(r['SMA 50'])

            with st.spinner(f"กำลังดึงข้อมูลบริษัทของ {t_ticker}..."):
                fnd = get_company_fundamentals(t_ticker)

            st.markdown(f"## {fnd['long_name']} ({t_ticker})")
            st.caption(f"กลุ่มธุรกิจ: {fnd['sector']}  |  อุตสาหกรรม: {fnd['industry']}  |  ราคาล่าสุด: ${t_price:,.2f}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Simons Score", f"{t_score}/100")
            c2.metric("เทรนด์ระยะสั้น", t_trend)
            c3.metric("โมเมนตัม 1 เดือน", f"{t_mom1m:+.2f}%")
            c4.metric("ความผันผวนรายปี", f"{t_vol:.1f}%")

            st.markdown("#### 1. ภาพรวมธุรกิจและแนวโน้มในอนาคต")
            with st.container(border=True):
                st.markdown("**ภาพรวมการประกอบธุรกิจ (จาก Yahoo Finance)**")
                st.text(fnd['summary'])
                st.markdown("**แนวโน้มธุรกิจในอนาคต**")
                st.write(get_sector_outlook(fnd['sector'], fnd['industry'], t_ticker))

            st.markdown("#### 2. ปัจจัยพื้นฐาน (Fundamentals)")
            f1, f2, f3 = st.columns(3)
            f1.metric("มูลค่าบริษัท (Market Cap)", format_money(fnd['market_cap']))
            f1.metric("P/E ปัจจุบัน", format_num(fnd['trailing_pe']))
            f2.metric("P/E คาดการณ์ (Forward)", format_num(fnd['forward_pe']))
            f2.metric("PEG Ratio", format_num(fnd['peg']))
            f3.metric("อัตรากำไรสุทธิ", format_pct(fnd['profit_margins']))
            f3.metric("การเติบโตรายได้ (YoY)", format_pct(fnd['revenue_growth']))

            f4, f5, f6 = st.columns(3)
            f4.metric("ROE", format_pct(fnd['return_on_equity']))
            f5.metric("Debt / Equity", format_num(fnd['debt_to_equity']))
            f6.metric("Free Cash Flow", format_money(fnd['free_cash_flow']))
            st.caption(f"มุมมองเฉลี่ยของนักวิเคราะห์ (Wall Street Consensus): {fnd['recommendation_key'] or 'ไม่มีข้อมูล'}")

            st.markdown("#### 3. เป้าราคานักวิเคราะห์ & กรอบ 52 สัปดาห์")
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("เป้าราคาต่ำ", format_money(fnd['target_low_price']))
            t2.metric("เป้าราคาเฉลี่ย", format_money(fnd['target_mean_price']))
            t3.metric("เป้าราคาสูง", format_money(fnd['target_high_price']))
            if fnd['target_mean_price'] and t_price > 0:
                upside = (fnd['target_mean_price'] / t_price - 1) * 100
                t4.metric("Upside ถึงเป้าเฉลี่ย", f"{upside:+.1f}%")
            else:
                t4.metric("Upside ถึงเป้าเฉลี่ย", "ไม่มีข้อมูล")

            if fnd['fifty_two_low'] and fnd['fifty_two_high'] and fnd['fifty_two_high'] > fnd['fifty_two_low']:
                pos = max(0, min(100, (t_price - fnd['fifty_two_low']) / (fnd['fifty_two_high'] - fnd['fifty_two_low']) * 100))
                st.progress(pos / 100)
                st.caption(
                    f"52W Low: ${fnd['fifty_two_low']:,.2f}  |  ราคาปัจจุบัน: ${t_price:,.2f}  |  "
                    f"52W High: ${fnd['fifty_two_high']:,.2f}  (อยู่ที่ {pos:.0f}% ของกรอบ 52 สัปดาห์)"
                )

            st.markdown("#### 4. โมเมนตัมและเส้นค่าเฉลี่ย")
            sma_position = "เหนือ" if t_price > t_sma50 else "ต่ำกว่า"
            st.write(
                f"ราคาปัจจุบันอยู่{sma_position}เส้นค่าเฉลี่ย 50 วัน (SMA 50 = ${t_sma50:,.2f}) "
                f"โมเมนตัมระยะ 1 เดือนอยู่ที่ {t_mom1m:+.2f}% และระยะ 1 สัปดาห์อยู่ที่ {t_mom1w:+.2f}%"
            )

            st.markdown("#### 5. มุมมองการลงทุน: ควรซื้อมาเติมในพอร์ตหรือไม่")
            if t_score >= 70:
                st.success(
                    f"ควรพิจารณาซื้อมาเติมในพอร์ต (Strong Buy) — คะแนนความแข็งแกร่งสูงถึง {t_score}/100 "
                    "แนวโน้มธุรกิจและโมเมนตัมราคาสอดคล้องเชิงบวก"
                )
            elif t_score >= 45:
                st.warning(
                    f"แนะนำให้รอดูสถานะ หรือยังไม่รีบซื้อเพิ่ม (Hold) — คะแนนอยู่ที่ {t_score}/100 "
                    "ควรถือต่อถ้ามีของเดิมอยู่แล้ว แต่หากจะซื้อเพิ่มควรรอจังหวะย่อตัวก่อน"
                )
            else:
                st.error(
                    f"ไม่ควรซื้อมาเติมในพอร์ตในเวลานี้ (Avoid) — คะแนนต่ำเพียง {t_score}/100 "
                    "โครงสร้างราคาอ่อนแอ ควรหลีกเลี่ยงจนกว่าสัญญาณจะดีขึ้น"
                )

            try:
                hist_chart = yf.download(t_ticker, period="6mo", progress=False)
                if not hist_chart.empty:
                    close_c = hist_chart['Close'].iloc[:, 0] if isinstance(hist_chart.columns, pd.MultiIndex) else hist_chart['Close']
                    st.markdown(f"#### กราฟราคาปิดย้อนหลัง 6 เดือนของ {t_ticker}")
                    st.line_chart(close_c)
            except Exception:
                pass

# ---------------------------------------------------------------------
# TAB 3 — MY PORTFOLIO
# ---------------------------------------------------------------------
with tab3:
    st.subheader("พอร์ตการลงทุนของคุณ (My Portfolio Analysis)")
    st.write("กรอกข้อมูลหุ้น จำนวนหุ้น และราคาต้นทุนที่คุณซื้อมา เพื่อคำนวณกำไร/ขาดทุนและรับคำแนะนำรายตัว")
    st.caption(
        "ระบบจะจดจำพอร์ตของคุณไว้อัตโนมัติ (บันทึกลงไฟล์บนเครื่อง/เซิร์ฟเวอร์ที่รันแอปนี้) ครั้งหน้าเปิดแอปมาไม่ต้องกรอกใหม่ "
        "— หมายเหตุ: ถ้าดีพลอยบน Streamlit Community Cloud ไฟล์อาจถูกล้างเมื่อมีการ redeploy แอปใหม่"
    )

    if "portfolio_input" not in st.session_state:
        st.session_state.portfolio_input = load_portfolio_from_disk()

    st.markdown(
        "แก้ไขตารางด้านล่างนี้เพื่อใส่พอร์ตของคุณ (ใส่ Ticker เดิมซ้ำได้หากซื้อคนละไม้ ระบบจะรวมต้นทุนเฉลี่ยให้ "
        "/ ระบุ Asset Class เป็น Growth หรือ Defensive เพื่อใช้เช็คสัดส่วน 60/40)"
    )
    edited_portfolio = st.data_editor(
        st.session_state.portfolio_input,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Asset Class": st.column_config.SelectboxColumn(
                "Asset Class", options=["Growth", "Defensive"], default="Growth"
            )
        }
    )

    try:
        data_changed = not edited_portfolio.equals(st.session_state.portfolio_input)
    except Exception:
        data_changed = True
    if data_changed:
        st.session_state.portfolio_input = edited_portfolio
        save_portfolio_to_disk(edited_portfolio)

    if st.button("ล้างพอร์ตที่บันทึกไว้"):
        st.session_state.portfolio_input = DEFAULT_PORTFOLIO.copy()
        save_portfolio_to_disk(DEFAULT_PORTFOLIO)
        st.rerun()

    with st.expander("ตั้งค่ากติกาความเสี่ยง (Risk Rules)"):
        stop_loss_pct = st.slider("Stop-Loss ตัดขาดทุนเมื่อขาดทุนเกิน (%)", min_value=5, max_value=50, value=30, step=5)
        max_holdings = st.slider("จำนวนหุ้นสูงสุดที่แนะนำถือ (กันการกระจายทุนมากเกินไป)", min_value=5, max_value=30, value=20)
        concentration_limit = st.slider("สัดส่วนสูงสุดต่อหุ้นก่อนเตือนกระจุกตัว (%)", min_value=10, max_value=50, value=25, step=5)

    if not edited_portfolio.empty:
        clean_rows = []
        for _, row in edited_portfolio.iterrows():
            ticker = str(row['Ticker']).strip().upper()
            shares = float(row['Shares']) if pd.notna(row['Shares']) else 0.0
            buy_price = float(row['Buy Price (USD)']) if pd.notna(row['Buy Price (USD)']) else 0.0
            asset_class = str(row.get('Asset Class', 'Growth')).strip() if pd.notna(row.get('Asset Class', 'Growth')) else 'Growth'
            if asset_class not in ("Growth", "Defensive"):
                asset_class = "Growth"
            if ticker and shares > 0 and buy_price > 0:
                clean_rows.append({'Ticker': ticker, 'Shares': shares, 'Buy Price (USD)': buy_price, 'Asset Class': asset_class})

        if clean_rows:
            df_clean = pd.DataFrame(clean_rows)
            df_clean['Cost Basis ($)'] = df_clean['Shares'] * df_clean['Buy Price (USD)']
            agg = df_clean.groupby('Ticker').agg(
                Shares=('Shares', 'sum'),
                CostBasis=('Cost Basis ($)', 'sum'),
                AssetClass=('Asset Class', 'first')
            ).reset_index()
            agg['Buy Price (USD)'] = agg['CostBasis'] / agg['Shares']
        else:
            agg = pd.DataFrame(columns=['Ticker', 'Shares', 'Buy Price (USD)', 'AssetClass'])

        portfolio_results = []
        price_history_map = {}
        for _, row in agg.iterrows():
            ticker = row['Ticker']
            shares = float(row['Shares'])
            buy_price = float(row['Buy Price (USD)'])
            asset_class = row['AssetClass']

            current_price = 0.0
            signal = "ถือประเมินสถานะ"
            score = 50
            trend = "ไม่ทราบ"

            match_row = df[df['Ticker'] == ticker]
            if not match_row.empty:
                current_price = float(match_row.iloc[0]['Price (USD)'])
                signal = match_row.iloc[0]['Simons Signal']
                score = int(match_row.iloc[0]['Score'])
                trend = match_row.iloc[0]['Trend']

            try:
                hist = yf.download(ticker, period="6mo", progress=False)
                if not hist.empty:
                    close_series = hist['Close'].iloc[:, 0] if isinstance(hist.columns, pd.MultiIndex) else hist['Close']
                    price_history_map[ticker] = close_series
                    if match_row.empty:
                        current_price = float(close_series.iloc[-1])
                        ref_price = float(close_series.iloc[-min(5, len(close_series))])
                        wk_change = (current_price - ref_price) / ref_price * 100
                        trend = "ขาขึ้น" if wk_change > 0.5 else ("ขาลง" if wk_change < -0.5 else "ไซด์เวย์")
            except Exception:
                if current_price == 0.0:
                    current_price = buy_price

            invested_value = shares * buy_price
            current_value = shares * current_price
            profit_loss_usd = current_value - invested_value
            profit_loss_pct = ((current_price - buy_price) / buy_price) * 100 if buy_price > 0 else 0.0

            portfolio_results.append({
                'Ticker': ticker,
                'Shares': shares,
                'Buy Price (USD)': buy_price,
                'Current Price (USD)': current_price,
                'Invested Value ($)': invested_value,
                'Current Value ($)': current_value,
                'Profit/Loss ($)': profit_loss_usd,
                'Profit/Loss (%)': profit_loss_pct,
                'Simons Signal': signal,
                'Score': score,
                'Trend': trend,
                'Asset Class': asset_class
            })

        if portfolio_results:
            df_res = pd.DataFrame(portfolio_results)

            total_invested = df_res['Invested Value ($)'].sum()
            total_current = df_res['Current Value ($)'].sum()
            total_pl = total_current - total_invested
            total_pl_pct = (total_pl / total_invested * 100) if total_invested > 0 else 0.0
            df_res['Weight in Portfolio (%)'] = (df_res['Current Value ($)'] / total_current * 100) if total_current > 0 else 0.0

            st.divider()
            with st.container(border=True):
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("มูลค่าลงทุนรวม", f"${total_invested:,.2f}")
                col_m2.metric("มูลค่าปัจจุบันรวม", f"${total_current:,.2f}")
                col_m3.metric("กำไร/ขาดทุนรวม", f"${total_pl:,.2f}", f"{total_pl_pct:.2f}%")

            if len(df_res) > max_holdings:
                st.warning(
                    f"พอร์ตของคุณมี {len(df_res)} ตัว ซึ่งเกินเกณฑ์ {max_holdings} ตัวที่ตั้งไว้ — "
                    "อาจเข้าข่ายกระจายทุนมากเกินไป (Diworsification)"
                )

            concentrated = df_res[df_res['Weight in Portfolio (%)'] > concentration_limit]
            if not concentrated.empty:
                names = ", ".join(concentrated['Ticker'].tolist())
                st.warning(
                    f"หุ้น {names} มีสัดส่วนเกิน {concentration_limit}% ของพอร์ตต่อตัว — "
                    "ความเสี่ยงกระจุกตัวสูง ควรพิจารณาปรับสมดุล (Rebalance)"
                )

            st.markdown("### รายละเอียดพอร์ตและผลตอบแทนรายตัว")
            st.dataframe(
                df_res.sort_values('Weight in Portfolio (%)', ascending=False)[
                    ['Ticker', 'Shares', 'Buy Price (USD)', 'Current Price (USD)', 'Profit/Loss ($)',
                     'Profit/Loss (%)', 'Weight in Portfolio (%)', 'Asset Class', 'Simons Signal', 'Trend', 'Score']
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
                hide_index=True
            )

            st.download_button(
                label="ดาวน์โหลดรายงานพอร์ต (CSV)",
                data=df_res.to_csv(index=False).encode('utf-8-sig'),
                file_name="portfolio_report.csv",
                mime="text/csv"
            )

            st.markdown("### สัดส่วนสินทรัพย์ Growth / Defensive (แนวทาง 60/40)")
            growth_weight = df_res.loc[df_res['Asset Class'] == 'Growth', 'Weight in Portfolio (%)'].sum()
            defensive_weight = df_res.loc[df_res['Asset Class'] == 'Defensive', 'Weight in Portfolio (%)'].sum()
            col_a1, col_a2 = st.columns(2)
            col_a1.metric("Growth", f"{growth_weight:.1f}%", "เป้าหมาย 60%")
            col_a2.metric("Defensive", f"{defensive_weight:.1f}%", "เป้าหมาย 40%")
            st.progress(min(int(growth_weight), 100) / 100)
            if abs(growth_weight - 60) > 10:
                tilt = "เอียงไปทาง Growth มากเกินไป" if growth_weight > 60 else "เอียงไปทาง Defensive มากเกินไป"
                st.warning(f"สัดส่วนปัจจุบันเบี่ยงจากกรอบ 60/40 เกิน 10 จุด ({tilt}) พิจารณาปรับสมดุลพอร์ต")
            else:
                st.success("สัดส่วน Growth/Defensive ยังอยู่ในกรอบ 60/40 ที่ยอมรับได้")

            valid_history = {t: s for t, s in price_history_map.items() if len(s) > 30}
            if len(valid_history) >= 2:
                st.markdown("### Correlation Matrix ระหว่างหุ้นในพอร์ต (6 เดือนย้อนหลัง)")
                st.caption(
                    "คู่หุ้นที่มี Correlation สูง (ใกล้ +1) เคลื่อนไหวตามกัน แทบไม่ช่วยกระจายความเสี่ยง "
                    "ส่วนคู่ที่เป็นลบ (ใกล้ -1) ช่วยป้องกันความเสี่ยงซึ่งกันและกันได้ดีกว่า"
                )
                returns_df = pd.DataFrame({t: s.pct_change() for t, s in valid_history.items()}).dropna(how='all')
                corr_matrix = returns_df.corr().round(2)
                st.dataframe(corr_matrix, use_container_width=True)

                high_corr_pairs, hedge_pairs = [], []
                tickers_list = corr_matrix.columns.tolist()
                for i in range(len(tickers_list)):
                    for j in range(i + 1, len(tickers_list)):
                        t1, t2 = tickers_list[i], tickers_list[j]
                        c = corr_matrix.loc[t1, t2]
                        if c > 0.7:
                            high_corr_pairs.append(f"{t1}-{t2} ({c:.2f})")
                        elif c < 0:
                            hedge_pairs.append(f"{t1}-{t2} ({c:.2f})")

                if high_corr_pairs:
                    st.warning("คู่หุ้นที่เคลื่อนไหวตามกันสูง (Correlation > 0.7): " + ", ".join(high_corr_pairs))
                if hedge_pairs:
                    st.success("คู่หุ้นที่ช่วยกระจายความเสี่ยงกันได้ดี (Correlation ติดลบ): " + ", ".join(hedge_pairs))

            if valid_history:
                st.markdown("### กราฟมูลค่าพอร์ตย้อนหลัง (ประมาณการ)")
                st.caption(
                    "สมมติว่าถือจำนวนหุ้นปัจจุบันคงที่ตลอด 6 เดือนที่ผ่านมา ใช้ดูทิศทางความผันผวนของพอร์ตโดยรวม "
                    "ไม่ใช่ผลตอบแทนจริงตามวันที่ซื้อจริง"
                )
                shares_map = dict(zip(df_res['Ticker'], df_res['Shares']))
                equity_df = pd.DataFrame({t: s * shares_map.get(t, 0) for t, s in valid_history.items()})
                st.line_chart(equity_df.sum(axis=1))

            st.markdown("### คำแนะนำเชิงกลยุทธ์รายตัว")
            st.caption(
                "หลักการ: ขาดทุนอย่างเดียวไม่ใช่สัญญาณขาย — จะแนะนำขาย/ตัดขาดทุนเฉพาะเมื่อราคาอยู่ในเทรนด์ขาลงจริง "
                "หรือขาดทุนเกินเกณฑ์ Stop-Loss ที่ตั้งไว้เท่านั้น"
            )
            for _, r in df_res.sort_values('Weight in Portfolio (%)', ascending=False).iterrows():
                pl_text = f"กำไร {r['Profit/Loss (%)']:.2f}%" if r['Profit/Loss (%)'] >= 0 else f"ขาดทุน {abs(r['Profit/Loss (%)']):.2f}%"
                base_info = (
                    f"{r['Ticker']} (ต้นทุนเฉลี่ย: ${r['Buy Price (USD)']:.2f} | "
                    f"ราคาปัจจุบัน: ${r['Current Price (USD)']:.2f} | {pl_text} | "
                    f"สัดส่วนในพอร์ต: {r['Weight in Portfolio (%)']:.1f}%)"
                )
                if r['Profit/Loss (%)'] <= -stop_loss_pct:
                    st.error(f"{base_info}: ขาดทุนเกินเกณฑ์ Stop-Loss ({stop_loss_pct}%) แนะนำพิจารณาตัดขาดทุน")
                elif "ซื้อสะสม" in r['Simons Signal']:
                    st.info(f"{base_info}: สถานะแข็งแกร่ง แนะนำให้ถือต่อหรือทยอยเพิ่มทุน")
                elif "ควรขาย" in r['Simons Signal'] and r['Trend'] == "ขาลง":
                    st.error(f"{base_info}: สัญญาณอ่อนแอและราคาอยู่ในเทรนด์ขาลงจริง แนะนำพิจารณาขายหรือลดสัดส่วน")
                elif "ควรขาย" in r['Simons Signal']:
                    st.warning(f"{base_info}: สัญญาณอ่อนแอ แต่ยังไม่ยืนยันเทรนด์ขาลงชัดเจน แนะนำเฝ้าดูใกล้ชิด")
                else:
                    st.warning(f"{base_info}: สัญญาณอยู่ในโซนพักตัว แนะนำให้ถือรอดูสถานะต่อไป")

# ---------------------------------------------------------------------
# TAB 4 — AI CHAT
# ---------------------------------------------------------------------
import time

with tab4:
    st.subheader("พูดคุยกับ AI ปรมาจารย์ Jim Simons")
    st.sidebar.caption(
        "AI Chat นี้ใช้ Google Gemini API (โมเดล Flash) ซึ่งมี free tier ถาวรจาก Google AI Studio "
        "— รับ API Key ฟรีได้ที่ aistudio.google.com/apikey"
    )
    gemini_api_key = st.sidebar.text_input("ใส่ Gemini API Key (ฟรีจาก Google AI Studio)", type="password")

    if gemini_api_key:
        portfolio_context = "ไม่มีข้อมูลพอร์ตในขณะนี้"
        if 'df_res' in dir() and isinstance(df_res, pd.DataFrame) and not df_res.empty:
            lines = []
            for _, r in df_res.iterrows():
                lines.append(
                    f"- หุ้น {r['Ticker']}: ถือ {r['Shares']} หุ้น, ต้นทุน ${r['Buy Price (USD)']:,.2f}, "
                    f"ราคาปัจจุบัน ${r['Current Price (USD)']:,.2f}, กำไร/ขาดทุน {r['Profit/Loss (%)']:.2f}%, "
                    f"สัดส่วน {r['Weight in Portfolio (%)']:.1f}%, ประเภท {r['Asset Class']}, สัญญาณโมเดล: {r['Simons Signal']}"
                )
            portfolio_context = "\n".join(lines)

        system_instruction = (
            "คุณคือ Jim Simons ผู้ก่อตั้ง Renaissance Technologies และปรมาจารย์กองทุน Quantitative "
            "คุณมองโลกผ่านตัวเลข สถิติ ความน่าจะเป็น และรูปแบบของข้อมูล ไม่ใช่การเก็งกำไรตามอารมณ์ "
            "เน้นย้ำเรื่องการบริหารความเสี่ยง การกระจายความเสี่ยง และการควบคุมขนาดของพอร์ต "
            "นี่คือข้อมูลพอร์ตการลงทุนปัจจุบันของผู้ใช้งาน:\n" + portfolio_context + "\n"
            "จงตอบคำถามด้วยน้ำเสียงสุขุม เป็นนักวิทยาศาสตร์ ตรงไปตรงมา มีตรรกะทางคณิตศาสตร์รองรับ ห้ามใช้อีโมจิเด็ดขาด"
        )

        if "messages" not in st.session_state or st.session_state.get("last_system_prompt") != system_instruction:
            st.session_state.messages = []
            st.session_state.last_system_prompt = system_instruction

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("ปรึกษาเรื่องพอร์ตหรือสถิติตลาดกับ Jim Simons..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Jim Simons กำลังคำนวณโมเดลคำตอบ..."):
                    try:
                        client = genai.Client(api_key=gemini_api_key)
                        
                        # แปลงประวัติข้อความ
                        gemini_contents = [
                            types.Content(
                                role=("user" if m["role"] == "user" else "model"),
                                parts=[types.Part.from_text(text=m["content"])]
                            )
                            for m in st.session_state.messages
                        ]
                        
                        max_retries = 3
                        reply = None
                        
                        for attempt in range(max_retries):
                            try:
                                # ใช้ gemini-2.5-flash สำหรับ google-genai SDK
                                response = client.models.generate_content(
                                    model="gemini-2.5-flash",
                                    contents=gemini_contents,
                                    config=types.GenerateContentConfig(
                                        system_instruction=system_instruction
                                    )
                                )
                                reply = response.text
                                break
                            except Exception as err:
                                if ("503" in str(err) or "UNAVAILABLE" in str(err)) and attempt < max_retries - 1:
                                    time.sleep(2)
                                    continue
                                raise err

                        if reply:
                            st.markdown(reply)
                            st.session_state.messages.append({"role": "assistant", "content": reply})
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ Gemini: {e}")
    else:
        st.info(
            "กรอก Gemini API Key ที่แถบซ้ายมือ (Sidebar) เพื่อเปิดใช้งานช่องแชทปรึกษาพอร์ตกับ Jim Simons "
        )