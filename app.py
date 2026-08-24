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

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="Simon Screener & Portfolio AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- กลไกสลับธีม (Light/Dark Mode) ---
st.sidebar.header("🎨 ตั้งค่าการแสดงผล")
theme_mode = st.sidebar.selectbox("เลือกธีมหน้าจอ", ["โหมดกลางคืน (Dark Mode)", "โหมดกลางวัน (Light Mode)"])

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

# --- CSS Styling ---
st.markdown(f"""
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
        padding: 25px;
        border-radius: 12px;
        border: 1px solid {border_color};
        margin-top: 20px;
        margin-bottom: 20px;
        color: {text_color};
        line-height: 1.6;
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
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {card_bg};
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        color: {text_color};
        border: 1px solid {border_color};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #238636 !important;
        color: white !important;
    }}
    </style>
""", unsafe_allow_html=True)

PORTFOLIO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio_data.json")

def load_portfolio_from_disk():
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
    return pd.DataFrame([
        {"Ticker": "AAPL", "Shares": 10.0, "Buy Price (USD)": 170.0, "Asset Class": "Growth"},
        {"Ticker": "TSLA", "Shares": 5.0, "Buy Price (USD)": 220.0, "Asset Class": "Growth"}
    ])

def save_portfolio_to_disk(df):
    try:
        df.to_json(PORTFOLIO_FILE, orient="records", force_ascii=False)
    except Exception as e:
        st.sidebar.error(f"บันทึกพอร์ตลงดิสก์ไม่สำเร็จ: {e}")

# --- Header Section ---
st.markdown(f"<h1 style='text-align: center; color: {heading_color};'>SIMON QUANT & PORTFOLIO AI ADVISOR</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: {text_color};'>ระบบสแกนหุ้นสหรัฐฯ และผู้ช่วยวิเคราะห์พอร์ตการลงทุนส่วนตัวด้วยโมเดลสถิติเชิงปริมาณ (Renaissance Technologies Style)</p>", unsafe_allow_html=True)
st.markdown("---")

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

    simons_score = 50
    if current_p > sma_50:
        simons_score += 25
    else:
        simons_score -= 20

    if return_1m > 0:
        simons_score += 20
    else:
        simons_score -= 15

    if volatility < 40:
        simons_score += 15
    else:
        simons_score -= 10

    simons_score = max(0, min(100, simons_score))

    if simons_score >= 70:
        recommendation = "ซื้อสะสม (Strong Buy)"
    elif simons_score >= 45:
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
        'Score': simons_score,
        'Price (USD)': current_p,
        'Momentum 1M (%)': return_1m,
        'Momentum 1W (%)': return_1w,
        'Trend': trend,
        'Volatility (%)': volatility,
        'SMA 50': sma_50
    }

@st.cache_data(ttl=1800)
def fetch_stock_data_and_simons_logic(ticker_list):
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
    all_rows = []
    total = len(ticker_list)
    progress_bar = st.progress(0.0, text="กำลังสแกนตลาด...")
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
        progress_bar.progress(done / total, text=f"สแกนแล้ว {done:,}/{total:,} ตัว (สำเร็จ {len(all_rows):,} ตัว)")

    progress_bar.empty()
    return pd.DataFrame(all_rows)

def get_company_business_info(ticker_symbol):
    try:
        t = yf.Ticker(ticker_symbol)
        info = t.info
        long_name = info.get('longName', ticker_symbol)
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        summary = info.get('longBusinessSummary', 'ไม่มีข้อมูลสรุปธุรกิจจากระบบ Yahoo Finance')
        return long_name, sector, industry, summary
    except Exception:
        return ticker_symbol, 'N/A', 'N/A', 'ไม่สามารถดึงข้อมูลธุรกิจได้ในขณะนี้'

def get_detailed_future_outlook(sector, industry, ticker):
    sector_lower = str(sector).lower()
    if "technology" in sector_lower or "semiconductor" in industry.lower():
        return (
            f"บริษัท {ticker} ตั้งอยู่ในกลุ่มอุตสาหกรรมเทคโนโลยีขั้นสูงและเซมิคอนดักเตอร์ ซึ่งเป็นหัวใจสำคัญของโครงสร้างพื้นฐานยุคปัญญาประดิษฐ์ (AI Infrastructure) "
            "แนวโน้มในอนาคตยังคงได้รับแรงหนุนมหาศาลจากงบลงทุน (CapEx) ของกลุ่ม Cloud Hyperscalers และการนำ AI ไปประยุกต์ใช้ในระดับองค์กร (Enterprise AI) "
            "ความเสี่ยงสำคัญคือวัฏจักรความต้องการชิป (Cyclical Demand) และอุปทานส่วนเกินที่อาจเกิดขึ้นได้ในระยะสั้น แต่ในระยะยาวมีความได้เปรียบเชิงการแข่งขันสูงจากความซับซ้อนทางเทคโนโลยี"
        )
    elif "consumer" in sector_lower:
        return (
            f"บริษัท {ticker} ดำเนินธุรกิจในกลุ่มสินค้าอุปโภคบริโภค ซึ่งมีความเชื่อมโยงโดยตรงกับกำลังซื้อของผู้บริโภค อัตราเงินเฟ้อ และทิศทางอัตราดอกเบี้ยนโยบาย "
            "แนวโน้มในอนาคตขึ้นอยู่กับความสามารถในการรักษาอำนาจในการกำหนดราคาสินค้า (Pricing Power) เพื่อป้องกันต้นทุนที่สูงขึ้น "
            "รวมถึงการปรับตัวสู่ช่องทางดิจิทัลและอีคอมเมิร์ซเพื่อรักษาฐานลูกค้าในระยะยาว"
        )
    elif "health" in sector_lower:
        return (
            f"บริษัท {ticker} อยู่ในกลุ่มเทคโนโลยีชีวภาพและสุขภาพ (Healthcare & Biotech) ซึ่งขับเคลื่อนด้วยสังคมผู้สูงอายุทั่วโลก (Aging Population) "
            "และนวัตกรรมการแพทย์เฉพาะบุคคล (Precision Medicine) แนวโน้มในอนาคตมีปัจจัยบวกจากการคิดค้นนวัตกรรมยาและเครื่องมือแพทย์ใหม่ๆ "
            "แต่มีความเสี่ยงด้านกฎระเบียบ การควบคุมราคายาจากภาครัฐ และผลการทดลองทางคลินิก (Clinical Trials)"
        )
    else:
        return (
            f"บริษัท {ticker} ในกลุ่มอุตสาหกรรม {sector} มีทิศทางเติบโตสอดคล้องกับภาพรวมเศรษฐกิจมหภาค (Macroeconomic Conditions) "
            "แนวโน้มในอนาคตขึ้นอยู่กับประสิทธิภาพในการบริหารจัดการต้นทุนห่วงโซ่อุปทาน (Supply Chain) และความสามารถในการปรับตัวรับมือกับความผันผวนของอัตราแลกเปลี่ยนและดอกเบี้ยโลก"
        )

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ ตั้งค่าแหล่งข้อมูล & สแกนหุ้น")
scan_mode = st.sidebar.radio(
    "เลือกแหล่งข้อมูลหุ้น",
    ["รายชื่อคัดสรร (~85 ตัว, เร็ว)", "ทั้งตลาดหุ้นสหรัฐฯ (NASDAQ + NYSE + AMEX)"]
)

if scan_mode.startswith("รายชื่อคัดสรร"):
    symbols = get_curated_symbols()
    with st.spinner("กำลังประมวลผลโมเดล Quant..."):
        df = fetch_stock_data_and_simons_logic(symbols)
else:
    with st.spinner("กำลังดึงรายชื่อหลักทรัพย์ทั้งหมด..."):
        full_list_df = get_full_us_market_symbols()

    st.sidebar.success(f"พบหลักทรัพย์ทั้งหมด {len(full_list_df):,} ตัว")
    exclude_etf = st.sidebar.checkbox("ไม่รวมกองทุน ETF (แนะนำ)", value=True)
    universe_df = full_list_df[~full_list_df['ETF']] if exclude_etf else full_list_df

    max_scan = st.sidebar.number_input(
        "จำนวนหุ้นสูงสุดที่จะสแกนรอบนี้",
        min_value=50, max_value=int(len(universe_df)),
        value=min(300, int(len(universe_df))), step=50
    )
    start_offset = st.sidebar.number_input(
        "เริ่มสแกนจากลำดับที่ (Offset)",
        min_value=0, max_value=max(0, int(len(universe_df)) - 1), value=0, step=max_scan
    )
    run_scan = st.sidebar.button("🚀 เริ่มสแกนตลาดเต็มรูปแบบ", use_container_width=True)

    if run_scan:
        target_symbols = universe_df['Symbol'].iloc[start_offset:start_offset + max_scan].tolist()
        st.session_state.full_scan_df = fetch_stock_data_batched(target_symbols)
        st.session_state.full_scan_range = (start_offset, start_offset + len(target_symbols))

    if "full_scan_df" in st.session_state:
        df = st.session_state.full_scan_df
        rng = st.session_state.get("full_scan_range", (0, len(df)))
        st.sidebar.caption(f"ผลสแกนล่าสุด: ช่วงลำดับ {rng[0]:,}-{rng[1]:,}")
    else:
        st.info("👈 กรุณาเลือกโหมดและคลิกเริ่มสแกนที่แถบด้านซ้ายมือเพื่อเริ่มต้นใช้งานระบบ")
        st.stop()

if df.empty:
    st.error("ไม่สามารถดึงข้อมูลได้ในขณะนี้ กรุณารีเฟรชหน้าเว็บ หรือตรวจสอบการเชื่อมต่อ")
else:
    # --- Tabs Layout ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 สแกนหุ้นตลาด", "📝 บทความวิเคราะห์หุ้นรายตัว", "💼 พอร์ตของฉัน & ใบเสร็จซื้อขาย", "🤖 คุยกับ AI Jim Simons"])

    with tab1:
        st.subheader("ตารางสแกนหุ้นสหรัฐฯ เชิงปริมาณ (Quantitative Screener)")
        col_f1, col_f2 = st.columns([2, 2])
        with col_f1:
            signal_filter = st.selectbox("กรองตามคำแนะนำโมเดล", ["ทั้งหมด", "ซื้อสะสม (Strong Buy)", "ถือ / เฝ้าสังเกต (Hold)", "ควรขาย / หลีกเลี่ยง (Sell / Avoid)"])
        with col_f2:
            search_query = st.text_input("🔍 ค้นหา Ticker หุ้นเจาะจงในตารางผลลัพธ์", "").upper()

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

    with tab2:
        st.subheader("📝 บทความวิเคราะห์หุ้นรายตัวเชิงลึก & แนวโน้มธุรกิจ")
        st.markdown("พิมพ์ชื่อย่อหุ้น (Ticker) เพื่อดูภาพรวมธุรกิจ แนวโน้มอนาคต และคำแนะนำการซื้อเข้าพอร์ต:")
        
        inner_search_ticker = st.text_input("🔍 พิมพ์ Ticker หุ้นที่ต้องการวิเคราะห์เจาะจง (เช่น NVDA, TSLA, AAPL, PLTR)", "NVDA").upper().strip()

        if not inner_search_ticker:
            st.info("กรุณากรอก Ticker หุ้นที่ต้องการวิเคราะห์")
        else:
            match_dd = df[df['Ticker'] == inner_search_ticker]
            if match_dd.empty:
                try:
                    with st.spinner(f"กำลังดึงข้อมูลสถิติของ {inner_search_ticker}..."):
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
                r_dd = match_dd.iloc[0]
                t_ticker = r_dd['Ticker']
                t_price = r_dd['Price (USD)']
                t_score = r_dd['Score']
                t_signal = r_dd['Simons Signal']
                t_trend = r_dd['Trend']
                t_mom1m = r_dd['Momentum 1M (%)']
                t_mom1w = r_dd['Momentum 1W (%)']
                t_vol = r_dd['Volatility (%)']
                t_sma50 = r_dd['SMA 50']

                long_name, sector, industry, biz_summary = get_company_business_info(t_ticker)
                detailed_outlook = get_detailed_future_outlook(sector, industry, t_ticker)

                # จุดที่แก้ไข: เพิ่ม unsafe_allow_html=True ให้ครบถ้วน
                st.markdown(f"<p><b>ราคาซื้อขายล่าสุด:</b> ${t_price:,.2f} USD</p>", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="article-box">
                    <h2>รายงานวิเคราะห์เชิงปริมาณ: หุ้น {t_ticker} ({long_name})</h2>
                    <p><b>กลุ่มธุรกิจ (Sector):</b> {sector} | <b>อุตสาหกรรม (Industry):</b> {industry}</p>
                    <hr style="border-color: {border_color};">
                    
                    <h3>1. ลักษณะการทำธุรกิจและแนวโน้มในอนาคต (Business Profile & Future Outlook)</h3>
                    <p><b>ภาพรวมการประกอบธุรกิจ:</b> {biz_summary}</p>
                    <p><b>การวิเคราะห์เจาะลึกแนวโน้มธุรกิจในอนาคต:</b> {detailed_outlook}</p>

                    <h3>2. สรุปภาพรวมคะแนนและความน่าจะเป็น (Quantitative Score & Verdict)</h3>
                    <p>โมเดลเชิงปริมาณประเมินความแข็งแกร่งของหลักทรัพย์ <b>{t_ticker}</b> ได้คะแนนรวมอยู่ที่ <b>{t_score}/100 คะแนน</b> โดยให้คำแนะนำสถานะทางสถิติเป็น: <b style="color: {heading_color};">{t_signal}</b> โครงสร้างราคาปัจจุบันเคลื่อนไหวอยู่ในภาวะ <b>{t_trend}</b></p>

                    <h3>3. การวิเคราะห์โมเมนตัมและเส้นค่าเฉลี่ย (Momentum & Trend Structure)</h3>
                    <ul>
                        <li><b>โมเมนตัมระยะ 1 เดือน (1M Momentum):</b> <b>{t_mom1m:+.2f}%</b></li>
                        <li><b>โมเมนตัมระยะ 1 สัปดาห์ (1W Momentum):</b> <b>{t_mom1w:+.2f}%</b></li>
                        <li><b>เส้นค่าเฉลี่ย 50 วัน (SMA 50):</b> <b>${t_sma50:,.2f} USD</b> (ราคาปัจจุบันอยู่{ "เหนือ" if t_price > t_sma50 else "ต่ำกว่า" }เส้น SMA 50)</li>
                    </ul>

                    <h3>4. การประเมินความเสี่ยงและความผันผวน (Risk & Volatility Profile)</h3>
                    <p>ค่าความผันผวนรายปี (Annualized Volatility) อยู่ที่ระดับ <b>{t_vol:.2f}%</b> ควบคุมความเสี่ยงตามหลักการของโมเดล Quant</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### 🎯 มุมมองการลงทุน: ควรซื้อมาเติมในพอร์ตหรือไม่?")
                if t_score >= 70:
                    st.markdown(f"""
                    <div class="verdict-box-yes">
                        <b>🟢 ควรพิจารณาซื้อมาเติมในพอร์ต (Strong Buy Recommendation)</b><br>
                        • <b>เหตุผล:</b> คะแนนความแข็งแกร่งสูงถึง <b>{t_score}/100 คะแนน</b> แนวโน้มธุรกิจและโมเมนตัมราคาสอดคล้องเชิงบวก เหมาะแก่การเพิ่มน้ำหนักลงทุน (Growth)
                    </div>
                    """, unsafe_allow_html=True)
                elif t_score >= 45:
                    st.markdown(f"""
                    <div class="article-box" style="border-color: #d29922; background-color: rgba(210, 153, 34, 0.1);">
                        <b>🟡 แนะนำให้รอดูสถานะ หรือยังไม่รีบซื้อเพิ่ม (Hold / Wait & See)</b><br>
                        • <b>เหตุผล:</b> คะแนนอยู่ที่ <b>{t_score}/100 คะแนน</b> ควรถือต่อถ้ามีของเดิม แต่หากจะซื้อเพิ่มควรรอจังหวะย่อตัว
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="verdict-box-no">
                        <b>🔴 ไม่ควรซื้อมาเติมในพอร์ตในเวลานี้ (Avoid / High Risk)</b><br>
                        • <b>เหตุผล:</b> คะแนนต่ำเพียง <b>{t_score}/100 คะแนน</b> โครงสร้างราคาอ่อนแอ ควรหลีกเลี่ยง
                    </div>
                    """, unsafe_allow_html=True)

                try:
                    hist_chart = yf.download(t_ticker, period="6mo", progress=False)
                    if not hist_chart.empty:
                        close_c = hist_chart['Close'].iloc[:, 0] if isinstance(hist_chart.columns, pd.MultiIndex) else hist_chart['Close']
                        st.subheader(f"📈 กราฟราคาปิดย้อนหลัง 6 เดือนของ {t_ticker}")
                        st.line_chart(close_c)
                except Exception:
                    pass

    with tab3:
        st.subheader("💼 พอร์ตการลงทุนของคุณ & ระบบสแกนใบเสร็จซื้อขาย (Receipt AI Scanner)")
        st.markdown("คุณสามารถกรอกข้อมูลพอร์ตด้านล่าง หรือ **อัปโหลดรูปภาพใบเสร็จ/Slip ซื้อขายหุ้น** เพื่อให้ AI อ่านข้อมูลและเติมลงตารางให้โดยอัตโนมัติ")

        with st.expander("📷 อัปโหลดใบเสร็จ/สลิปซื้อขายหุ้นเพื่อบันทึกลงพอร์ตอัตโนมัติ (AI OCR)", expanded=False):
            uploaded_receipt = st.file_uploader("เลือกรูปภาพใบเสร็จ (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
            ai_key_receipt = st.sidebar.text_input("🔑 OpenAI API Key (สำหรับสแกนใบเสร็จ & AI Chat)", type="password", key="receipt_key")
            
            if uploaded_receipt is not None:
                image = Image.open(uploaded_receipt)
                st.image(image, caption="ใบเสร็จที่คุณอัปโหลด", width=300)
                
                if st.button("🚀 สแกนใบเสร็จและเพิ่มเข้าพอร์ต"):
                    if not ai_key_receipt:
                        st.error("กรุณากรอก OpenAI API Key ที่แถบด้านซ้ายมือ (Sidebar) ก่อนใช้งานฟังก์ชันสแกนใบเสร็จ")
                    else:
                        with st.spinner("AI กำลังแกะข้อมูลจากใบเสร็จ..."):
                            try:
                                buffered = io.BytesIO()
                                image.save(buffered, format="JPEG")
                                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

                                client = openai.OpenAI(api_key=ai_key_receipt)
                                response = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": "คุณคือนักสกัดข้อมูลจากใบเสร็จซื้อขายหุ้น จงอ่านรูปภาพแล้วตอบกลับมาเป็น JSON บริสุทธิ์รูปแบบนี้เท่านั้น โดยไม่มีข้อความอื่น: {\"Ticker\": \"AAPL\", \"Shares\": 10.0, \"Buy Price (USD)\": 175.5}"
                                        },
                                        {
                                            "role": "user",
                                            "content": [
                                                {"type": "text", "text": "กรุณาอ่าน Ticker, จำนวนหุ้น (Shares) และราคาซื้อต่อหุ้น (Buy Price (USD)) จากใบเสร็จนี้"},
                                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                                            ]
                                        }
                                    ],
                                    max_tokens=150
                                )
                                res_text = response.choices[0].message.content.strip()
                                if res_text.startswith("```json"):
                                    res_text = res_text[7:-3].strip()
                                elif res_text.startswith("```"):
                                    res_text = res_text[3:-3].strip()
                                
                                extracted_data = json.loads(res_text)
                                new_ticker = str(extracted_data.get("Ticker", "")).upper()
                                new_shares = float(extracted_data.get("Shares", 0))
                                new_price = float(extracted_data.get("Buy Price (USD)", 0))

                                if new_ticker and new_shares > 0 and new_price > 0:
                                    if "portfolio_input" not in st.session_state:
                                        st.session_state.portfolio_input = load_portfolio_from_disk()
                                    
                                    new_row = pd.DataFrame([{
                                        "Ticker": new_ticker,
                                        "Shares": new_shares,
                                        "Buy Price (USD)": new_price,
                                        "Asset Class": "Growth"
                                    }])
                                    st.session_state.portfolio_input = pd.concat([st.session_state.portfolio_input, new_row], ignore_index=True)
                                    save_portfolio_to_disk(st.session_state.portfolio_input)
                                    st.success(f"สแกนสำเร็จ! เพิ่มหุ้น {new_ticker} จำนวน {new_shares} หุ้น ที่ราคา ${new_price} ลงในพอร์ตเรียบร้อยแล้ว")
                                    st.rerun()
                                else:
                                    st.error("AI ไม่สามารถอ่านข้อมูลหุ้นจากใบเสร็จนี้ได้ชัดเจน กรุณากรอกข้อมูลลงในตารางด้วยตนเอง")
                            except Exception as e:
                                st.error(f"เกิดข้อผิดพลาดในการประมวลผลรูปภาพ: {e}")

        if "portfolio_input" not in st.session_state:
            st.session_state.portfolio_input = load_portfolio_from_disk()

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

        if st.button("🗑️ ล้างพอร์ตเป็นค่าเริ่มต้น"):
            default_df = pd.DataFrame([
                {"Ticker": "AAPL", "Shares": 10.0, "Buy Price (USD)": 170.0, "Asset Class": "Growth"},
                {"Ticker": "TSLA", "Shares": 5.0, "Buy Price (USD)": 220.0, "Asset Class": "Growth"}
            ])
            st.session_state.portfolio_input = default_df
            save_portfolio_to_disk(default_df)
            st.rerun()

        df_res = pd.DataFrame()
        if not edited_portfolio.empty:
            clean_rows = []
            for _, row in edited_portfolio.iterrows():
                ticker = str(row['Ticker']).strip().upper()
                shares = float(row['Shares']) if pd.notna(row['Shares']) else 0.0
                buy_price = float(row['Buy Price (USD)']) if pd.notna(row['Buy Price (USD)']) else 0.0
                asset_class = str(row.get('Asset Class', 'Growth')).strip() if pd.notna(row.get('Asset Class', 'Growth')) else 'Growth'
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
            for _, row in agg.iterrows():
                ticker = row['Ticker']
                shares = float(row['Shares'])
                buy_price = float(row['Buy Price (USD)'])
                asset_class = row['AssetClass']

                current_price = 0.0
                signal = "ถือประเมินสถานะ"
                score = 50
                trend = "ไม่ทราบ"
                volatility = 0.0
                sma_50 = 0.0
                mom_1m = 0.0

                match_row = df[df['Ticker'] == ticker]
                if not match_row.empty:
                    current_price = float(match_row.iloc[0]['Price (USD)'])
                    signal = match_row.iloc[0]['Simons Signal']
                    score = int(match_row.iloc[0]['Score'])
                    trend = match_row.iloc[0]['Trend']
                    volatility = float(match_row.iloc[0]['Volatility (%)'])
                    sma_50 = float(match_row.iloc[0]['SMA 50'])
                    mom_1m = float(match_row.iloc[0]['Momentum 1M (%)'])

                try:
                    hist = yf.download(ticker, period="6mo", progress=False)
                    if not hist.empty:
                        close_series = hist['Close'].iloc[:, 0] if isinstance(hist.columns, pd.MultiIndex) else hist['Close']
                        if match_row.empty:
                            current_price = float(close_series.iloc[-1])
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

                st.markdown("---")
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("💰 มูลค่าลงทุนรวม", f"${total_invested:,.2f}")
                col_m2.metric("📈 มูลค่าพอร์ตปัจจุบัน", f"${total_current:,.2f}")
                col_m3.metric("📊 กำไร/ขาดทุนรวม", f"${total_pl:,.2f}", f"{total_pl_pct:.2f}%")

                st.markdown("---")
                st.subheader("💡 บทสรุปแผนการปรับพอร์ต (Quantitative Action Summary)")
                
                buy_candidates = df_res[df_res['Score'] >= 70]
                sell_candidates = df_res[df_res['Score'] < 45]

                col_act1, col_act2 = st.columns(2)
                with col_act1:
                    st.markdown("#### 🟢 หุ้นที่ควรพิจารณา 'ซื้อเพิ่ม' (Strong Buy)")
                    if not buy_candidates.empty:
                        for _, bc in buy_candidates.iterrows():
                            st.markdown(f"""
                            <div class="action-box-buy">
                                <b>📌 {bc['Ticker']}</b> (คะแนน: {bc['Score']}/100)<br>
                                • โมเมนตัมขาขึ้นชัดเจน ราคาอยู่เหนือเส้น SMA 50<br>
                                • แนะนำทยอยสะสมเพิ่มเพื่อเพิ่มน้ำหนักความเติบโตในพอร์ต
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("ไม่มีหุ้นในพอร์ตที่เข้าเงื่อนไขซื้อสะสมในรอบนี้ (คะแนน >= 70)")

                with col_act2:
                    st.markdown("#### 🔴 หุ้นที่ควรพิจารณา 'ขาย / ลดความเสี่ยง' (Sell / Avoid)")
                    if not sell_candidates.empty:
                        for _, sc in sell_candidates.iterrows():
                            st.markdown(f"""
                            <div class="action-box-sell">
                                <b>📌 {sc['Ticker']}</b> (คะแนน: {sc['Score']}/100)<br>
                                • สัญญาณทางเทคนิคอ่อนแอหรือหลุดเส้น SMA 50<br>
                                • แนะนำพิจารณาขายทำกำไรหรือตัดขาดทุนเพื่อจำกัดความเสี่ยง
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.success("ยอดเยี่ยม! ไม่มีหุ้นในพอร์ตที่เข้าข่ายต้องขายออกในรอบนี้")

                st.markdown("---")
                st.subheader("📋 สรุปภาพรวมพอร์ตการลงทุน")
                st.dataframe(
                    df_res[['Ticker', 'Shares', 'Buy Price (USD)', 'Current Price (USD)', 'Profit/Loss ($)', 'Profit/Loss (%)', 'Weight in Portfolio (%)', 'Asset Class', 'Simons Signal', 'Score']],
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
                    label="📥 ดาวน์โหลดรายงานพอร์ต (CSV)",
                    data=df_res.to_csv(index=False).encode('utf-8-sig'),
                    file_name="portfolio_report.csv",
                    mime="text/csv"
                )

    with tab4:
        st.subheader("🤖 พูดคุยกับ AI ปรมาจารย์ Jim Simons")
        openai_api_key = st.sidebar.text_input("🔑 ใส่ OpenAI API Key (สำหรับ AI Chat)", type="password", key="chat_key")

        if openai_api_key:
            portfolio_context = "ไม่มีข้อมูลพอร์ตในขณะนี้"
            if 'df_res' in locals() and not df_res.empty:
                portfolio_summary_lines = []
                for _, r in df_res.iterrows():
                    portfolio_summary_lines.append(
                        f"- หุ้น {r['Ticker']}: ถือ {r['Shares']} หุ้น, ต้นทุน ${r['Buy Price (USD)']:,.2f}, "
                        f"ราคาปัจจุบัน ${r['Current Price (USD)']:,.2f}, กำไร/ขาดทุน {r['Profit/Loss (%)']:.2f}%, "
                        f"สัดส่วน {r['Weight in Portfolio (%)']:.1f}%, ประเภท {r['Asset Class']}, สัญญาณโมเดล: {r['Simons Signal']}"
                    )
                portfolio_context = "\n".join(portfolio_summary_lines)

            system_instruction = (
                "คุณคือ Jim Simons ผู้ก่อตั้ง Renaissance Technologies และปรมาจารย์กองทุน Quantitative "
                "คุณมองโลกผ่านตัวเลข สถิติ ความน่าจะเป็น และรูปแบบของข้อมูล ไม่ใช่การเก็งกำไรตามอารมณ์ "
                "เน้นย้ำเรื่องการบริหารความเสี่ยง การกระจายความเสี่ยง และการควบคุมขนาดของพอร์ต "
                "ผู้ใช้งานท่านนี้มีเป้าหมายเน้นการลงทุนแบบเติบโตสูง (Growth) "
                "นี่คือข้อมูลพอร์ตการลงทุนปัจจุบันของผู้ใช้งาน:\n" + portfolio_context + "\n"
                "จงตอบคำถามด้วยน้ำเสียงสุขุม เป็นนักวิทยาศาสตร์ ตรงไปตรงมา มีตรรกะทางคณิตศาสตร์รองรับ ห้ามใช้อีโมจิเด็ดขาด"
            )

            if "messages" not in st.session_state or st.session_state.get("last_system_prompt") != system_instruction:
                st.session_state.messages = [
                    {"role": "system", "content": system_instruction}
                ]
                st.session_state.last_system_prompt = system_instruction

            for message in st.session_state.messages:
                if message["role"] != "system":
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            if prompt := st.chat_input("ปรึกษาเรื่องพอร์ตหรือสถิติตลาดกับ Jim Simons..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Jim Simons กำลังประมวลผลโมเดลสถิติ..."):
                        try:
                            client = openai.OpenAI(api_key=openai_api_key)
                            response = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=st.session_state.messages
                            )
                            reply = response.choices[0].message.content
                            st.markdown(reply)
                            st.session_state.messages.append({"role": "assistant", "content": reply})
                        except Exception as e:
                            st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ OpenAI: {e}")
        else:
            st.info("💡 กรุณากรอก OpenAI API Key ที่แถบด้านซ้ายมือ (Sidebar) เพื่อเปิดใช้งานแชทผู้ช่วย AI และระบบสแกนใบเสร็จ")