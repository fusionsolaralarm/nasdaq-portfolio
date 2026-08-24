import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import openai
import json
import os

PORTFOLIO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio_data.json")

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
    return pd.DataFrame([
        {"Ticker": "AAPL", "Shares": 10.0, "Buy Price (USD)": 170.0, "Asset Class": "Growth"},
        {"Ticker": "TSLA", "Shares": 5.0, "Buy Price (USD)": 220.0, "Asset Class": "Growth"},
        {"Ticker": "JNJ", "Shares": 8.0, "Buy Price (USD)": 150.0, "Asset Class": "Defensive"}
    ])

def save_portfolio_to_disk(df):
    """บันทึกพอร์ตปัจจุบันลงไฟล์ ครั้งถัดไปที่เปิดแอปจะโหลดข้อมูลเดิมกลับมาให้อัตโนมัติ"""
    try:
        df.to_json(PORTFOLIO_FILE, orient="records", force_ascii=False)
    except Exception as e:
        st.sidebar.error(f"บันทึกพอร์ตลงดิสก์ไม่สำเร็จ: {e}")

st.set_page_config(
    page_title="Simon Screener & Portfolio AI",
    layout="wide"
)

st.title("Simon Screener - Quant & Portfolio AI Advisor")
st.markdown("ระบบสแกนหุ้นสหรัฐฯ และผู้ช่วยวิเคราะห์พอร์ตการลงทุนส่วนตัวในสไตล์ Jim Simons (Renaissance Technologies)")

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
    """ดึงรายชื่อหุ้น/หลักทรัพย์ที่จดทะเบียนทั้งหมดในตลาดสหรัฐฯ (NASDAQ + NYSE + AMEX)"""
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

# --- เลือกขอบเขตหุ้นที่จะสแกน ---
st.sidebar.markdown("### ขอบเขตหุ้นที่จะสแกน")
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

    max_scan = st.sidebar.number_input(
        "จำนวนหุ้นสูงสุดที่จะสแกนในรอบนี้",
        min_value=50, max_value=int(len(universe_df)),
        value=min(300, int(len(universe_df))), step=50
    )
    start_offset = st.sidebar.number_input(
        "เริ่มสแกนจากลำดับที่ (offset)",
        min_value=0, max_value=max(0, int(len(universe_df)) - 1), value=0, step=max_scan
    )
    run_scan = st.sidebar.button("เริ่มสแกน")

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
    st.error("ไม่สามารถดึงข้อมูลได้ในขณะนี้ กรุณารีเฟรชหน้าเว็บใหม่อีกครั้ง หรือลองลดจำนวนหุ้นที่สแกนต่อรอบ")
else:
    tab1, tab2, tab3 = st.tabs(["สแกนหุ้นตลาด", "พอร์ตของฉัน", "คุยกับ AI Jim Simons"])

    with tab1:
        st.subheader("ตารางสแกนหุ้นสหรัฐฯ เชิงปริมาณ")
        signal_filter = st.selectbox("กรองตามคำแนะนำโมเดล", ["ทั้งหมด", "ซื้อสะสม (Strong Buy)", "ถือ / เฝ้าสังเกต (Hold)", "ควรขาย / หลีกเลี่ยง (Sell / Avoid)"])

        df_filtered = df if signal_filter == "ทั้งหมด" else df[df['Simons Signal'] == signal_filter]

        search_query = st.text_input("ค้นหา Ticker หุ้น", "").upper()
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
        st.subheader("พอร์ตการลงทุนของคุณ (My Portfolio Analysis)")
        st.markdown("กรอกข้อมูลหุ้น จำนวนหุ้น และราคาต้นทุนที่คุณซื้อมา เพื่อคำนวณกำไร/ขาดทุน วิเคราะห์สัดส่วน และบทวิเคราะห์รายตัว")
        st.caption("ระบบจะจดจำพอร์ตของคุณไว้อัตโนมัติ (บันทึกลงไฟล์บนดิสก์)")

        if "portfolio_input" not in st.session_state:
            st.session_state.portfolio_input = load_portfolio_from_disk()

        st.markdown("แก้ไขตารางด้านล่างนี้เพื่อใส่พอร์ตของคุณ (ระบุ Asset Class เป็น Growth หรือ Defensive):")
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

        col_save1, col_save2 = st.columns([1, 4])
        with col_save1:
            if st.button("ล้างพอร์ตที่บันทึกไว้"):
                default_df = pd.DataFrame([
                    {"Ticker": "AAPL", "Shares": 10.0, "Buy Price (USD)": 170.0, "Asset Class": "Growth"},
                    {"Ticker": "TSLA", "Shares": 5.0, "Buy Price (USD)": 220.0, "Asset Class": "Growth"},
                    {"Ticker": "JNJ", "Shares": 8.0, "Buy Price (USD)": 150.0, "Asset Class": "Defensive"}
                ])
                st.session_state.portfolio_input = default_df
                save_portfolio_to_disk(default_df)
                st.rerun()

        with st.expander("ตั้งค่าเป้าหมายพอร์ตและความเสี่ยง"):
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                target_growth_pct = st.slider("เป้าหมายสัดส่วน Growth (%)", min_value=0, max_value=100, value=60, step=5)
            with col_t2:
                max_concentration = st.slider("เกณฑ์เตือนความกระจุกตัว หุ้นตัวเดียวเกิน (%)", min_value=10, max_value=50, value=25, step=5)

        df_res = pd.DataFrame()
        price_history_map = {}

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

                st.markdown("---")
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("มูลค่าลงทุนรวม", f"${total_invested:,.2f}")
                col_m2.metric("มูลค่าปัจจุบันรวม", f"${total_current:,.2f}")
                col_m3.metric("กำไร/ขาดทุนรวม", f"${total_pl:,.2f}", f"{total_pl_pct:.2f}%")

                # --- 1. ตรวจสอบสัดส่วน Growth vs Defensive & เป้าหมาย ---
                st.markdown("### 1. วิเคราะห์สัดส่วน Asset Class (Growth vs Defensive)")
                class_group = df_res.groupby('AssetClass')['Current Value ($)'].sum()
                current_growth_val = class_group.get('Growth', 0.0)
                current_def_val = class_group.get('Defensive', 0.0)
                actual_total = current_growth_val + current_def_val

                actual_growth_pct = (current_growth_val / actual_total * 100) if actual_total > 0 else 0.0
                actual_def_pct = (current_def_val / actual_total * 100) if actual_total > 0 else 0.0
                target_def_pct = 100.0 - target_growth_pct

                col_ac1, col_ac2 = st.columns(2)
                with col_ac1:
                    st.metric("สัดส่วน Growth ปัจจุบัน", f"{actual_growth_pct:.1f}%", f"เป้าหมาย: {target_growth_pct}%")
                with col_ac2:
                    st.metric("สัดส่วน Defensive ปัจจุบัน", f"{actual_def_pct:.1f}%", f"เป้าหมาย: {target_def_pct}%")

                growth_diff = abs(actual_growth_pct - target_growth_pct)
                if growth_diff > 10.0:
                    st.warning(f"⚠️ **เตือนความเบี่ยงเบนสัดส่วน:** สัดส่วน Growth ปัจจุบันเบี่ยงเบนจากเป้าหมาย {growth_diff:.1f}% (เกินกรอบ 10 จุด) แนะนำพิจารณา Rebalance พอร์ต")
                else:
                    st.success("✅ สัดส่วนสินทรัพย์อยู่ในกรอบเป้าหมายที่ยอมรับได้ (เบี่ยงเบนไม่เกิน 10%)")

                # --- 2. Concentration Risk ตรวจสอบความกระจุกตัว ---
                st.markdown("### 2. การตรวจสอบความเสี่ยงด้านความกระจุกตัว (Concentration Risk)")
                overweight_stocks = df_res[df_res['Weight in Portfolio (%)'] > max_concentration]
                if not overweight_stocks.empty:
                    for _, ow in overweight_stocks.iterrows():
                        st.error(f"🚨 **เตือนความกระจุกตัว:** หุ้น **{ow['Ticker']}** มีสัดส่วนสูงถึง {ow['Weight in Portfolio (%)']:.1f}% ของพอร์ต (เกินเกณฑ์เตือนที่ {max_concentration}%) ควรพิจารณาลดความเสี่ยงหรือกระจายลงทุนเพิ่ม")
                else:
                    st.success(f"✅ ไม่มีหุ้นตัวใดมีสัดส่วนเกินเกณฑ์เตือน ({max_concentration}%) ของพอร์ต")

                # --- 3. ตารางพอร์ตหลัก ---
                st.markdown("### 3. รายละเอียดพอร์ตและผลตอบแทนรายตัว")
                st.dataframe(
                    df_res[['Ticker', 'Shares', 'Buy Price (USD)', 'Current Price (USD)', 'Profit/Loss ($)', 'Profit/Loss (%)', 'Weight in Portfolio (%)', 'Asset Class', 'Simons Signal', 'Trend', 'Score']],
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

                # --- 4. บทวิเคราะห์หุ้นรายตัวในพอร์ต ---
                st.markdown("### 4. บทวิเคราะห์หุ้นรายตัวในพอร์ต (Individual Stock Analysis)")
                st.markdown("เจาะลึกสถานะเชิงปริมาณของแต่ละตัวที่คุณถืออยู่ เพื่อประกอบการตัดสินใจ Rebalance หรือถือต่อ")
                
                for _, r in df_res.iterrows():
                    t_sym = r['Ticker']
                    t_class = r['Asset Class']
                    t_sig = r['Simons Signal']
                    t_score = r['Score']
                    t_trend = r['Trend']
                    t_pl_pct = r['Profit/Loss (%)']
                    t_weight = r['Weight in Portfolio (%)']

                    with st.expander(f"📌 วิเคราะห์หุ้น: {t_sym} ({t_class}) — น้ำหนักในพอร์ต {t_weight:.1f}%"):
                        col_da1, col_da2, col_da3 = st.columns(3)
                        col_da1.metric("คะแนนโมเดล Simons", f"{t_score}/100 คะแนน", f"สถานะ: {t_sig}")
                        col_da2.metric("ผลกำไร/ขาดทุนสะสม", f"{t_pl_pct:+.2f}%")
                        col_da3.metric("แนวโน้มระยะสั้น", f"{t_trend}")

                        # ข้อความวิเคราะห์อัตโนมัติเชิงปริมาณ
                        analysis_text = f"**บทวิเคราะห์เชิงปริมาณสำหรับ {t_sym}:**\n"
                        if t_score >= 70:
                            analysis_text += f"- หุ้นตัวนี้มีความแข็งแกร่งสูงมากตามโมเดลควอนต์ (Score {t_score}) ราคาอยู่เหนือเส้นค่าเฉลี่ยและมีโมเมนตัมเชิงบวก เหมาะแก่การถือครองต่อหรือสะสมเพิ่ม\n"
                        elif t_score >= 45:
                            analysis_text += f"- หุ้นตัวนี้อยู่ในเกณฑ์ปานกลาง (Score {t_score}) โมเดลแนะนำให้ 'ถือ / เฝ้าสังเกต' ทิศทางราคาเพื่อรอสัญญาณชัดเจน\n"
                        else:
                            analysis_text += f"- หุ้นตัวนี้มีความอ่อนแอเชิงปริมาณ (Score {t_score}) สัญญาณเตือนให้ระมัดระวัง อาจพิจารณาตัดขายทำกำไรหรือตัดขาดทุนตามวินิจฉัย\n"

                        if t_pl_pct < -20:
                            analysis_text += f"- ⚠️ ขาดทุนสะสมค่อนข้างสูง ({t_pl_pct:.2f}%) ควรตรวจสอบว่าปัจจัยพื้นฐานหรือเทรนด์เปลี่ยนไปจากเดิมหรือไม่เพื่อป้องกันการจมทุน\n"
                        elif t_pl_pct > 30:
                            analysis_text += f"- 💡 ทำกำไรได้ดีเยี่ยม ({t_pl_pct:+.2f}%) พิจารณาล็อกกำไรบางส่วน (Take Profit) หากสัดส่วนในพอร์ตเริ่มใหญ่เกินไป\n"

                        if t_class == "Growth" and t_trend == "ขาลง":
                            analysis_text += f"- เนื่องจากจัดอยู่ในหมวด Growth แต่เทรนด์ระยะสั้นเป็นขาลง อาจมีความผันผวนสูงในช่วงที่ตลาดปรับฐาน\n"
                        elif t_class == "Defensive" and t_trend == "ขาขึ้น":
                            analysis_text += f"- หุ้นหมวด Defensive ตัวนี้ทำหน้าที่พอร์ตได้ดี มีความมั่นคงและให้ผลตอบแทนสอดคล้องกับทิศทางตลาดที่ดี\n"

                        st.markdown(analysis_text)

                # --- 5. Correlation Matrix ---
                st.markdown("### 5. Correlation Matrix (ความสัมพันธ์ระหว่างหุ้นในพอร์ต)")
                if len(price_history_map) >= 2:
                    price_df = pd.DataFrame(price_history_map).dropna()
                    if not price_df.empty:
                        returns_df = price_df.pct_change().dropna()
                        corr_matrix = returns_df.corr()

                        st.dataframe(corr_matrix.style.background_gradient(cmap="coolwarm", axis=None), use_container_width=True)

                        # วิเคราะห์หาคู่ที่สัมพันธ์สูงหรือติดลบ
                        high_corr_pairs = []
                        neg_corr_pairs = []
                        cols = corr_matrix.columns
                        for i in range(len(cols)):
                            for j in range(i + 1, len(cols)):
                                val = corr_matrix.iloc[i, j]
                                pair_name = f"{cols[i]} & {cols[j]}"
                                if val > 0.7:
                                    high_corr_pairs.append((pair_name, val))
                                elif val < 0:
                                    neg_corr_pairs.append((pair_name, val))

                        if high_corr_pairs:
                            st.warning("⚠️ **เตือนคู่หุ้นที่ Correlation สูง (>0.7):** หุ้นเหล่านี้เคลื่อนไหวไปในทิศทางเดียวกันเกือบตลอดเวลา ถือคู่กันอาจทำให้กระจายความเสี่ยงไม่ 
                            for p, v in high_corr_pairs:
                                st.write(f"- `{p}`: สัมประสิทธิ์สหสัมพันธ์ = {v:.2f}")
                        else:
                            st.success("✅ ไม่พบหุ้นคู่ใดในพอร์ตที่มีความสัมพันธ์สูงเกิน 0.7 (มีการกระจายความเสี่ยงที่ดี)")

                        if neg_corr_pairs:
                            st.success("💡 **คู่หุ้นที่ Correlation ติดลบ (ช่วย Hedge กันได้ดี):**")
                            for p, v in neg_corr_pairs:
                                st.write(f"- `{p}`: สัมประสิทธิ์สหสัมพันธ์ = {v:.2f} (ช่วยลดความผันผวนรวมของพอร์ตได้ดี)")
                    else:
                        st.info("ข้อมูลราคาประวัติศาสตร์ไม่เพียงพอสำหรับการคำนวณ Correlation")
                else:
                    st.info("กรุณาใส่หุ้นในพอร์ตอย่างน้อย 2 ตัว เพื่อคำนวณ Correlation Matrix")

                # --- 6. กราฟมูลค่าพอร์ตย้อนหลัง 6 เดือน (Equity Curve) ---
                st.markdown("### 6. ประมาณการกราฟมูลค่าพอร์ตย้อนหลัง 6 เดือน (Equity Curve)")
                st.caption("หมายเหตุ: กราฟนี้คำนวณโดยสมมติว่าถือจำนวนหุ้นปัจจุบันคงที่ย้อนหลังไป 6 เดือน เพื่อดูภาพรวมความเคลื่อนไหวของพอร์ต ไม่ใช่ผลตอบแทนจริงตามวันที่ซื้อจริง")

                if price_history_map:
                    combined_price_df = pd.DataFrame(price_history_map).dropna()
                    if not combined_price_df.empty:
                        # คำนวณมูลค่าพอร์ตในแต่ละวันย้อนหลังโดยใช้จำนวนหุ้นปัจจุบัน
                        shares_series = pd.Series({r['Ticker']: r['Shares'] for _, r in df_res.iterrows()})
                        # กรองเฉพาะหุ้นที่มีข้อมูลครบ
                        common_tickers = [t for t in shares_series.index if t in combined_price_df.columns]
                        if common_tickers:
                            sub_price = combined_price_df[common_tickers]
                            sub_shares = shares_series[common_tickers]
                            portfolio_value_series = (sub_price * sub_shares).sum(axis=1)

                            st.line_chart(portfolio_value_series)
                        else:
                            st.info("ไม่สามารถรวมข้อมูลกราฟพอร์ตได้เนื่องจากข้อมูลราคาไม่ตรงกัน")
                    else:
                        st.info("ไม่พบข้อมูลราคาเพียงพอสำหรับการสร้างกราฟย้อนหลัง")
                else:
                    st.info("ยังไม่มีข้อมูลหุ้นในพอร์ตสำหรับสร้างกราฟ")

    with tab3:
        st.subheader("พูดคุยกับ AI ปรมาจารย์ Jim Simons")
        openai_api_key = st.sidebar.text_input("ใส่ OpenAI API Key (สำหรับเปิดใช้งาน AI Chat)", type="password")

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

            if prompt := st.chat_input("ปรึกษาเรื่องพอร์ต Growth หรือสถิติตลาดกับ Jim Simons..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Jim Simons กำลังคำนวณโมเดลคำตอบ..."):
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
            st.info("กรอก OpenAI API Key ที่แถบซ้ายมือ (Sidebar) เพื่อเปิดใช้งานช่องแชทปรึกษาพอร์ต Growth กับ Jim Simons ครับ")