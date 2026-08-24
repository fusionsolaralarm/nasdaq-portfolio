import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import openai

st.set_page_config(
    page_title="Simon Screener & Portfolio AI",
    layout="wide"
)

st.title("Simon Screener - Quant & Portfolio AI Advisor")
st.markdown("ระบบสแกนหุ้นสหรัฐฯ และผู้ช่วยวิเคราะห์พอร์ตการลงทุนส่วนตัวในสไตล์ Jim Simons (Renaissance Technologies)")

@st.cache_data
def get_us_stock_symbols():
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

symbols = get_us_stock_symbols()

@st.cache_data(ttl=1800)
def fetch_stock_data_and_simons_logic(ticker_list):
    data_list = []
    for ticker in ticker_list:
        try:
            hist = yf.download(ticker, period="6mo", progress=False)
            if hist.empty or len(hist) < 30:
                continue

            if isinstance(hist.columns, pd.MultiIndex):
                close_series = hist['Close'].iloc[:, 0]
            else:
                close_series = hist['Close']

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

            # เทรนด์ระยะสั้น ใช้แยกระหว่าง "ขาลงจริง" กับ "แค่ราคาต่ำกว่าทุน"
            if return_1w > 0.5:
                trend = "ขาขึ้น"
            elif return_1w < -0.5:
                trend = "ขาลง"
            else:
                trend = "ไซด์เวย์"

            data_list.append({
                'Ticker': ticker,
                'Simons Signal': recommendation,
                'Score': simons_score,
                'Price (USD)': current_p,
                'Momentum 1M (%)': return_1m,
                'Momentum 1W (%)': return_1w,
                'Trend': trend,
                'Volatility (%)': volatility,
                'SMA 50': sma_50
            })
        except Exception:
            continue

    return pd.DataFrame(data_list)

with st.spinner("กำลังดึงราคาล่าสุดและคำนวณโมเดลตลาดสหรัฐฯ..."):
    df = fetch_stock_data_and_simons_logic(symbols)

if df.empty:
    st.error("ไม่สามารถดึงข้อมูลได้ในขณะนี้ กรุณารีเฟรชหน้าเว็บใหม่อีกครั้ง")
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
        st.markdown("กรอกข้อมูลหุ้น จำนวนหุ้น และราคาต้นทุนที่คุณซื้อมา เพื่อคำนวณกำไร/ขาดทุนและรับคำแนะนำรายตัว")

        if "portfolio_input" not in st.session_state:
            st.session_state.portfolio_input = pd.DataFrame([
                {"Ticker": "AAPL", "Shares": 10.0, "Buy Price (USD)": 170.0, "Asset Class": "Growth"},
                {"Ticker": "TSLA", "Shares": 5.0, "Buy Price (USD)": 220.0, "Asset Class": "Growth"}
            ])

        st.markdown("แก้ไขตารางด้านล่างนี้เพื่อใส่พอร์ตของคุณ (สามารถพิมพ์เพิ่มแถวหรือแก้ไขข้อมูลได้โดยตรง ใส่ Ticker เดิมซ้ำได้หากซื้อคนละไม้ ระบบจะรวมต้นทุนเฉลี่ยให้ / ระบุ Asset Class เป็น Growth หรือ Defensive เพื่อใช้เช็คสัดส่วน 60/40):")
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

        # ตั้งค่ากติกาความเสี่ยง
        with st.expander("ตั้งค่ากติกาความเสี่ยง (Risk Rules)"):
            stop_loss_pct = st.slider("Stop-Loss ตัดขาดทุนเมื่อขาดทุนเกิน (%)", min_value=5, max_value=50, value=30, step=5)
            max_holdings = st.slider("จำนวนหุ้นสูงสุดที่แนะนำถือ (กันการกระจายทุนมากเกินไป)", min_value=5, max_value=30, value=20)

        if not edited_portfolio.empty:
            # รวมแถว ticker ซ้ำ เป็นต้นทุนเฉลี่ยถ่วงน้ำหนัก (weighted average cost)
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
            price_history_map = {}  # เก็บ close series 6 เดือนของแต่ละ ticker ไว้ใช้ทำ correlation + equity curve
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

                if len(df_res) > max_holdings:
                    st.warning(f"พอร์ตของคุณมี {len(df_res)} ตัว ซึ่งเกินเกณฑ์ {max_holdings} ตัวที่ตั้งไว้ — อาจเข้าข่ายกระจายทุนมากเกินไป (Diworsification) ทำให้ผลตอบแทนเฉลี่ยลู่เข้าใกล้ตลาดโดยรวมมากขึ้นและลดประโยชน์จากการเลือกหุ้นรายตัว")

                # เช็คสัดส่วนเดี่ยวเกิน 25% ของพอร์ต (Concentration Risk)
                concentrated = df_res[df_res['Weight in Portfolio (%)'] > 25]
                if not concentrated.empty:
                    names = ", ".join(concentrated['Ticker'].tolist())
                    st.warning(f"หุ้น {names} มีสัดส่วนเกิน 25% ของพอร์ตต่อตัว — ความเสี่ยงกระจุกตัวสูง ควรพิจารณาปรับสมดุล (Rebalance) หากสัดส่วนเบี่ยงจากเป้าหมายเกิน 10%")

                st.markdown("### รายละเอียดพอร์ตและผลตอบแทนรายตัว")
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

                # --- สัดส่วน Growth / Defensive (แนวทาง 60/40) ---
                st.markdown("### สัดส่วนสินทรัพย์ Growth / Defensive (แนวทาง 60/40)")
                growth_weight = df_res.loc[df_res['Asset Class'] == 'Growth', 'Weight in Portfolio (%)'].sum()
                defensive_weight = df_res.loc[df_res['Asset Class'] == 'Defensive', 'Weight in Portfolio (%)'].sum()
                col_a1, col_a2 = st.columns(2)
                col_a1.metric("Growth", f"{growth_weight:.1f}%", f"เป้าหมาย 60%")
                col_a2.metric("Defensive", f"{defensive_weight:.1f}%", f"เป้าหมาย 40%")
                st.progress(min(int(growth_weight), 100) / 100)
                if abs(growth_weight - 60) > 10:
                    tilt = "เอียงไปทาง Growth มากเกินไป" if growth_weight > 60 else "เอียงไปทาง Defensive มากเกินไป"
                    st.warning(f"สัดส่วนปัจจุบันเบี่ยงจากกรอบ 60/40 เกิน 10 จุด ({tilt}) พิจารณาปรับสมดุลพอร์ตให้ใกล้เป้าหมายมากขึ้น")
                else:
                    st.success("สัดส่วน Growth/Defensive ยังอยู่ในกรอบ 60/40 ที่ยอมรับได้ (คลาดเคลื่อนไม่เกิน 10 จุด)")

                # --- Correlation Matrix เพื่อเช็คการกระจายความเสี่ยง ---
                valid_history = {t: s for t, s in price_history_map.items() if len(s) > 30}
                if len(valid_history) >= 2:
                    st.markdown("### Correlation Matrix ระหว่างหุ้นในพอร์ต (6 เดือนย้อนหลัง)")
                    st.caption("หลักการ: คู่หุ้นที่มีค่า Correlation สูง (ใกล้ +1) เคลื่อนไหวตามกันแทบไม่ช่วยกระจายความเสี่ยง ส่วนคู่ที่เป็นลบ (ใกล้ -1) ช่วยป้องกันความเสี่ยงซึ่งกันและกันได้ดีกว่า")
                    returns_df = pd.DataFrame({t: s.pct_change() for t, s in valid_history.items()}).dropna(how='all')
                    corr_matrix = returns_df.corr().round(2)
                    st.dataframe(corr_matrix, use_container_width=True)

                    high_corr_pairs = []
                    hedge_pairs = []
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
                        st.warning("คู่หุ้นที่เคลื่อนไหวตามกันสูง (Correlation > 0.7): " + ", ".join(high_corr_pairs) + " — ถือพร้อมกันจำนวนมากไม่ได้ช่วยกระจายความเสี่ยงเท่าที่ควร")
                    if hedge_pairs:
                        st.success("คู่หุ้นที่ช่วยกระจายความเสี่ยงกันได้ดี (Correlation ติดลบ): " + ", ".join(hedge_pairs))

                # --- กราฟมูลค่าพอร์ตย้อนหลัง (ประมาณการโดยสมมติจำนวนหุ้นปัจจุบันคงที่ตลอด 6 เดือน) ---
                if valid_history:
                    st.markdown("### กราฟมูลค่าพอร์ตย้อนหลัง (ประมาณการ)")
                    st.caption("หมายเหตุ: กราฟนี้สมมติว่าถือจำนวนหุ้นปัจจุบันคงที่ตลอด 6 เดือนที่ผ่านมา ใช้เพื่อดูทิศทางความผันผวนของพอร์ตโดยรวม ไม่ใช่ผลตอบแทนจริงตามวันที่ซื้อจริง")
                    shares_map = dict(zip(df_res['Ticker'], df_res['Shares']))
                    equity_df = pd.DataFrame({t: s * shares_map.get(t, 0) for t, s in valid_history.items()})
                    equity_curve = equity_df.sum(axis=1)
                    st.line_chart(equity_curve)

                st.markdown("### คำแนะนำเชิงกลยุทธ์รายตัว")
                st.caption("หลักการ: ขาดทุนอย่างเดียวไม่ใช่สัญญาณขาย — จะแนะนำขาย/ตัดขาดทุน เฉพาะเมื่อราคากำลังอยู่ในเทรนด์ขาลงจริง หรือขาดทุนเกินเกณฑ์ Stop-Loss ที่ตั้งไว้เท่านั้น")
                for _, r in df_res.iterrows():
                    pl_text = f"กำไร {r['Profit/Loss (%)']:.2f}%" if r['Profit/Loss (%)'] >= 0 else f"ขาดทุน {abs(r['Profit/Loss (%)']):.2f}%"
                    base_info = (
                        f"{r['Ticker']} (ต้นทุนเฉลี่ย: ${r['Buy Price (USD)']:.2f} | "
                        f"ราคาปัจจุบัน: ${r['Current Price (USD)']:.2f} | {pl_text} | "
                        f"สัดส่วนในพอร์ต: {r['Weight in Portfolio (%)']:.1f}%)"
                    )

                    if r['Profit/Loss (%)'] <= -stop_loss_pct:
                        st.error(f"{base_info}: ขาดทุนเกินเกณฑ์ Stop-Loss ที่ตั้งไว้ ({stop_loss_pct}%) แนะนำพิจารณาตัดขาดทุนเพื่อควบคุมความเสี่ยงของพอร์ต")
                    elif "ซื้อสะสม" in r['Simons Signal']:
                        st.info(f"{base_info}: สถานะพอร์ตอยู่ในเกณฑ์ดีและโมเดลประเมินว่าแข็งแกร่ง แนะนำให้ถือต่อหรือทยอยเพิ่มทุน")
                    elif "ควรขาย" in r['Simons Signal'] and r['Trend'] == "ขาลง":
                        st.error(f"{base_info}: สัญญาณทางสถิติอ่อนแอและราคากำลังอยู่ในเทรนด์ขาลงจริง แนะนำให้พิจารณาขายหรือลดสัดส่วน")
                    elif "ควรขาย" in r['Simons Signal']:
                        st.warning(f"{base_info}: สัญญาณทางสถิติอ่อนแอ แต่ราคายังไม่ยืนยันเทรนด์ขาลงชัดเจน (แค่ต่ำกว่าทุนไม่ใช่สัญญาณขายในตัวมันเอง) แนะนำเฝ้าดูอย่างใกล้ชิดก่อนตัดสินใจ")
                    else:
                        st.warning(f"{base_info}: สัญญาณอยู่ในโซนพักตัว แนะนำให้ถือรอดูสถานะต่อไป")

    with tab3:
        st.subheader("พูดคุยกับ AI ปรมาจารย์ Jim Simons")
        openai_api_key = st.sidebar.text_input("ใส่ OpenAI API Key (สำหรับเปิดใช้งาน AI Chat)", type="password")

        if openai_api_key:
            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {"role": "system", "content": "คุณคือ Jim Simons ปรมาจารย์กองทุน Quantitative ระดับโลก (Renaissance Technologies) คอยให้คำแนะนำเรื่องพอร์ตหุ้น สถิติตลาด และการลงทุนด้วยตรรกะคณิตศาสตร์อย่างเป็นกันเอง ห้ามใช้อีโมจิเด็ดขาด"}
                ]

            for message in st.session_state.messages:
                if message["role"] != "system":
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            if prompt := st.chat_input("ปรึกษาเรื่องพอร์ตหรือหุ้นกับ Jim Simons..."):
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
            st.info("กรอก OpenAI API Key ที่แถบซ้ายมือ (Sidebar) เพื่อเปิดใช้งานช่องแชทปรึกษาพอร์ตกับ Jim Simons ครับ")