import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import openai

st.set_page_config(
    page_title="Simon Screener & Ultimate Portfolio AI",
    layout="wide"
)

st.title("Simon Screener - Ultimate Quant & Portfolio AI Advisor")
st.markdown("ระบบสแกนหุ้นสหรัฐฯ วิเคราะห์ความเสี่ยงพอร์ต และผู้ช่วย AI ระดับเฮดจ์ฟันด์ (Renaissance Technologies)")

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
            
            return_1m = (current_p - price_1m_ago) / price_1m_ago * 100
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

            data_list.append({
                'Ticker': ticker,
                'Simons Signal': recommendation,
                'Score': simons_score,
                'Price (USD)': current_p,
                'Momentum 1M (%)': return_1m,
                'Volatility (%)': volatility,
                'SMA 50': sma_50
            })
        except Exception:
            continue
            
    return pd.DataFrame(data_list)

with st.spinner("กำลังดึงราคาล่าสุดและคำนวณโมเดลเชิงปริมาณตลาดสหรัฐฯ..."):
    df = fetch_stock_data_and_simons_logic(symbols)

if df.empty:
    st.error("ไม่สามารถดึงข้อมูลได้ในขณะนี้ กรุณารีเฟรชหน้าเว็บใหม่อีกครั้ง")
else:
    tab1, tab2, tab3 = st.tabs(["สแกนหุ้นตลาด", "พอร์ตของฉัน (ขั้นสูง)", "คุยกับ AI Jim Simons"])
    
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
                "Volatility (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "Score": st.column_config.NumberColumn(format="%d คะแนน"),
            },
            use_container_width=True,
            hide_index=True
        )

    with tab2:
        st.subheader("พอร์ตการลงทุนของคุณและการวิเคราะห์ความเสี่ยงเชิงปริมาณ")
        st.markdown("กรอกข้อมูลหุ้น จำนวนหุ้น และราคาต้นทุน ระบบจะคำนวณกำไร/ขาดทุน สัดส่วนน้ำหนักพอร์ต และประเมินความเสี่ยงรวมให้อัตโนมัติ")
        
        if "portfolio_input" not in st.session_state:
            st.session_state.portfolio_input = pd.DataFrame([
                {"Ticker": "AAPL", "Shares": 10.0, "Buy Price (USD)": 170.0},
                {"Ticker": "TSLA", "Shares": 5.0, "Buy Price (USD)": 220.0},
                {"Ticker": "NVDA", "Shares": 8.0, "Buy Price (USD)": 110.0}
            ])
            
        edited_portfolio = st.data_editor(st.session_state.portfolio_input, num_rows="dynamic", use_container_width=True)
        
        if not edited_portfolio.empty:
            portfolio_results = []
            for index, row in edited_portfolio.iterrows():
                ticker = str(row['Ticker']).strip().upper()
                shares = float(row['Shares'])
                buy_price = float(row['Buy Price (USD)'])
                
                if not ticker or shares <= 0 or buy_price <= 0:
                    continue
                    
                current_price = 0.0
                signal = "ถือประเมินสถานะ"
                score = 50
                volatility = 30.0
                
                match_row = df[df['Ticker'] == ticker]
                if not match_row.empty:
                    current_price = float(match_row.iloc[0]['Price (USD)'])
                    signal = match_row.iloc[0]['Simons Signal']
                    score = int(match_row.iloc[0]['Score'])
                    volatility = float(match_row.iloc[0]['Volatility (%)'])
                else:
                    try:
                        hist = yf.download(ticker, period="1mo", progress=False)
                        if not hist.empty:
                            current_price = float(hist['Close'].iloc[-1])
                    except Exception:
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
                    'Volatility (%)': volatility,
                    'Simons Signal': signal,
                    'Score': score
                })
                
            if portfolio_results:
                df_res = pd.DataFrame(portfolio_results)
                
                total_invested = df_res['Invested Value ($)'].sum()
                total_current = df_res['Current Value ($)'].sum()
                total_pl = total_current - total_invested
                total_pl_pct = (total_pl / total_invested * 100) if total_invested > 0 else 0.0
                
                if total_current > 0:
                    df_res['Allocation (%)'] = (df_res['Current Value ($)'] / total_current) * 100
                else:
                    df_res['Allocation (%)'] = 0.0
                
                # คำนวณความเสี่ยงรวมของพอร์ต (Weighted Portfolio Volatility)
                weighted_volatility = (df_res['Volatility (%)'] * (df_res['Allocation (%)'] / 100)).sum()
                
                st.markdown("---")
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("มูลค่าลงทุนรวม", f"${total_invested:,.2f}")
                col_m2.metric("มูลค่าปัจจุบันรวม", f"${total_current:,.2f}")
                col_m3.metric("กำไร/ขาดทุนรวม", f"${total_pl:,.2f}", f"{total_pl_pct:.2f}%")
                col_m4.metric("ความเสี่ยงรวมพอร์ต (Vol)", f"{weighted_volatility:.2f}%")
                
                st.markdown("### สัดส่วนและผลตอบแทนรายตัวในพอร์ต")
                st.dataframe(
                    df_res[['Ticker', 'Shares', 'Buy Price (USD)', 'Current Price (USD)', 'Allocation (%)', 'Profit/Loss ($)', 'Profit/Loss (%)', 'Simons Signal', 'Score']],
                    column_config={
                        "Buy Price (USD)": st.column_config.NumberColumn(format="$%.2f"),
                        "Current Price (USD)": st.column_config.NumberColumn(format="$%.2f"),
                        "Allocation (%)": st.column_config.NumberColumn(format="%.2f%%"),
                        "Profit/Loss ($)": st.column_config.NumberColumn(format="$%.2f"),
                        "Profit/Loss (%)": st.column_config.NumberColumn(format="%.2f%%"),
                        "Score": st.column_config.NumberColumn(format="%d คะแนน"),
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown("### กราฟแสดงสัดส่วนเงินลงทุนในพอร์ต (Allocation)")
                chart_data = df_res.set_index('Ticker')['Current Value ($)']
                st.bar_chart(chart_data)
                
                st.markdown("### คำแนะนำเชิงกลยุทธ์และจัดระเบียบพอร์ต (Portfolio Rebalancing)")
                for _, r in df_res.iterrows():
                    pl_text = f"กำไร {r['Profit/Loss (%)']:.2f}%" if r['Profit/Loss (%)'] >= 0 else f"ขาดทุน {r['Profit/Loss (%)']:.2f}%"
                    
                    # คำแนะนำตามหลักการน้ำหนักและการกระจายความเสี่ยง
                    alloc_advice = ""
                    if r['Allocation (%)'] > 40:
                        alloc_advice = " คำเตือน: หุ้นตัวนี้มีสัดส่วนกระจุกตัวสูงเกิน 40% ของพอร์ต ควรกระจายความเสี่ยงลดน้ำหนักลง"
                    
                    if "ซื้อสะสม" in r['Simons Signal']:
                        st.info(f"{r['Ticker']} (ต้นทุน: ${r['Buy Price (USD)']:.2f} | ปัจจุบัน: ${r['Current Price (USD)']:.2f} | น้ำหนัก: {r['Allocation (%)']:.2f}% | {pl_text}): สภาพพอร์ตแข็งแกร่ง แนะนำถือต่อหรือทยอยเพิ่มทุน{alloc_advice}")
                    elif "ถือ" in r['Simons Signal']:
                        st.warning(f"{r['Ticker']} (ต้นทุน: ${r['Buy Price (USD)']:.2f} | ปัจจุบัน: ${r['Current Price (USD)']:.2f} | น้ำหนัก: {r['Allocation (%)']:.2f}% | {pl_text}): สัญญาณอยู่ในโซนพักตัว แนะนำถือรอดูสถานะต่อไป{alloc_advice}")
                    else:
                        st.error(f"{r['Ticker']} (ต้นทุน: ${r['Buy Price (USD)']:.2f} | ปัจจุบัน: ${r['Current Price (USD)']:.2f} | น้ำหนัก: {r['Allocation (%)']:.2f}% | {pl_text}): สัญญาณทางสถิติอ่อนแอและความเสี่ยงสูง แนะนำพิจารณาขายทำกำไรหรือตัดขาดทุน")
                
                # ปุ่มดาวน์โหลดสรุปพอร์ตเป็น CSV
                csv_data = df_res.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="ดาวน์โหลดรายงานสรุปพอร์ต (CSV)",
                    data=csv_data,
                    file_name="my_quant_portfolio_report.csv",
                    mime="text/csv"
                )

    with tab3:
        st.subheader("พูดคุยกับ AI ปรมาจารย์ Jim Simons")
        openai_api_key = st.sidebar.text_input("ใส่ OpenAI API Key (สำหรับเปิดใช้งาน AI Chat)", type="password")

        if openai_api_key:
            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {"role": "system", "content": "คุณคือ Jim Simons ปรมาจารย์กองทุน Quantitative ระดับโลก (Renaissance Technologies) คอยให้คำแนะนำเรื่องพอร์ตหุ้น การบริหารความเสี่ยง สถิติตลาด และการลงทุนด้วยตรรกะคณิตศาสตร์อย่างเป็นทางการ ห้ามใช้อีโมจิเด็ดขาด"}
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