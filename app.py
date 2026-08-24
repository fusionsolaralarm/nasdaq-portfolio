import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import openai

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Simon Screener & Portfolio AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Simon Screener - Quant & Portfolio AI Advisor")
st.markdown("ระบบสแกนหุ้นสหรัฐฯ และผู้ช่วยวิเคราะห์พอร์ตการลงทุนส่วนตัวในสไตล์ Jim Simons (Renaissance Technologies)")

# รายชื่อหุ้นตลาดสหรัฐฯ สำหรับสแกน
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

@st.cache_data(ttl=1800) # แคชข้อมูล 30 นาทีเพื่อให้ดึงราคาล่าสุดรวดเร็ว
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

            # ตรรกะ Quant Score แบบ Jim Simons
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
                recommendation = "🟢 ซื้อสะสม (Strong Buy)"
            elif simons_score >= 45: 
                recommendation = "🟡 ถือ / เฝ้าสังเกต (Hold)"
            else: 
                recommendation = "🔴 ควรขาย / หลีกเลี่ยง (Sell / Avoid)"

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

with st.spinner("🤖 กำลังดึงราคาล่าสุดและคำนวณโมเดลตลาดสหรัฐฯ..."):
    df = fetch_stock_data_and_simons_logic(symbols)

if df.empty:
    st.error("ไม่สามารถดึงข้อมูลได้ในขณะนี้ กรุณารีเฟรชหน้าเว็บใหม่อีกครั้ง")
else:
    # แบ่งหน้าจอเป็น Tab หลัก: 1. สแกนหุ้น 2. วิเคราะห์พอร์ตส่วนตัว 3. แชทร่วมกับ AI
    tab1, tab2, tab3 = st.tabs(["📊 สแกนหุ้นตลาด (Market Screener)", "💼 จัดการและวิเคราะห์พอร์ตของฉัน (My Portfolio)", "💬 คุยกับ AI Jim Simons"])
    
    with tab1:
        st.subheader("📊 ตารางสแกนหุ้นสหรัฐฯ เชิงปริมาณ")
        signal_filter = st.selectbox("กรองตามคำแนะนำโมเดล", ["ทั้งหมด", "🟢 ซื้อสะสม (Strong Buy)", "🟡 ถือ / เฝ้าสังเกต (Hold)", "🔴 ควรขาย / หลีกเลี่ยง (Sell / Avoid)"])
        
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
            width="stretch",
            hide_index=True
        )

    with tab2:
        st.subheader("💼 พอร์ตการลงทุนของคุณ (My Portfolio Analysis)")
        st.markdown("กรอกชื่อหุ้นและจำนวนหุ้นที่คุณถืออยู่ เพื่อให้โมเดล AI ตรวจสอบสถานะและคำแนะนำซื้อ/ขายรายตัวทันที")
        
        # ฟอร์มกรอกพอร์ต
        col_input1, col_input2 = st.columns(2)
        with col_input1:
            user_tickers = st.text_input("ระบุ Ticker หุ้นที่คุณถือ (คั่นด้วยคอมมา เช่น AAPL, TSLA, NVDA)", "AAPL, MSFT, TSLA")
        
        if user_tickers:
            tickers_list = [t.strip().upper() for t in user_tickers.split(",") if t.strip()]
            portfolio_data = []
            
            for t in tickers_list:
                # ดึงข้อมูลหุ้นตัวนั้นๆ สดๆ
                match_row = df[df['Ticker'] == t]
                if not match_row.empty:
                    row = match_row.iloc[0].to_dict()
                    portfolio_data.append(row)
                else:
                    # กรณีหุ้นนอกเหนือจากลิสต์สแกน ให้ดึงแยกเฉพาะตัว
                    try:
                        hist = yf.download(t, period="3mo", progress=False)
                        if not hist.empty:
                            cp = float(hist['Close'].iloc[-1])
                            portfolio_data.append({
                                'Ticker': t,
                                'Simons Signal': "🟡 ถือประเมินสถานะ",
                                'Score': 50,
                                'Price (USD)': cp,
                                'Momentum 1M (%)': 0,
                                'Volatility (%)': 30,
                                'SMA 50': cp
                            })
                    except Exception:
                        continue
                        
            if portfolio_data:
                df_port = pd.DataFrame(portfolio_data)
                st.markdown("### 📋 ผลวิเคราะห์พอร์ตของคุณโดย Jim Simons Model:")
                st.dataframe(
                    df_port[['Ticker', 'Simons Signal', 'Score', 'Price (USD)', 'Momentum 1M (%)', 'Volatility (%)']],
                    column_config={
                        "Price (USD)": st.column_config.NumberColumn(format="$%.2f"),
                        "Momentum 1M (%)": st.column_config.NumberColumn(format="%.2f%%"),
                        "Score": st.column_config.NumberColumn(format="%d คะแนน"),
                    },
                    width="stretch",
                    hide_index=True
                )
                
                # คำแนะนำเชิงลึกสำหรับพอร์ต
                st.markdown("---")
                st.markdown("### 💡 คำแนะนำเพิ่มเติมสำหรับพอร์ตของคุณ:")
                for index, row in df_port.iterrows():
                    if "ซื้อสะสม" in row['Simons Signal']:
                        st.success(f"**{row['Ticker']}**: โมเดลประเมินว่าอยู่ในเกณฑ์แข็งแกร่ง (Score: {row['Score']}) โมเมนตัมเป็นบวก เหมาะสมที่จะ **ถือต่อหรือทยอยซื้อเพิ่ม**")
                    elif "ถือ" in row['Simons Signal']:
                        st.warning(f"**{row['Ticker']}**: สัญญาณอยู่ในโซนพักตัว (Score: {row['Score']}) แนะนำให้ **ถือรอดูสถานะ** และคุมความเสี่ยง")
                    else:
                        st.error(f"**{row['Ticker']}**: สัญญาณทางสถิติอ่อนแอและความเสี่ยงสูง (Score: {row['Score']}) โมเดลแนะนำให้ **พิจารณาขายทำกำไรหรือตัดขาดทุน** เพื่อความปลอดภัยของพอร์ต")
            else:
                st.warning("ไม่พบข้อมูลหุ้นที่คุณกรอก กรุณาตรวจสอบตัวย่อ Ticker อีกครั้ง")

    with tab3:
        st.subheader("💬 พูดคุยกับ AI ปรมาจารย์ Jim Simons")
        openai_api_key = st.sidebar.text_input("🔑 ใส่ OpenAI API Key (สำหรับเปิดใช้งาน AI Chat)", type="password")

        if openai_api_key:
            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {"role": "system", "content": "คุณคือ Jim Simons ปรมาจารย์กองทุน Quantitative ระดับโลก (Renaissance Technologies) คอยให้คำแนะนำเรื่องพอร์ตหุ้น สถิติตลาด และการลงทุนด้วยตรรกะคณิตศาสตร์อย่างเป็นกันเอง"}
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
            st.info("💡 กรอก OpenAI API Key ที่แถบซ้ายมือ (Sidebar) เพื่อเปิดใช้งานช่องแชทปรึกษาพอร์ตกับ Jim Simons ครับ!")