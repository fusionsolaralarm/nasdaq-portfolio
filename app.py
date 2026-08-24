import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import openai

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Simon Screener",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Simon Screener - Quantitative Model & AI Assistant")
st.markdown("ระบบสแกนหุ้นและผู้ช่วย AI อัจฉริยะในสไตล์กองทุน Renaissance Technologies (Jim Simons)")

# ฟังก์ชันดึงรายชื่อหุ้นทั้งหมด (S&P 500 + หุ้นเติบโต/หุ้นจิ๋ว)
@st.cache_data
def get_us_stock_symbols():
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df_sp500 = table[0]
        symbols = df_sp500['Symbol'].tolist()
        symbols = [s.replace('.', '-') for s in symbols]
        
        extra_stocks = [
            "ASTS", "LUNR", "IONQ", "SOUN", "RGTI", "QBTS", "ACHR", "JOBY", 
            "PLTR", "RIVN", "HOOD", "ARM", "SMCI", "RDDT", "DJT", "BBAI", 
            "HLGN", "CLSK", "RIOT", "MARA", "HUT", "BITF", "OPEN", "LMND"
        ]
        for s in extra_stocks:
            if s not in symbols:
                symbols.append(s)
        return symbols
    except Exception:
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "ASTS", "LUNR", "IONQ", "PLTR", "SMCI"]

symbols = get_us_stock_symbols()

@st.cache_data(ttl=3600)
def fetch_stock_data_and_simons_logic(ticker_list):
    data_list = []
    target_symbols = ticker_list[:180]
    
    for ticker in target_symbols:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            name = info.get('longName', ticker)
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            market_cap = info.get('marketCap', 0)
            pe_ratio = info.get('trailingPE', np.nan)
            roe = info.get('returnOnEquity', np.nan)
            
            hist = stock.history(period="6mo")
            if not hist.empty and len(hist) > 50:
                current_p = hist['Close'].iloc[-1]
                sma_50 = hist['Close'].rolling(50).mean().iloc[-1]
                return_1m = (current_p - hist['Close'].iloc[-20]) / hist['Close'].iloc[-20] * 100
                volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100
            else:
                return_1m = 0
                sma_50 = price
                volatility = 50

            simons_score = 50 
            if pd.notna(pe_ratio) and pe_ratio > 0:
                if pe_ratio < 25: simons_score += 15
                elif pe_ratio > 60: simons_score -= 10
            
            if price > sma_50: simons_score += 20
            else: simons_score -= 15
                
            if return_1m > 0: simons_score += 10
            else: simons_score -= 10
                
            if pd.notna(roe) and roe > 0.15: simons_score += 15
                
            if simons_score >= 75: recommendation = "🟢 ซื้อสะสม (Strong Buy)"
            elif simons_score >= 55: recommendation = "🟡 ถือ / เฝ้าสังเกต (Hold)"
            else: recommendation = "🔴 ควรขาย / หลีกเลี่ยง (Sell / Avoid)"

            data_list.append({
                'Ticker': ticker,
                'Name': name,
                'Simons Signal': recommendation,
                'Score': simons_score,
                'Price (USD)': price,
                'Market Cap ($)': market_cap,
                'P/E Ratio': pe_ratio,
                'ROE (%)': roe * 100 if roe else np.nan,
                'Momentum 1M (%)': return_1m,
                'Volatility (%)': volatility,
                'Sector': sector,
                'Industry': industry
            })
        except Exception:
            continue
            
    return pd.DataFrame(data_list)

with st.spinner("🤖 กำลังประมวลผลโมเดลสถิติตลาดสหรัฐฯ..."):
    df = fetch_stock_data_and_simons_logic(symbols)

if df.empty:
    st.error("ไม่สามารถดึงข้อมูลได้ในขณะนี้")
else:
    st.sidebar.header("🔍 ตัวกรองเชิงปริมาณ (Filters)")
    
    signal_filter = st.sidebar.selectbox("คำแนะนำจากโมเดล", ["ทั้งหมด", "🟢 ซื้อสะสม (Strong Buy)", "🟡 ถือ / เฝ้าสังเกต (Hold)", "🔴 ควรขาย / หลีกเลี่ยง (Sell / Avoid)"])
    if signal_filter != "ทั้งหมด":
        df_filtered = df[df['Simons Signal'] == signal_filter]
    else:
        df_filtered = df.copy()

    search_query = st.sidebar.text_input("ค้นหา Ticker หรือชื่อบริษัท", "").upper()
    if search_query:
        df_filtered = df_filtered[df_filtered['Ticker'].str.contains(search_query) | df_filtered['Name'].str.upper().str.contains(search_query)]
        
    sectors = ['ทั้งหมด'] + list(df['Sector'].dropna().unique())
    selected_sector = st.sidebar.selectbox("กลุ่มอุตสาหกรรม (Sector)", sectors)
    if selected_sector != 'ทั้งหมด':
        df_filtered = df_filtered[df_filtered['Sector'] == selected_sector]

    st.subheader(f"📊 ผลการสแกนโมเดลเชิงปริมาณ ({len(df_filtered)} บริษัท)")
    
    df_filtered = df_filtered.sort_values(by='Score', ascending=False)

    st.dataframe(
        df_filtered[['Ticker', 'Name', 'Simons Signal', 'Score', 'Price (USD)', 'P/E Ratio', 'ROE (%)', 'Momentum 1M (%)', 'Sector']],
        column_config={
            "Price (USD)": st.column_config.NumberColumn(format="$%.2f"),
            "P/E Ratio": st.column_config.NumberColumn(format="%.2f"),
            "ROE (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Momentum 1M (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Score": st.column_config.NumberColumn(format="%d คะแนน"),
        },
        width="stretch",
        hide_index=True
    )
    
    st.markdown("---")
    st.subheader("🔍 เจาะลึกมุมมองรายตัวหุ้น")
    selected_ticker = st.selectbox("เลือกหุ้นเพื่อดูกราฟและวิเคราะห์", df['Ticker'].unique())
    
    if selected_ticker:
        row = df[df['Ticker'] == selected_ticker].iloc[0]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("คำแนะนำ", row['Simons Signal'])
            st.metric("Score", f"{row['Score']} / 100")
        with col2:
            st.metric("ราคา", f"${row['Price (USD)']:.2f}")
            st.metric("โมเมนตัม 1 เดือน", f"{row['Momentum 1M (%)']:.2f}%")
        with col3:
            st.metric("ความผันผวน", f"{row['Volatility (%)']:.2f}%")
            st.metric("P/E Ratio", f"{row['P/E Ratio']:.2f}" if pd.notna(row['P/E Ratio']) else "N/A")
            
        st.write(f"**บริษัท:** {row['Name']} ({row['Sector']} / {row['Industry']})")
        
        hist_data = yf.Ticker(selected_ticker).history(period="1y")
        if not hist_data.empty:
            st.line_chart(hist_data['Close'])

# --- ส่วนของ AI Chat: พูดคุยกับ Jim Simons ---
st.markdown("---")
st.subheader("💬 พูดคุยกับ AI ปรมาจารย์ Jim Simons")
st.markdown("พิมพ์พูดคุย สอบถามเทคนิคการลงทุนเชิงปริมาณ (Quant) หรือปรึกษาเรื่องหุ้นกับ AI ได้เลยครับ")

openai_api_key = st.sidebar.text_input("🔑 ใส่ OpenAI API Key (สำหรับเปิดใช้งาน AI Chat)", type="password")

if openai_api_key:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "คุณคือ Jim Simons ปรมาจารย์กองทุน Quantitative ระดับโลก (Renaissance Technologies) คอยให้คำแนะนำเรื่องหุ้น สถิติตลาด และการลงทุนด้วยตรรกะคณิตศาสตร์อย่างเป็นกันเอง"}
        ]
        
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
    if prompt := st.chat_input("พิมพ์คำถามถึง Jim Simons ที่นี่..."):
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
    st.info("💡 **วิธีเปิดใช้งาน AI Chat:** ให้กรอก **OpenAI API Key** ของคุณลงในช่องว่างที่ Sidebar ด้านซ้ายมือ หน้าต่างแชทพูดคุยกับ Jim Simons จะปรากฏขึ้นมาทันทีครับ!")