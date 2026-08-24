import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ตั้งค่าหน้าเว็บให้เป็นแบบกว้าง
st.set_page_config(
    page_title="Nick's Portfolio - NASDAQ Screener", layout="wide"
)

st.title("📊 Nick's Portfolio: NASDAQ Growth Screener")
st.markdown(
    "พอร์ตจำลอง 1,000 USD | แนวคิด Scale Economies & Quality Compounder |"
    " Rebalance ทุก 2-3 เดือน"
)

# รายชื่อหุ้นใน NASDAQ
nasdaq_growth_watchlist = [
    "NVDA",
    "MSFT",
    "AAPL",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AVGO",
    "NFLX",
    "AMD",
    "PLTR",
    "NOW",
    "MELI",
    "MU",
    "CRWD",
]


# ฟังก์ชันดึงข้อมูลหุ้น
@st.cache_data(ttl=1800)
def load_data():
  results = []
  for ticker in nasdaq_growth_watchlist:
    try:
      stock = yf.Ticker(ticker)
      info = stock.info
      profit_margin = info.get("profitMargins", np.nan)
      revenue_growth = info.get("revenueGrowth", np.nan)

      data = stock.history(period="1y")
      if data.empty:
        continue

      close_price = data["Close"].iloc[-1]
      sma_50 = data["Close"].rolling(window=50).mean().iloc[-1]
      std_20 = data["Close"].rolling(window=20).std().iloc[-1]
      rolling_mean = data["Close"].rolling(window=20).mean().iloc[-1]

      lower_band = rolling_mean - (std_20 * 2)
      upper_band = rolling_mean + (std_20 * 2)

      support = round(float(min(lower_band, sma_50)), 2)
      resistance = round(float(upper_band), 2)

      score = 0
      if not np.isnan(revenue_growth) and revenue_growth > 0.10:
        score += 3
      if not np.isnan(profit_margin) and profit_margin > 0.15:
        score += 3
      if close_price <= support * 1.05:
        score += 4

      results.append({
          "Ticker": ticker,
          "Close ($)": round(float(close_price), 2),
          "Support ($)": support,
          "Resistance ($)": resistance,
          (
              "Rev_Growth"
          ): (
              f"{round(revenue_growth * 100, 1)}%"
              if not np.isnan(revenue_growth)
              else "N/A"
          ),
          (
              "Profit_Margin"
          ): (
              f"{round(profit_margin * 100, 1)}%"
              if not np.isnan(profit_margin)
              else "N/A"
          ),
          "Score": score,
      })
    except Exception:
      continue
  return pd.DataFrame(results)


# ปุ่มกดรีเฟรชข้อมูลเรียลไทม์
if st.button("🔄 ดึงข้อมูลและอัปเดตราคาแบบเรียลไทม์"):
  st.cache_data.clear()
  st.success("อัปเดตข้อมูลล่าสุดเรียบร้อยแล้ว!")

df = load_data()

if not df.empty:
  df_top10 = df.sort_values(by="Score", ascending=False).head(10)

  st.subheader("🏆 Top 10 Growth Stocks (Selected)")
  st.dataframe(df_top10, use_container_width=True)

  st.subheader("📈 เปรียบเทียบราคาปัจจุบันกับแนวรับ-แนวต้าน")
  chart_data = df_top10.set_index("Ticker")[
      ["Close ($)", "Support ($)", "Resistance ($)"]
  ]
  st.bar_chart(chart_data)

  st.markdown("---")
  st.subheader("📌 กฎการบริหารพอร์ต (Portfolio Rules)")
  st.markdown("""
    * **Capital:** เริ่มต้นพอร์ต 1,000 USD (จัดสรรลงทุนในหุ้น Top 10 ตัวละประมาณ 100 USD)
    * **Rebalance Rule:** ทำการตรวจสุขภาพพอร์ตและ Rebalance ทุกๆ 2-3 เดือน
    * **Kill Condition:** ตัดใจขายทันทีเมื่อปัจจัยพื้นฐานเปลี่ยน (Thesis Broken) หรือยอดขายหดตัวต่อเนื่อง
    """)
else:
  st.error("กำลังโหลดข้อมูล กรุณารอสักครู่...")
  