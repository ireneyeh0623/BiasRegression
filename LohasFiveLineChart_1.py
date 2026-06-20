import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ==============================================================================
# 1. 系統環境配置
# ==============================================================================

st.set_page_config(page_title="股價五線譜", layout="wide")

# ==============================================================================
# 2. 側邊欄：使用者參數輸入區
# ==============================================================================

st.sidebar.header("查詢設定")
# 股票代號輸入：預設為 2330.TW (台積電)
stock_id = st.sidebar.text_input("股票代號(如2330或AAPL)", "2330")
# 日期範圍選擇：設定資料擷取的起始與結束時間
start_date = st.sidebar.date_input("起始日期(YYYY/MM/DD)", datetime(2022, 10, 3))
end_date = st.sidebar.date_input("結束日期(YYYY/MM/DD)", datetime.now())
# 視覺主題切換：根據網頁環境調整背景顏色
theme_choice = st.sidebar.radio("圖表主題(對應網頁背景)", ["亮色(白色背景)", "深色(深色背景)"])
# 開始計算觸發按鈕
calculate_btn = st.sidebar.button("開始計算")

# ==============================================================================
# 3. 視覺樣式優化 (CSS 強制覆蓋背景色切換邏輯)
# ==============================================================================

# 根據主題選擇，動態調整 Streamlit 元件與圖表的顏色配比
if theme_choice == "深色(深色背景)":
    chart_template = "plotly_dark"
    font_color = "white"
    bg_color = "#0E1117"
    st.markdown("""
        <style>
        /* 強制側邊欄、主背景、文字顏色為深色 */
        [data-testid="stSidebar"], .stApp, header { background-color: #0E1117 !important; color: white !important; }
        .stMarkdown, p, h1, h2, h3, span { color: white !important; }

        /* 調整輸入框文字顏色 */
        input { color: white !important; background-color: #262730 !important; }
        </style>
        """, unsafe_allow_html=True)
else:
    chart_template = "plotly_white"
    font_color = "black"
    bg_color = "#FFFFFF"
    st.markdown("""
        <style>
        /* 1. 強制背景與文字顏色 */
        [data-testid="stSidebar"], .stApp, header { 
            background-color: #FFFFFF !important; 
            color: black !important; 
        }
        .stMarkdown, p, h1, h2, h3, span { color: black !important; }
        
        /* 2. 徹底消除輸入框右側的陰影與淡淡格線 */
        div[data-baseweb="input"], 
        div[data-baseweb="input"] > div,
        div[data-baseweb="input"] input {
            background-color: white !important;
            border-color: #dcdcdc !important; /* 設定一個淺灰色的統一邊框 */
            box-shadow: none !important;      /* 移除所有陰影 */
        }
        
        /* 針對日期選取器內部的特殊容器進行修正 */
        div[role="combobox"] {
            background-color: white !important;
            border: none !important;
        }

        /* 3. 強制按鈕內部的所有文字元素變白 */
        div.stButton > button {
            background-color: #000000 !important;
            border: 1px solid #000000 !important;
            font-weight: bold !important;
        }
        div.stButton > button * {
            color: #FFFFFF !important;
        }
        
        div.stButton > button:hover {
            background-color: #333333 !important;
        }

        /* 4. 側邊欄與輸入框整體調整 */
        [data-testid="stSidebar"] { border-right: 1px solid #f0f2f6; }
        input { 
            color: black !important; 
            background-color: white !important; 
        }
        </style>
        """, unsafe_allow_html=True)


# ==============================================================================
# 4. 主程式執行邏輯
# ==============================================================================

st.title("📈 股價五線譜")

# 判斷邏輯：如果按鈕「還沒被按下」
if not calculate_btn:
    st.info("💡 請點開左上角選單 [ >> ] 在左側面板設定參數後，按「開始計算」即可產出圖表")
# 判斷邏輯：按下按鈕後才執行抓取資料的動作：
else:
    # 若使用者已輸入含 . 的代號（如 2330.TW），直接使用；否則依序嘗試原始、.TW、.TWO
    if '.' in stock_id:
        candidates = [stock_id]
    else:
        candidates = [stock_id, f"{stock_id}.TW", f"{stock_id}.TWO"]

    data = pd.DataFrame()
    search_id = candidates[0]
    for candidate in candidates:
        _dl = yf.download(candidate, start=start_date, end=end_date, auto_adjust=True)
        if not _dl.empty:
            data = _dl
            search_id = candidate
            break

    # 顯示最終使用的股票代碼
    st.markdown(f"<h3 style='color: {font_color};'>{search_id}</h3>", unsafe_allow_html=True)
    
    if not data.empty:
        # 相容新版 yfinance MultiIndex 欄位結構：直接萃取 Close 欄位，避免 reset_index 後欄名不穩定
        close_series = data['Close']
        # 若為 DataFrame（新版 yfinance 多層索引會回傳 DataFrame），取第一欄
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]

        # 建立乾淨的工作用 DataFrame，明確指定欄位名稱
        df = pd.DataFrame({
            'Date': close_series.index,
            'Close_1D': close_series.values.flatten()
        }).reset_index(drop=True)

        # 移除時區資訊（yfinance 新版可能回傳 tz-aware 日期，Plotly 繪圖需純日期）
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

        # 排除無交易資料的日期 (NaN)，確保計算精確度
        df = df.dropna(subset=['Close_1D'])
        # 建立時間索引 (0, 1, 2...) 作為線性回歸的自變數 X
        df['Time_Idx'] = np.arange(len(df))
        
        # --- B. 核心原理：線性回歸與標準差軌道計算 ---
        if len(df) > 1:
            # 1. 線性回歸：找出股價的中長線趨勢中心
            # 公式為：$y = ax + b$ (a=斜率, b=截距)
            z = np.polyfit(df['Time_Idx'], df['Close_1D'], 1)
            p = np.poly1d(z)
            df['Trend_Line'] = p(df['Time_Idx']) # 產出回歸趨勢線數據
            
            # 2. 計算離差標準差 (Standard Deviation)：衡量股價對趨勢線的波動幅度
            # 標準差愈大，代表股價波動愈劇烈，軌道間距也會隨之拉大
            std_dev = (df['Close_1D'] - df['Trend_Line']).std()

            # 3. 建立五線譜軌道
            # 高點軌道：趨勢中線 + 1倍或2倍標準差
            # 低點軌道：趨勢中線 - 1倍或2倍標準差
            df['Upper_2'] = df['Trend_Line'] + 2 * std_dev # 極端樂觀線
            df['Upper_1'] = df['Trend_Line'] + 1 * std_dev # 樂觀線
            df['Lower_1'] = df['Trend_Line'] - 1 * std_dev # 悲觀線
            df['Lower_2'] = df['Trend_Line'] - 2 * std_dev # 極端悲觀線

            # --- C. 視覺化繪圖 (Plotly 互動式圖表) ---
            fig = go.Figure()
            
            # 設定收盤價曲線顏色，使其在不同背景下清晰可見
            line_color = "white" if theme_choice == "深色(深色背景)" else "black"
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close_1D'], name='收盤價', 
                                     line=dict(color=line_color, width=1.5)))
            
            # 定義各軌道線的顏色與名稱
            colors = ['#FF4136', '#FF851B', '#0074D9', '#2ECC40', '#3D9970']
            names = ['極端樂觀', '樂觀', '趨勢中線', '悲觀', '極端悲觀']
            bands = ['Upper_2', 'Upper_1', 'Trend_Line', 'Lower_1', 'Lower_2']
            
            for idx, band in enumerate(bands):
                # 趨勢中線使用實線，其餘 SD 軌道使用虛線以利區分
                fig.add_trace(go.Scatter(x=df['Date'], y=df[band], name=names[idx], 
                                         line=dict(dash='dash' if 'Trend' not in band else 'solid', 
                                                   color=colors[idx], width=1)))

            # 圖表版面優化
            fig.update_layout(
                height=600, 
                template=chart_template, 
                hovermode='x unified', # 統一滑鼠懸停資訊
                paper_bgcolor=bg_color,
                plot_bgcolor=bg_color,
                font=dict(color=font_color),
                xaxis=dict(color=font_color, tickfont=dict(color=font_color)),
                yaxis=dict(color=font_color, tickfont=dict(color=font_color)),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(color=font_color))
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- D. 數據摘要與投資評語 ---
            st.header("📊 數據摘要")
            last_price = df['Close_1D'].iloc[-1] # 最後一筆交易價格
            trend_val = df['Trend_Line'].iloc[-1] # 對應日期的趨勢線值
            # 計算目前股價落在幾倍標準差位置 ($\sigma$ 值)
            sigma = (last_price - trend_val) / std_dev
            
            # 儀表板數值呈現
            col1, col2, col3 = st.columns(3)
            col1.metric("最後收盤價", f"{last_price:.2f}")
            col2.metric("回歸中線", f"{trend_val:.2f}")
            col3.metric("目前區間", f"{sigma:.2f} σ")
            
            # 根據 $\sigma$ (Sigma) 值給予自動化投資評註
            if sigma > 2:
                st.error(f"⚠️ 目前價格極度高估（高於中線 {sigma:.2f} 個標準差）")
            elif sigma < -2:
                st.success(f"✅ 目前價格極度低估（低於中線 {abs(sigma):.2f} 個標準差）")
            else:
                st.info(f"💡 目前價格處於合理回歸區間內")
        else:
            st.warning("資料量不足以計算回歸線。")
    else:
        st.error(f"找不到股票資料（已嘗試：{', '.join(candidates)}），請檢查代號或日期。")