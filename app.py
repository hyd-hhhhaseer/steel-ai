import streamlit as st
import pandas as pd
import xgboost as xgb
import plotly.express as px
import os

# 1. 页面基础配置
st.set_page_config(page_title="材料工程垂直大模型", page_icon="🔩", layout="wide")

# 2. 数据加载与缓存
@st.cache_resource
def load_data_and_model():
    # 尝试读取数据，文件名必须与上传的一致
    file_name = "钢型（数据清洗）.xlsx - Sheet1.csv"
    
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
    else:
        # 如果还没上传CSV，生成演示数据避免报错
        st.warning("⚠️ 尚未检测到 CSV 数据文件，正在使用【演示模式】。请在 GitHub 上传 '钢型（数据清洗）.xlsx - Sheet1.csv'")
        data = {
            '对比项目': ['Demo-Steel-A', 'Demo-Steel-B', 'Demo-Steel-C'],
            '材料说明': ['高耐磨冷作模具钢', '耐腐蚀塑料模具钢', '通用热作模具钢'],
            '适用标准': ['GB/T Demo', 'ISO Demo', 'ASTM Demo'],
            'C_Avg': [1.5, 0.4, 0.38], 'Cr_Avg': [12.0, 13.0, 5.0], 
            'Mn_Avg': [0.4, 0.5, 0.4], 'Mo_Avg': [0.5, 0.0, 1.3],
            'Ni_Avg': [0.0, 0.0, 0.0], 'V_Avg': [0.3, 0.0, 1.0],
            'HRC_Avg': [60, 32, 52]
        }
        df = pd.DataFrame(data)

    # 特征处理
    feature_cols = ['C_Avg', 'Cr_Avg', 'Mn_Avg', 'Mo_Avg', 'Ni_Avg', 'V_Avg']
    target_col = 'HRC_Avg'
    
    for col in feature_cols + [target_col]:
        if col not in df.columns:
            df[col] = 0
            
    X = df[feature_cols].fillna(0)
    y = df[target_col].fillna(0)

    # 训练模型
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X, y)
    
    return df, model

df, model = load_data_and_model()

# --- 侧边栏 ---
st.sidebar.header("⚙️ 成分模拟实验室")
st.sidebar.info("调整下方化学成分(%)，右侧将实时预测硬度。")

def user_input_features():
    c = st.sidebar.slider('C (碳)', 0.0, 3.5, 0.4, 0.01)
    cr = st.sidebar.slider('Cr (铬)', 0.0, 20.0, 5.0, 0.1)
    mn = st.sidebar.slider('Mn (锰)', 0.0, 5.0, 0.5, 0.1)
    mo = st.sidebar.slider('Mo (钼)', 0.0, 5.0, 0.2, 0.1)
    ni = st.sidebar.slider('Ni (镍)', 0.0, 10.0, 0.0, 0.1)
    v = st.sidebar.slider('V (钒)', 0.0, 5.0, 0.1, 0.1)
    return pd.DataFrame({'C_Avg': c, 'Cr_Avg': cr, 'Mn_Avg': mn, 
                         'Mo_Avg': mo, 'Ni_Avg': ni, 'V_Avg': v}, index=[0])

input_df = user_input_features()

# --- 主页面 ---
st.title("🔩 材料工程技术垂类模型")
st.markdown(f"当前数据库包含 **{len(df)}** 种材料数据")

# 模块 1: 预测
st.subheader("1. 性能预测引擎 (XGBoost)")
col1, col2 = st.columns([1, 2])

with col1:
    prediction = model.predict(input_df)[0]
    st.metric(label="预测硬度 (HRC)", value=f"{prediction:.1f}")
    if prediction > 55: st.error("高硬度 (冷作/刀具)")
    elif prediction > 40: st.warning("中硬度 (热作/结构)")
    else: st.success("低硬度 (预硬/韧性)")

with col2:
    impt = pd.DataFrame({'Element': ['C', 'Cr', 'Mn', 'Mo', 'Ni', 'V'], 'Importance': model.feature_importances_})
    st.plotly_chart(px.bar(impt, x='Element', y='Importance', title="元素影响权重"), use_container_width=True)

# 模块 2: 检索
st.divider()
st.subheader("2. 智能选材助手")
query = st.text_input("🔍 输入关键词（如：'耐腐蚀'、'Cr12'）：")

if query:
    mask = df['材料说明'].astype(str).str.contains(query, case=False, na=False) | \
           df['适用标准'].astype(str).str.contains(query, case=False, na=False) | \
           df['对比项目'].astype(str).str.contains(query, case=False, na=False)
    results = df[mask]
    st.dataframe(results[['对比项目', 'HRC_Avg', '材料说明', '适用标准']], hide_index=True)
else:
    st.dataframe(df.head(5))
