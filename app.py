import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 入力フォームを作成
st.title('3-year Survival rate Prediction')


# モデルファイルのパス
model_path = os.path.join(os.path.dirname(__file__), 'RFS_v2_compressed.joblib')
rsf_final = joblib.load(model_path)

# 入力フォームでの特徴量の入力
age = st.slider('Age', 0, 100, 0)
sex = st.selectbox('Sex', ['', 'Male', 'Female'])
laparotomy = st.selectbox('History of Laparotomy', ['', 'Yes', 'No'])
ASA_3 = st.selectbox('ASA-PS≧3', ['', 'Yes', 'No'])
bmi = st.slider('BMI', 0.0, 50.0, 0.0, step=0.1)
cT_detail = st.selectbox('cT Category', ['', 'T1', 'T2', 'T3', 'T4'])
cN_detail = st.selectbox('cN Category', ['', 'N0', 'N1', 'N2', 'N3'])
cN3_l = st.selectbox('側方リンパ節転移', ['', '-', '+'])
distance_AV_i = st.slider('Distance from AV', 0.0, 10.0, 0.0, step=0.1)
pre_Tx = st.selectbox('Preoperative therapy', ['', 'CRT', 'CT', 'RT', 'none'])
open0lap1 = st.selectbox('Approach', ['', 'Open', 'Lap'])
procedure = st.selectbox('Procedure', ['', 'LAR', 'ISR', 'Hartmann', 'APR', 'TPE'])
LPND_lateral = st.selectbox('LPND', ['', '無', '片側', '両側'])

# 各選択が選ばれていない場合のチェック
if '' in [sex, laparotomy, ASA_3, cT_detail, cN_detail, cN3_l, pre_Tx, open0lap1, procedure, LPND_lateral]:
    st.warning('Please fill out all fields.')
else:
    # Preoperative therapyの値に応じて変数を設定
    pre_Tx_CRT = 1 if pre_Tx == 'CRT' else 0
    pre_Tx_CT = 1 if pre_Tx == 'CT' else 0
    pre_Tx_RT = 1 if pre_Tx == 'RT' else 0
    if pre_Tx == 'none':
        pre_Tx_CRT = 0
        pre_Tx_CT = 0
        pre_Tx_RT = 0

    # ユーザーの入力をデータフレームに変換し、トレーニング時の特徴量の順序に合わせる
    input_data = pd.DataFrame({
        'sex': [0 if sex == 'Male' else 1],
        'laparotomy': [1 if laparotomy == 'Yes' else 0],
        'cT_detail': {'T1': 0, 'T2': 1, 'T3': 2, 'T4': 3}[cT_detail],
        'cN_detail': {'N0': 0, 'N1': 1, 'N2': 2, 'N3': 3}[cN_detail],
        'cN3_l': {'-': 0, '+': 1}[cN3_l],
        'pre_Tx_CRT': [pre_Tx_CRT],
        'pre_Tx_CT': [pre_Tx_CT],
        'pre_Tx_RT': [pre_Tx_RT],
        'open0lap1': {'Open': 0, 'Lap': 1}[open0lap1],
        'procedure': {'LAR': 0, 'ISR': 1, 'APR': 2, 'Hartmann': 3, 'TPE': 4}[procedure],
        'LPND_lateral': {'無': 0, '片側': 1, '両側': 2}[LPND_lateral],
        'ASA_3': {'No': 0, 'Yes': 1}[ASA_3],
        'age': [age],
        'bmi': [bmi],
        'distance_AV_i': [distance_AV_i]
    }, columns=[
        'sex', 'laparotomy', 'cT_detail', 'cN_detail', 'cN3_l', 'pre_Tx_CRT', 'pre_Tx_CT', 'pre_Tx_RT', 
        'open0lap1', 'procedure', 'LPND_lateral', 'ASA_3', 'age', 'bmi', 'distance_AV_i'
    ])  # 特徴量の順序をトレーニング時に合わせる

    # カテゴリ型に変換する列のリスト
    categorical_cols = [
        'sex', 'laparotomy', 'cT_detail', 'cN_detail', 'cN3_l', 'pre_Tx_CRT', 'pre_Tx_CT', 'pre_Tx_RT', 
        'open0lap1', 'procedure', 'LPND_lateral', 'ASA_3'
    ]

    # 特定の列をカテゴリ型に変換
    input_data[categorical_cols] = input_data[categorical_cols].astype('category')

    # ボタンを追加
    if st.button('Calculate'):
        # 3年生存率の予測
        survival_functions = rsf_final.predict_survival_function(input_data, return_array=False)
        time_points = survival_functions[0].x  # 1つ目の生存関数から時間点を取得
        three_year_point_index = np.argmin(np.abs(time_points - 3))  # 3年に最も近い時間点のインデックスを求める

        # 3年目の生存確率を計算
        predicted_probabilities = [fn(time_points[three_year_point_index]) for fn in survival_functions]
        predicted_probabilities_rounded = round(predicted_probabilities[0] * 100, 1)

        # 結果を表示
        st.write(f'Predicted 3-year Survival Rate: {predicted_probabilities_rounded:.1f}%')

        # エフェクトの追加
        if predicted_probabilities_rounded >= 90:
            st.balloons()
        elif predicted_probabilities_rounded < 70:
            st.snow()
