import streamlit as st
import pandas as pd
import pickle
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# -------------------------
# Load Model & Data
# -------------------------
@st.cache_resource
def load_model():
    with open("risk_prediction_pipeline.pkl", "rb") as file:
        model = pickle.load(file)
    return model

@st.cache_data
def load_training_data():
    return pd.read_csv("Data_Model_for_Training.csv")

model = load_model()
training_data = load_training_data()

# Ambil fitur (anggap kolom target bernama "target" atau mirip)
feature_columns = training_data.drop(columns=[col for col in training_data.columns if "target" in col.lower()]).columns.tolist()

st.set_page_config(page_title="Risk Prediction App", layout="wide")
st.title("📊 Risk Prediction Dashboard dengan Interpretabilitas")
st.markdown("Upload dataset baru dan lihat hasil prediksi beserta interpretasi model menggunakan **SHAP** dan **LIME**.")

# -------------------------
# Upload Data
# -------------------------
uploaded_file = st.file_uploader("📂 Upload Data (CSV)", type=["csv"])

if uploaded_file:
    input_data = pd.read_csv(uploaded_file)
    st.subheader("📋 Data yang di-upload")
    st.dataframe(input_data.head())

    missing_cols = [col for col in feature_columns if col not in input_data.columns]
    if missing_cols:
        st.error(f"❌ Kolom berikut hilang: {missing_cols}")
    else:
        X = input_data[feature_columns]
        predictions = model.predict(X)
        input_data["Prediction"] = predictions

        st.subheader("🔮 Hasil Prediksi")
        st.dataframe(input_data.head())

        # -------------------------
        # SHAP Explanation
        # -------------------------
        st.subheader("📌 SHAP Explanation")

        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        # Summary Plot
        st.markdown("**Global Feature Importance (SHAP Summary Plot)**")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X, show=False)
        st.pyplot(fig)

        # Force Plot untuk 1 baris
        st.markdown("**Local Explanation (Force Plot)**")
        row_idx = st.slider("Pilih baris data untuk dilihat SHAP Force Plot", 0, len(X)-1, 0)
        shap_html = shap.plots.force(explainer.expected_value, shap_values[row_idx].values, X.iloc[row_idx,:], matplotlib=False)
        st.components.v1.html(shap_html.html(), height=300)

        # -------------------------
        # LIME Explanation
        # -------------------------
        st.subheader("📌 LIME Explanation")

        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data[feature_columns].values,
            feature_names=feature_columns,
            class_names=["Class 0", "Class 1"],
            mode="classification"
        )

        row_idx_lime = st.slider("Pilih baris data untuk LIME", 0, len(X)-1, 0, key="lime_row")
        exp = lime_explainer.explain_instance(
            X.iloc[row_idx_lime].values,
            model.predict_proba,
            num_features=10
        )

        st.markdown(f"**LIME Explanation untuk Baris {row_idx_lime}:**")
        st.write(exp.as_list())

        fig_lime = exp.as_pyplot_figure()
        st.pyplot(fig_lime)

else:
    st.info("Silakan upload file CSV untuk mulai melakukan prediksi.")
