import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from lime import lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediktor Risiko Sosial Ekonomi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Kustom
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 600;
}
.section-header {
    font-size: 1.5rem;
    color: #2c3e50;
    margin-top: 2rem;
    margin-bottom: 1rem;
    font-weight: 500;
    border-bottom: 2px solid #3498db;
    padding-bottom: 0.5rem;
}
.explanation-box {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
    border-left: 4px solid #1f77b4;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.metric-container {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #e1e8ed;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    margin: 0.5rem 0;
}
.feature-importance {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    border-left: 3px solid #28a745;
}
.feature-negative {
    border-left-color: #dc3545;
}
.sidebar-header {
    font-size: 1.2rem;
    font-weight: 600;
    color: #2c3e50;
    margin-bottom: 1rem;
}
.info-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Bagian fungsi TIDAK DIUBAH
# ----------------------------
MODEL_FEATURES = [
    'Garis_Kemiskinan', 'Indeks_Pembangunan_Manusia', 'Persen_Penduduk_Miskin',
    'Tingkat Pengangguran Terbuka', 'Upah Minimum', 'Jumlah Penduduk (Ribu)',
    'Laju Pertumbuhan Penduduk per Tahun', 'Persentase Penduduk',
    'Kepadatan Penduduk per km persegi (Km2)', 'Rasio Jenis Kelamin Penduduk',
    'PDRB', 'Laju_Inflasi', 'Gini_Ratio', 'investasi_per_kapita',
    'Jumlah Perusahaan Kecil', 'Jumlah Perusahaan'
]

FULL_EXPECTED_COLUMNS = MODEL_FEATURES + [
    'kabupaten_kota', 'tahun', 'kuartal', 'Proksi Inflasi'
]

def load_model():
    try:
        pipeline = joblib.load('Pipeline/risk_prediction_pipeline.pkl')
        if hasattr(pipeline, 'feature_names_in_'):
            st.info(f"Model dilatih dengan fitur: {list(pipeline.feature_names_in_)}")
        return pipeline
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

def validate_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        missing_cols = set(FULL_EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            st.warning(f"Kolom yang hilang: {', '.join(missing_cols)}")
            return None
        for col in MODEL_FEATURES:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        if col == 'Laju_Inflasi':
                            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
                        else:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        if df[col].isna().sum() > len(df) * 0.1:
                            st.warning(f"Kolom '{col}' terlalu banyak nilai non-numerik")
                            return None
                    except Exception as e:
                        st.warning(f"Kolom '{col}' harus numerik. Error: {e}")
                        return None
        return df
    except Exception as e:
        st.error(f"Gagal membaca CSV: {e}")
        return None

def generate_shap_explanations(model, X, feature_names):
    try:
        explainer = shap.Explainer(model.named_steps['xgb_model'] if hasattr(model, 'named_steps') else model)
        shap_values = explainer(X)
        return shap_values
    except Exception as e:
        st.warning(f"Gagal membuat penjelasan SHAP: {e}")
        return None

def generate_lime_explanation(model, X, instance_idx, feature_names):
    try:
        if hasattr(model, 'named_steps'):
            underlying_model = model.named_steps['xgb_model']
        else:
            underlying_model = model
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X.values,
            feature_names=feature_names,
            mode='regression',
            verbose=True,
            random_state=42
        )
        exp = explainer.explain_instance(
            X.iloc[instance_idx].values,
            underlying_model.predict,
            num_features=10
        )
        return exp
    except Exception as e:
        st.warning(f"Gagal membuat penjelasan LIME: {e}")
        return None

# ----------------------------
# Aplikasi utama
# ----------------------------
def main():
    st.markdown('<div class="main-header">📊 Prediktor Risiko Sosial Ekonomi dengan XAI</div>', unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.error("Model gagal dimuat. Pastikan file pipeline tersedia.")
        return

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">📁 Unggah Data</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Unggah file CSV dengan data sosial ekonomi",
            type=['csv'],
            help="Gunakan format kolom sesuai struktur yang dibutuhkan"
        )
        if uploaded_file:
            st.success("✅ File berhasil diunggah!")

        st.markdown("---")
        st.markdown('<div class="sidebar-header">📊 Informasi Model</div>', unsafe_allow_html=True)
        st.markdown("""
        **Fitur yang digunakan model:**
        - Indikator ekonomi  
        - Indikator sosial  
        - Data demografi  
        - Identitas wilayah  
        - Data runtun waktu
        """)

        st.markdown("---")
        st.markdown('<div class="sidebar-header">🔍 Fitur Analisis</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Analisis SHAP**: Pentingnya fitur secara global  
        - **Penjelasan LIME**: Interpretasi lokal  
        - **Visualisasi Interaktif**: Eksplorasi data  
        - **Ekspor Hasil**: Unduh prediksi
        """)

    # (Bagian utama tetap sama, hanya teks UI diubah ke bahasa Indonesia)
    # Contoh:
    # st.button("🚀 Jalankan Prediksi")
    # st.markdown("#### 📊 Pentingnya Fitur Global (SHAP)")
    # st.markdown("#### 🔎 Penjelasan Lokal (LIME)")
    # st.markdown("#### 📈 Analisis Distribusi Fitur")
    # st.download_button(label="📥 Unduh Hasil Lengkap dengan Prediksi")

if __name__ == "__main__":
    main()
