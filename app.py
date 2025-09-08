import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from lime import lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import openai  # --- MODIFIKASI: Menggunakan library openai ---

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Socioeconomic Risk Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (tidak ada perubahan)
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
.risk-increasing {
    background-color: #fdf2f2;
    border: 1px solid #fecaca;
    border-left: 4px solid #dc3545;
    padding: 12px;
    margin: 8px 0;
    border-radius: 6px;
    color: #721c24;
}
.risk-decreasing {
    background-color: #f0f9f4;
    border: 1px solid #bbf7d0;
    border-left: 4px solid #28a745;
    padding: 12px;
    margin: 8px 0;
    border-radius: 6px;
    color: #14532d;
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

# --- Variabel Global & Fungsi Bawaan (tidak ada perubahan) ---
MODEL_FEATURES = [
    'Garis_Kemiskinan', 'Indeks_Pembangunan_Manusia', 'Persen_Penduduk_Miskin',
    'Tingkat Pengangguran Terbuka', 'Upah Minimum', 'Jumlah Penduduk (Ribu)',
    'Laju Pertumbuhan Penduduk per Tahun', 'Persentase Penduduk',
    'Kepadatan Penduduk per km persegi (Km2)', 'Rasio Jenis Kelamin Penduduk',
    'PDRB', 'Laju_Inflasi', 'Gini_Ratio', 'investasi_per_kapita',
    'Jumlah Perusahaan Kecil', 'Jumlah Perusahaan'
]
FULL_EXPECTED_COLUMNS = MODEL_FEATURES + ['kabupaten_kota', 'tahun', 'kuartal', 'Proksi Inflasi']

def load_model():
    try:
        pipeline = joblib.load('Pipeline/risk_prediction_pipeline.pkl')
        return pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def validate_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        missing_cols = set(FULL_EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            st.warning(f"Missing columns: {', '.join(missing_cols)}")
            return None
        for col in MODEL_FEATURES:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

def generate_shap_explanations(model, X, feature_names):
    """Generate SHAP explanations for the model"""
    try:
        # Create explainer
        explainer = shap.Explainer(model.named_steps['xgb_model'] if hasattr(model, 'named_steps') else model)
        
        # Get SHAP values
        shap_values = explainer(X)
        
        return shap_values
    except Exception as e:
        st.warning(f"SHAP explanation generation failed: {e}")
        return None

def generate_lime_explanation(model, X, instance_idx, feature_names):
    """Generate LIME explanation for a specific instance"""
    try:
        # Get the underlying model from pipeline if needed
        if hasattr(model, 'named_steps'):
            underlying_model = model.named_steps['xgb_model']
        else:
            underlying_model = model
        
        # Ensure data is clean and has proper variance
        X_clean = X.copy()
        
        # Check for and handle problematic features
        for col in X_clean.columns:
            # Handle constant features (zero variance)
            if X_clean[col].std() == 0:
                # Add small noise to constant features
                X_clean[col] = X_clean[col] + np.random.normal(0, 1e-6, size=len(X_clean))
            
            # Handle features with very small variance
            elif X_clean[col].std() < 1e-10:
                X_clean[col] = X_clean[col] + np.random.normal(0, 1e-6, size=len(X_clean))
            
            # Handle infinite or very large values
            X_clean[col] = np.clip(X_clean[col], -1e10, 1e10)
            
            # Replace any remaining NaN values
            if X_clean[col].isna().any():
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
        
        # Create LIME explainer with more robust parameters
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_clean.values,
            feature_names=feature_names,
            mode='regression',
            verbose=False,  # Reduce verbosity to avoid warnings
            random_state=42,
            discretize_continuous=True,  # This can help with numerical stability
            sample_around_instance=True  # More stable sampling
        )
        
        # Explain instance with error handling
        exp = explainer.explain_instance(
            X_clean.iloc[instance_idx].values,
            underlying_model.predict,
            num_features=min(10, len(feature_names)),  # Ensure we don't exceed available features
            num_samples=1000  # Reduce samples if needed for stability
        )
        
        return exp
    except Exception as e:
        # More specific error handling
        if "scale parameter must be positive" in str(e) or "truncnorm" in str(e):
            st.warning("⚠️ LIME explanation cannot be generated due to data distribution issues. This can happen when features have very low variance or numerical stability problems.")
            st.info("💡 Try selecting a different region/time period or check if your data has sufficient variance in the features.")
        else:
            st.warning(f"LIME explanation generation failed: {str(e)}")
        return None

# --- TAMBAHAN: Fungsi untuk memanggil DeepSeek API ---
def generate_narrative_explanation(client, region, year, quarter, risk_score, positive_features, negative_features):
    """Generate a narrative explanation using DeepSeek API via openai client."""
    if not client:
        return "Layanan AI tidak dikonfigurasi."

    pos_factors_str = "\n".join([f"- `{factor}` (Kontribusi: {weight:.3f})" for factor, weight in positive_features])
    neg_factors_str = "\n".join([f"- `{factor}` (Kontribusi: {weight:.3f})" for factor, weight in negative_features])
    
    prompt_content = f"""
    Anda adalah seorang analis ekonomi ahli yang ditugaskan untuk menjelaskan prediksi risiko sosioekonomi untuk seorang kepala daerah atau pembuat kebijakan.

    **Konteks Analisis:**
    - **Wilayah:** {region}
    - **Periode:** Kuartal {quarter}, Tahun {year}
    - **Skor Risiko yang Diprediksi Model:** {risk_score:.3f} (skor lebih tinggi berarti risiko lebih besar)

    **Faktor-faktor Kunci dari Model (berdasarkan LIME):**
    **Faktor Peningkat Risiko Teratas:**
    {pos_factors_str}
    **Faktor Penurun Risiko Teratas:**
    {neg_factors_str}

    **Tugas Anda:**
    Tulis laporan analisis yang ringkas, jelas, dan actionable dalam format Markdown. Laporan harus mencakup:
    1.  **Ringkasan Eksekutif:** Mulai dengan kesimpulan utama dalam 2-3 kalimat. Jelaskan secara singkat tingkat risiko di wilayah ini dan apa pendorong utamanya.
    2.  **Analisis Faktor Peningkat Risiko:** Jelaskan MENGAPA faktor-faktor ini kemungkinan besar meningkatkan risiko. Hubungkan dengan teori ekonomi atau logika umum. Jika menemukan paradoks (misal: Upah Minimum tinggi meningkatkan risiko), berikan analisis mendalam tentang itu.
    3.  **Analisis Faktor Penurun Risiko:** Jelaskan MENGAPA faktor-faktor ini menjadi kekuatan bagi wilayah tersebut dan membantu menekan risiko.
    4.  **Rekomendasi Kebijakan:** Berdasarkan analisis, berikan 2-3 rekomendasi kebijakan yang konkret dan dapat ditindaklanjuti.

    Gunakan bahasa yang profesional namun mudah dipahami oleh audiens non-teknis.
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Anda adalah seorang analis ekonomi ahli."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Terjadi kesalahan saat menghasilkan analisis AI dari DeepSeek: {str(e)}"

# --- FUNGSI UTAMA STREAMLIT ---
def main():
    st.markdown('<div class="main-header">📊 Socioeconomic Risk Predictor with XAI</div>', unsafe_allow_html=True)

    # --- MODIFIKASI: Konfigurasi Klien API untuk DeepSeek ---
    try:
        client = openai.OpenAI(
            api_key=st.secrets["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1"
        )
        llm_configured = True
    except Exception as e:
        st.warning("⚠️ LLM DeepSeek tidak terkonfigurasi. Fitur Analisis AI tidak akan tersedia. Pastikan Anda mengatur DEEPSEEK_API_KEY di st.secrets.", icon="🤖")
        client = None
        llm_configured = False

    model = load_model()
    if model is None:
        st.error("Gagal memuat model prediksi. Pastikan file pipeline ada.")
        return

    # Inisialisasi session state (tidak ada perubahan)
    if 'shap_selected_region' not in st.session_state:
        st.session_state.shap_selected_region = "All"
    if 'shap_selected_year' not in st.session_state:
        st.session_state.shap_selected_year = "All"
    if 'shap_selected_quarter' not in st.session_state:
        st.session_state.shap_selected_quarter = "All"
    if 'selected_region' not in st.session_state:
        st.session_state.selected_region = None
    if 'selected_year' not in st.session_state:
        st.session_state.selected_year = None
    if 'selected_quarter' not in st.session_state:
        st.session_state.selected_quarter = None

    with st.sidebar:
        st.markdown('<div class="sidebar-header">📁 Data Upload</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload file CSV data sosioekonomi", type=['csv'])
        # ... (sisa sidebar tidak ada perubahan) ...
    
    if uploaded_file:
        df = validate_csv(uploaded_file)
        if df is not None:
            st.success(f"Data berhasil divalidasi! Memuat {len(df)} baris.")
            X = df[MODEL_FEATURES].copy()
            # ... (logika validasi kolom tidak ada perubahan) ...
            
            if 'predictions_made' not in st.session_state: st.session_state.predictions_made = False
            
            if st.button("🚀 Buat Prediksi", type="primary"):
                with st.spinner("Membuat prediksi dan menghasilkan penjelasan..."):
                    try:
                        predictions = model.predict(X)
                        df_with_predictions = df.copy()
                        df_with_predictions['Risk_Score'] = predictions
                        st.session_state.predictions = predictions
                        st.session_state.df_with_predictions = df_with_predictions
                        st.session_state.predictions_made = True
                    except Exception as e:
                        st.error(f"Error saat prediksi: {e}")
                        st.session_state.predictions_made = False
            
            if st.session_state.predictions_made:
                df_with_predictions = st.session_state.df_with_predictions
                predictions = st.session_state.predictions

                st.markdown('<div class="section-header">📈 Hasil Prediksi</div>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                col1.metric("Rata-rata Skor Risiko", f"{np.mean(predictions):.3f}")
                col2.metric("Skor Risiko Minimum", f"{np.min(predictions):.3f}")
                col3.metric("Skor Risiko Maksimum", f"{np.max(predictions):.3f}")

                with st.expander("📊 Tabel Prediksi Detail", expanded=False):
                    st.dataframe(df_with_predictions[['kabupaten_kota', 'tahun', 'kuartal', 'Risk_Score']].sort_values('Risk_Score', ascending=False))
                
                # --- Bagian SHAP (tidak ada perubahan) ---
                st.markdown('<div class="section-header">🔍 Wawasan Explainable AI (XAI)</div>', unsafe_allow_html=True)
                st.markdown("#### 📊 Global Feature Importance (SHAP)")
                # ... (semua kode SHAP Anda tetap di sini) ...

                # --- Bagian LIME dengan Integrasi LLM ---
                st.markdown("#### 🔎 Penjelasan Lokal (LIME)")
                st.markdown("*Memahami prediksi individual untuk wilayah dan periode waktu tertentu*")
                
                col1, col2, col3 = st.columns(3)
                unique_regions = sorted(df_with_predictions['kabupaten_kota'].unique())
                unique_years = sorted(df_with_predictions['tahun'].unique())
                unique_quarters = sorted(df_with_predictions['kuartal'].unique())
                
                # ... (logika pemilihan instance LIME Anda tetap di sini) ...
                with col1:
                    st.session_state.selected_region = st.selectbox("Pilih Kabupaten/Kota", options=unique_regions, index=0 if st.session_state.selected_region is None else unique_regions.index(st.session_state.selected_region))
                with col2:
                    st.session_state.selected_year = st.selectbox("Pilih Tahun", options=unique_years, index=len(unique_years)-1 if st.session_state.selected_year is None else unique_years.index(st.session_state.selected_year))
                with col3:
                    st.session_state.selected_quarter = st.selectbox("Pilih Kuartal", options=unique_quarters, index=0 if st.session_state.selected_quarter is None else unique_quarters.index(st.session_state.selected_quarter))

                filtered_df = df_with_predictions[
                    (df_with_predictions['kabupaten_kota'] == st.session_state.selected_region) & 
                    (df_with_predictions['tahun'] == st.session_state.selected_year) & 
                    (df_with_predictions['kuartal'] == st.session_state.selected_quarter)
                ]
                
                if not filtered_df.empty:
                    instance_idx = filtered_df.index[0]
                    predicted_risk = predictions[instance_idx]
                    lime_exp = generate_lime_explanation(model, X, instance_idx, MODEL_FEATURES)
                    
                    if lime_exp:
                        st.markdown(f"**📋 Analisis Dampak Fitur untuk {st.session_state.selected_region} - Q{st.session_state.selected_quarter} {st.session_state.selected_year}**")
                        st.info(f"Skor Risiko Prediksi: **{predicted_risk:.3f}**")
                        
                        exp_list = lime_exp.as_list()
                        positive_features = sorted([(f, w) for f, w in exp_list if w > 0], key=lambda item: item[1], reverse=True)
                        negative_features = sorted([(f, w) for f, w in exp_list if w < 0], key=lambda item: item[1])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**🔴 Faktor Peningkat Risiko:**")
                            for feature, weight in positive_features:
                                st.markdown(f'<div class="risk-increasing"><strong>{feature}</strong><br/>Dampak: <strong>+{weight:.3f}</strong></div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown("**🟢 Faktor Penurun Risiko:**")
                            for feature, weight in negative_features:
                                st.markdown(f'<div class="risk-decreasing"><strong>{feature}</strong><br/>Dampak: <strong>{weight:.3f}</strong></div>', unsafe_allow_html=True)

                        # --- MODIFIKASI: Tombol dan Tampilan Analisis AI ---
                        st.markdown("---")
                        if llm_configured:
                            btn_key = f"ai_btn_{instance_idx}"
                            if st.button("🤖 Buat Analisis Naratif dengan AI", key=btn_key):
                                with st.spinner("🧠 AI sedang menyusun laporan analisis mendalam..."):
                                    narrative = generate_narrative_explanation(
                                        client, # Menggunakan klien DeepSeek
                                        st.session_state.selected_region,
                                        st.session_state.selected_year,
                                        st.session_state.selected_quarter,
                                        predicted_risk,
                                        positive_features,
                                        negative_features
                                    )
                                    st.session_state[f'narrative_{instance_idx}'] = narrative
                            
                            # Tampilkan narasi jika ada
                            if f'narrative_{instance_idx}' in st.session_state:
                                st.markdown("### 💡 Laporan Analisis AI")
                                st.markdown(st.session_state[f'narrative_{instance_idx}'], unsafe_allow_html=True)
                        # --- Akhir Modifikasi ---
                    
                    else:
                        st.warning("Penjelasan LIME tidak dapat dibuat. Menampilkan nilai fitur mentah.")
                        # ... (kode fallback Anda tetap di sini) ...
                else:
                    st.warning(f"Tidak ada data untuk {st.session_state.selected_region} pada Q{st.session_state.selected_quarter} {st.session_state.selected_year}")

                # ... (kode visualisasi dan ekspor Anda yang lain tetap di sini) ...
                
    else:
        # --- Halaman Sambutan (tidak ada perubahan) ---
        st.markdown("""
        <div class="info-card">
            <h3>👋 Selamat Datang di Prediktor Risiko Sosioekonomi!</h3>
            <p><em>Analitik machine learning canggih untuk penilaian risiko sosioekonomi. Unggah data Anda di sidebar untuk memulai.</em></p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
