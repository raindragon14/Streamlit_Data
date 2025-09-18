import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from lime import lime_tabular
import matplotlib.pyplot as plt
import openai
import warnings

warnings.filterwarnings('ignore')

# ======================================================================================
# KONFIGURASI DAN GAYA (STYLING)
# ======================================================================================

st.set_page_config(
    page_title="Small Business Failure Risk Predictor",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan yang lebih menarik
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; font-weight: 600;
    }
    .section-header {
        font-size: 1.5rem; color: #2c3e50; margin-top: 2rem; margin-bottom: 1rem;
        font-weight: 500; border-bottom: 2px solid #3498db; padding-bottom: 0.5rem;
    }
    .sidebar-header {
        font-size: 1.2rem; color: #2c3e50; font-weight: 600; margin-bottom: 1rem; text-align: center;
    }
    .risk-increasing {
        background-color: #fdf2f2; border: 1px solid #fecaca; border-left: 4px solid #dc3545;
        padding: 12px; margin: 8px 0; border-radius: 6px; color: #721c24;
    }
    .risk-decreasing {
        background-color: #f0f9f4; border: 1px solid #bbf7d0; border-left: 4px solid #28a745;
        padding: 12px; margin: 8px 0; border-radius: 6px; color: #14532d;
    }
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;
        padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# ======================================================================================
# VARIABEL GLOBAL & FUNGSI UTILITAS
# ======================================================================================

MODEL_FEATURES = [
    'Garis_Kemiskinan', 'Indeks_Pembangunan_Manusia', 'Persen_Penduduk_Miskin',
    'Tingkat Pengangguran Terbuka', 'Upah Minimum', 'Jumlah Penduduk (Ribu)',
    'Laju Pertumbuhan Penduduk per Tahun', 'Persentase Penduduk',
    'Kepadatan Penduduk per km persegi (Km2)', 'Rasio Jenis Kelamin Penduduk',
    'PDRB', 'Laju_Inflasi', 'Gini_Ratio', 'investasi_per_kapita'
]
FULL_EXPECTED_COLUMNS = MODEL_FEATURES + ['kabupaten_kota', 'tahun', 'kuartal', 'Proksi Inflasi']


# ======================================================================================
# FUNGSI-FUNGSI DENGAN CACHING UNTUK PERFORMA
# ======================================================================================

@st.cache_resource
def load_model():
    """Memuat pipeline model dari file dan menyimpannya di cache."""
    try:
        pipeline = joblib.load('Pipeline/risk_prediction_pipeline.pkl')
        return pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def validate_and_process_csv(uploaded_file):
    """Memvalidasi dan memproses file CSV, hasilnya di-cache."""
    try:
        df = pd.read_csv(uploaded_file)
        missing_cols = set(FULL_EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            st.warning(f"Kolom yang hilang: {', '.join(missing_cols)}")
            return None
        
        for col in MODEL_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

        # Menghapus baris dengan nilai NaN pada fitur-fitur penting
        original_rows = len(df)
        df.dropna(subset=MODEL_FEATURES, inplace=True)
        if len(df) < original_rows:
            st.info(f"Ditemukan dan dihapus {original_rows - len(df)} baris dengan data yang tidak lengkap.")

        if df.empty:
            st.error("Tidak ada data yang valid tersisa setelah pembersihan. Mohon periksa file Anda.")
            return None

        return df
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

@st.cache_data
def run_predictions_and_shap(_model, df):
    """Menjalankan prediksi dan kalkulasi SHAP, hasilnya di-cache."""
    X = df[MODEL_FEATURES].copy()
    predictions = _model.predict(X)
    df_with_predictions = df.copy()
    df_with_predictions['Risk_Score'] = predictions

    # Ekstraksi model XGBoost dari pipeline
    underlying_model = _model.named_steps.get('xgb_model', _model)
    
    try:
        explainer = shap.Explainer(underlying_model)
        shap_values = explainer(X)
    except Exception as e:
        st.warning(f"Gagal menghasilkan penjelasan SHAP: {e}")
        shap_values = None

    return df_with_predictions, X, shap_values

def generate_lime_explanation(model, X, instance_idx):
    """Generate LIME explanation. Dibuat lebih robust."""
    try:
        underlying_model = model.named_steps.get('xgb_model', model)
        X_clean = X.copy()
        for col in X_clean.columns:
            if X_clean[col].std() < 1e-9:
                X_clean[col] += np.random.normal(0, 1e-6, size=len(X_clean))
        
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_clean.values,
            feature_names=MODEL_FEATURES,
            mode='regression',
            verbose=False,
            random_state=42,
            discretize_continuous=True,
            sample_around_instance=True
        )
        exp = explainer.explain_instance(
            X.iloc[instance_idx].values,
            underlying_model.predict,
            num_features=min(10, len(MODEL_FEATURES))
        )
        return exp
    except Exception as e:
        st.warning(f"Gagal menghasilkan penjelasan LIME: {e}")
        return None

# ======================================================================================
# FUNGSI-FUNGSI INTEGRASI AI (DEEPSEEK)
# ======================================================================================

def get_ai_client():
    """Menginisialisasi dan mengembalikan klien OpenAI untuk DeepSeek."""
    try:
        client = openai.OpenAI(
            api_key=st.secrets["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1"
        )
        return client, True
    except Exception:
        return None, False
        
def build_ai_prompt(region, year, quarter, risk_score, positive_features, negative_features, shap_analysis, lime_failed=False):
    """Membangun prompt yang dinamis dan terstruktur untuk LLM."""
    
    pos_factors_str = "\n".join([f"- `{factor}` (Kontribusi: {weight:.3f})" for factor, weight in positive_features])
    neg_factors_str = "\n".join([f"- `{factor}` (Kontribusi: {weight:.3f})" for factor, weight in negative_features])

    shap_section = ""
    if shap_analysis:
        shap_pos_str = "\n".join([f"- `{factor}` (Rata-rata SHAP: {weight:.4f})" for factor, weight in shap_analysis['positive_factors'][:5]])
        shap_neg_str = "\n".join([f"- `{factor}` (Rata-rata SHAP: {weight:.4f})" for factor, weight in shap_analysis['negative_factors'][:5]])
        shap_section = f"""
**Analisis Global (SHAP) - Pola Umum di Seluruh Wilayah:**
* Faktor Global Peningkat Risiko: {shap_pos_str}
* Faktor Global Penurun Risiko: {shap_neg_str}
"""
    
    lime_intro = (
        "**Peringatan:** Analisis LIME gagal. Faktor-faktor berikut adalah perkiraan berdasarkan deviasi dari nilai median, bukan kontribusi model langsung." if lime_failed 
        else "**Faktor-faktor Kunci dari Model (berdasarkan LIME - Analisis Lokal untuk Wilayah Ini):**"
    )
    
    return f"""
    Anda adalah seorang analis ekonomi dan konsultan Usaha Kecil ahli yang ditugaskan untuk menjelaskan prediksi risiko kegagalan usaha kecil untuk seorang kepala daerah, pembuat kebijakan, atau lembaga pemberdayaan Usaha Kecil.
    
    Catatan : 
    1. "Persentase Penduduk" dihitung dengan (jumlah penduduk di wilayah itu)/jumlah penduduk Jawa Timur lalu dikalikan dengan 100%
    2. "investasi_per_kapita" dihitung dengan jumlah investasi pada wilayah tersebut dibagi dengan jumlah tenaga kerja
    
    **Konteks Analisis:**
    - **Wilayah:** {region}
    - **Periode:** Kuartal {quarter}, Tahun {year}
    - **Skor Risiko Kegagalan Usaha Kecil:** {risk_score:.3f} (skor lebih tinggi berarti risiko kegagalan usaha kecil lebih besar, skor berupa hasil normalisasi z-score)

    **Faktor-faktor Kunci dari Model (berdasarkan LIME - Analisis Lokal untuk Wilayah Ini):**
    **Faktor Peningkat Risiko Kegagalan Usaha Kecil:**
    {pos_factors_str}
    **Faktor Penurun Risiko Kegagalan Usaha Kecil:**
    {neg_factors_str}
    {shap_section}

    **Tugas Anda:**
    Tulis laporan analisis yang ringkas, jelas, objektif dan actionable dalam format Markdown. Laporan harus mencakup:
    
    1.  **Ringkasan Eksekutif:** Mulai dengan kesimpulan utama dalam 2-3 kalimat. Jelaskan secara singkat tingkat risiko kegagalan Usaha Kecil di wilayah ini dan apa pendorong utamanya.
    
    2.  **Analisis Faktor Lokal vs Global:** Bandingkan hasil LIME (spesifik untuk wilayah ini) dengan pola SHAP global. Jelaskan apakah wilayah ini mengikuti pola umum kegagalan Usaha Kecil atau memiliki karakteristik unik.
    
    3.  **Analisis Faktor Peningkat Risiko:** Jelaskan MENGAPA faktor-faktor ini kemungkinan besar meningkatkan risiko kegagalan Usaha Kecil. Hubungkan dengan teori ekonomi Usaha Kecil, akses permodalan, daya beli masyarakat, atau kondisi pasar lokal.
    
    4.  **Analisis Faktor Penurun Risiko:** Jelaskan MENGAPA faktor-faktor ini menjadi kekuatan pendukung keberhasilan Usaha Kecil di wilayah tersebut dan membantu menekan risiko kegagalan.
    
    5.  **Rekomendasi Kebijakan Usaha Kecil:** Berdasarkan analisis lokal dan global, berikan 3-4 rekomendasi kebijakan yang konkret dan dapat ditindaklanjuti untuk pemberdayaan Usaha Kecil, dengan prioritas berdasarkan dampak yang diharapkan terhadap pengurangan risiko kegagalan usaha kecil.

    Gunakan bahasa yang profesional namun mudah dipahami oleh audiens non-teknis. Fokus pada konteks Usaha Kecil, seperti akses permodalan, daya beli masyarakat, infrastruktur bisnis, dan ekosistem kewirausahaan. Jika ada data SHAP yang tersedia, pastikan untuk mengintegrasikan insight global dengan analisis lokal LIME.
    """

def generate_narrative_explanation(client, prompt):
    """Memanggil API DeepSeek dan mengembalikan respons."""
    if not client:
        return "Layanan AI tidak dikonfigurasi."
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Anda adalah seorang analis ekonomi ahli yang fokus pada Usaha Kecil."},
                {"role": "user", "content": prompt}
            ],
            temperature=st.session_state.get('llm_temperature', 0.8),
            max_tokens=st.session_state.get('llm_max_tokens', 3000)
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Terjadi kesalahan saat menghasilkan analisis AI dari DeepSeek: {str(e)}"

# ======================================================================================
# FUNGSI-FUNGSI UNTUK MENAMPILKAN KOMPONEN UI
# ======================================================================================

def display_sidebar(df_is_loaded):
    """Menampilkan semua elemen di sidebar."""
    with st.sidebar:
        st.markdown('<div class="sidebar-header">📁 Data Upload</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload file CSV data sosioekonomi", type=['csv'])
        
        if df_is_loaded:
            if st.button("🔄 Analisis Data Baru"):
                # Menghapus cache dan state untuk analisis baru
                st.cache_data.clear()
                for key in list(st.session_state.keys()):
                    if key != 'llm_temperature' and key != 'llm_max_tokens':
                        del st.session_state[key]
                st.rerun()

        st.markdown("---")
        st.markdown("### 📋 Format Data")
        with st.expander("Lihat Kolom Wajib"):
            st.json({
                "Fitur Model": MODEL_FEATURES,
                "Metadata": ['kabupaten_kota', 'tahun', 'kuartal', 'Proksi Inflasi']
            })
        return uploaded_file

def display_welcome_page():
    """Menampilkan halaman sambutan jika belum ada file yang diunggah."""
    st.markdown("""
        <div class="info-card">
            <h3>👋 Selamat Datang di Prediktor Risiko Kegagalan Usaha Kecil!</h3>
            <p>Unggah data sosioekonomi Anda di sidebar kiri untuk memulai analisis prediktif dan mendapatkan wawasan mendalam dengan Explainable AI (XAI) dan Generative AI.</p>
        </div>
    """, unsafe_allow_html=True)

def display_prediction_overview(df_pred):
    """Menampilkan ringkasan hasil prediksi."""
    st.markdown('<div class="section-header">📈 Hasil Prediksi Risiko</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Rata-rata Skor Risiko", f"{df_pred['Risk_Score'].mean():.3f}")
    col2.metric("Skor Risiko Minimum", f"{df_pred['Risk_Score'].min():.3f}")
    col3.metric("Skor Risiko Maksimum", f"{df_pred['Risk_Score'].max():.3f}")

    with st.expander("📊 Tabel Prediksi Detail (Dapat Diurutkan)", expanded=False):
        st.dataframe(
            df_pred[['kabupaten_kota', 'tahun', 'kuartal', 'Risk_Score']]
            .sort_values('Risk_Score', ascending=False)
            .rename(columns={'Risk_Score': 'Risiko_Kegagalan'})
        )

def display_shap_analysis(X, shap_values):
    """Menampilkan analisis SHAP (global)."""
    st.markdown('<div class="section-header">🔍 Analisis Global (SHAP)</div>', unsafe_allow_html=True)
    st.info("SHAP (SHapley Additive exPlanations) menunjukkan kontribusi rata-rata setiap fitur terhadap prediksi di seluruh dataset. Ini membantu mengidentifikasi faktor-faktor paling berpengaruh secara umum.", icon="💡")
    
    if shap_values is None:
        st.warning("Analisis SHAP tidak tersedia.")
        return

    # Global Feature Importance
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("Pentingnya Fitur Global untuk Risiko Kegagalan Usaha Kecil")
    st.pyplot(fig)
    
    st.markdown("---")

    # SHAP Summary Plot
    st.markdown("#### Dampak Fitur Individual")
    st.write("Plot ini menunjukkan bagaimana nilai sebuah fitur (warna) mempengaruhi prediksi (posisi pada sumbu x). Merah berarti nilai fitur tinggi, biru berarti rendah.")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig)

def display_lime_and_ai_analysis(model, X, df_pred, client):
    """Menampilkan analisis LIME (lokal) dan integrasi AI."""
    st.markdown('<div class="section-header">🎯 Analisis Lokal (LIME) & Wawasan AI</div>', unsafe_allow_html=True)
    st.info("LIME (Local Interpretable Model-agnostic Explanations) menjelaskan prediksi untuk **satu data point spesifik**. Pilih wilayah dan periode di bawah untuk melihat faktor apa yang mendorong risikonya.", icon="💡")

    col1, col2, col3 = st.columns(3)
    unique_regions = sorted(df_pred['kabupaten_kota'].unique())
    unique_years = sorted(df_pred['tahun'].unique())
    unique_quarters = sorted(df_pred['kuartal'].unique())

    selected_region = col1.selectbox("Pilih Kabupaten/Kota", unique_regions)
    selected_year = col2.selectbox("Pilih Tahun", unique_years)
    selected_quarter = col3.selectbox("Pilih Kuartal", unique_quarters)

    # Cari instance yang sesuai
    instance_df = df_pred[
        (df_pred['kabupaten_kota'] == selected_region) & 
        (df_pred['tahun'] == selected_year) & 
        (df_pred['kuartal'] == selected_quarter)
    ]

    if instance_df.empty:
        st.warning("Tidak ada data untuk kombinasi yang dipilih.")
        return

    instance_idx = instance_df.index[0]
    predicted_risk = instance_df.iloc[0]['Risk_Score']
    
    st.subheader(f"Analisis untuk: {selected_region} (Q{selected_quarter} {selected_year})")
    st.metric("Prediksi Skor Risiko", f"{predicted_risk:.3f}")
    
    lime_exp = generate_lime_explanation(model, X, instance_idx)
    
    if lime_exp:
        exp_list = lime_exp.as_list()
        positive_features = sorted([(f, w) for f, w in exp_list if w > 0], key=lambda i: i[1], reverse=True)
        negative_features = sorted([(f, w) for f, w in exp_list if w < 0], key=lambda i: i[1])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔴 Faktor Peningkat Risiko (Lokal)**")
            for feature, weight in positive_features:
                st.markdown(f'<div class="risk-increasing"><strong>{feature}</strong><br/>Dampak: <strong>+{weight:.3f}</strong></div>', unsafe_allow_html=True)
        with col2:
            st.markdown("**🟢 Faktor Penurun Risiko (Lokal)**")
            for feature, weight in negative_features:
                st.markdown(f'<div class="risk-decreasing"><strong>{feature}</strong><br/>Dampak: <strong>{weight:.3f}</strong></div>', unsafe_allow_html=True)
        
        # Integrasi AI
        if client:
            if st.button("🤖 Buat Analisis Naratif dengan AI", key=f"ai_{instance_idx}"):
                with st.spinner("🧠 AI sedang menyusun laporan..."):
                    shap_analysis = st.session_state.get('shap_analysis_summary')
                    prompt = build_ai_prompt(selected_region, selected_year, selected_quarter, predicted_risk,
                                             positive_features, negative_features, shap_analysis)
                    narrative = generate_narrative_explanation(client, prompt)
                    st.session_state[f'narrative_{instance_idx}'] = narrative
    else:
        # Fallback jika LIME gagal
        st.warning("LIME gagal, analisis AI akan menggunakan perbandingan data mentah dengan median.")
        instance_data = X.loc[instance_idx]
        median_values = X.median()
        diffs = (instance_data - median_values) / median_values
        
        positive_features = sorted([(f, v) for f, v in diffs.items() if v > 0], key=lambda i: i[1], reverse=True)
        negative_features = sorted([(f, v) for f, v in diffs.items() if v < 0], key=lambda i: i[1])

        if client:
             if st.button("🤖 Buat Analisis Naratif dengan AI (Fallback)", key=f"ai_fallback_{instance_idx}"):
                with st.spinner("🧠 AI sedang menyusun laporan..."):
                    shap_analysis = st.session_state.get('shap_analysis_summary')
                    prompt = build_ai_prompt(selected_region, selected_year, selected_quarter, predicted_risk,
                                             positive_features, negative_features, shap_analysis, st.session_state.analysis_language, lime_failed=True)
                    narrative = generate_narrative_explanation(client, prompt)
                    st.session_state[f'narrative_{instance_idx}'] = narrative

    if f'narrative_{instance_idx}' in st.session_state:
        st.markdown("---")
        st.markdown("### 💡 Laporan Analisis AI")
        st.markdown(st.session_state[f'narrative_{instance_idx}'])

# ======================================================================================
# FUNGSI UTAMA (MAIN)
# ======================================================================================

def main():
    st.markdown('<div class="main-header">🏪 Prediktor Risiko Kegagalan Usaha Kecil dengan XAI</div>', unsafe_allow_html=True)
    
    model = load_model()
    client, llm_configured = get_ai_client()
    
    if not llm_configured:
        st.warning("⚠️ Kunci API DeepSeek tidak terkonfigurasi. Fitur Analisis AI tidak akan tersedia. Atur `DEEPSEEK_API_KEY` di `st.secrets`.", icon="🤖")

    if model is None:
        st.error("Gagal memuat model. Aplikasi tidak dapat berjalan.")
        return

    # Menggunakan session state untuk menyimpan df agar tidak hilang saat interaksi
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    
    uploaded_file = display_sidebar(st.session_state.processed_df is not None)

    if uploaded_file is not None:
        st.session_state.processed_df = validate_and_process_csv(uploaded_file)
    
    if st.session_state.processed_df is not None:
        df = st.session_state.processed_df
        df_with_predictions, X, shap_values = run_predictions_and_shap(model, df)
        
        # Menyimpan ringkasan SHAP untuk prompt AI
        if shap_values is not None and 'shap_analysis_summary' not in st.session_state:
             avg_shap = np.mean(shap_values.values, axis=0)
             shap_analysis_summary = {
                'positive_factors': sorted([(MODEL_FEATURES[i], avg_shap[i]) for i in np.where(avg_shap > 0)[0]], key=lambda x: x[1], reverse=True),
                'negative_factors': sorted([(MODEL_FEATURES[i], avg_shap[i]) for i in np.where(avg_shap < 0)[0]], key=lambda x: x[1])
             }
             st.session_state.shap_analysis_summary = shap_analysis_summary

        # Tampilan menggunakan Tabs untuk UX yang lebih baik
        tab1, tab2, tab3 = st.tabs(["📊 Ringkasan Prediksi", "🌍 Analisis Global (SHAP)", "🎯 Analisis Lokal & AI"])

        with tab1:
            display_prediction_overview(df_with_predictions)
        with tab2:
            display_shap_analysis(X, shap_values)
        with tab3:
            display_lime_and_ai_analysis(model, X, df_with_predictions, client)
    else:
        display_welcome_page()

if __name__ == "__main__":
    main()
