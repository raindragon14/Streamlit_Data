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
        
def build_ai_prompt(region, year, quarter, risk_score, local_factors, global_factors, instance_shap_factors):
    """
    Membangun prompt yang kaya konteks untuk LLM.

    Args:
        region (str): Nama wilayah yang dianalisis.
        year (int): Tahun analisis.
        quarter (int): Kuartal analisis.
        risk_score (float): Skor risiko yang diprediksi.
        local_factors (dict): Faktor dari LIME atau fallback SHAP instance. Berisi 'positive' dan 'negative'.
        global_factors (dict): Faktor dari SHAP summary. Berisi 'positive_factors' dan 'negative_factors'.
        instance_shap_factors (dict): Faktor SHAP spesifik untuk instance ini. Berisi 'positive' dan 'negative'.
    """
    
    # Helper function untuk menghindari repetisi kode
    def format_factors(factors_dict, is_lime=False):
        """Mengubah dictionary faktor menjadi string markdown yang rapi."""
        # LIME memberikan tuple ('feature condition', weight), sedangkan SHAP hanya (feature, weight)
        if is_lime:
            return "\n".join([f"- `{factor[0]}` (Kontribusi: {factor[1]:.3f})" for factor in factors_dict])
        return "\n".join([f"- `{factor}` (Kontribusi: {weight:.3f})" for factor, weight in factors_dict.items()])

    # Memformat semua data XAI untuk dimasukkan ke dalam prompt
    local_pos_str = format_factors(local_factors['positive'], is_lime=True)
    local_neg_str = format_factors(local_factors['negative'], is_lime=True)
    
    global_pos_str = "\n".join([f"- `{factor}`" for factor, _ in global_factors['positive_factors'][:5]])
    global_neg_str = "\n".join([f"- `{factor}`" for factor, _ in global_factors['negative_factors'][:5]])
    
    instance_shap_pos_str = format_factors(instance_shap_factors['positive'])
    instance_shap_neg_str = format_factors(instance_shap_factors['negative'])

    # Template prompt utama
    prompt = f"""
**Peran dan Tujuan:**
Anda adalah seorang Ahli Strategi Ekonomi dan Kebijakan Publik. Peran Anda adalah mengubah data kuantitatif dari model AI menjadi sebuah cetak biru kebijakan yang actionable, memiliki prioritas logis, dan visi jangka panjang untuk pemberdayaan Usaha Kecil.

---

**Konteks Analisis:**
- **Wilayah:** {region}
- **Periode:** Kuartal {quarter}, Tahun {year}
- **Skor Risiko Kegagalan Usaha Kecil:** {risk_score:.3f} (Skor Z-score; >0 berarti risiko lebih tinggi dari rata-rata).

---

**Data Penjelasan Model (Explainable AI):**

**1. Analisis Lokal (LIME - Faktor Paling Berpengaruh untuk Wilayah Ini):**
   - **Peningkat Risiko:**
{local_pos_str if local_pos_str else "     - Tidak ada yang signifikan"}
   - **Penurun Risiko:**
{local_neg_str if local_neg_str else "     - Tidak ada yang signifikan"}

**2. Analisis Lokal (SHAP - Kontribusi Pasti per Fitur untuk Wilayah Ini):**
   - **Peningkat Risiko:**
{instance_shap_pos_str if instance_shap_pos_str else "     - Tidak ada yang signifikan"}
   - **Penurun Risiko:**
{instance_shap_neg_str if instance_shap_neg_str else "     - Tidak ada yang signifikan"}

**3. Analisis Global (SHAP - Pola Umum di Seluruh Wilayah):**
   - **Peningkat Risiko Teratas (Secara Umum):**
{global_pos_str}
   - **Penurun Risiko Teratas (Secara Umum):**
{global_neg_str}

---

**Instruksi Kunci untuk Analisis:**
1.  **Cari Anomali Strategis:** Bandingkan faktor lokal (dari LIME & SHAP Instance) dengan faktor global. Jika ada perbedaan mencolok (misal, faktor yang tidak penting secara global justru menjadi pendorong utama di wilayah ini), jelaskan mengapa anomali ini penting dan apa implikasinya bagi perumusan kebijakan.
2.  **Lakukan Analisis Akar Masalah (Root Cause):** Untuk 1-2 faktor peningkat risiko utama, berikan hipotesis mengapa faktor tersebut menjadi masalah di wilayah ini. Gunakan data lain (seperti IPM, UMK) untuk mendukung hipotesis Anda.
3.  **Manfaatkan Aset Lokal:** Identifikasi faktor penurun risiko sebagai 'aset strategis' wilayah. Jelaskan bagaimana aset ini dapat dimanfaatkan (di-leverage) dalam rekomendasi kebijakan Anda.

---

**Struktur Laporan Wajib (Gunakan Format Markdown):**
1.  **Ringkasan Eksekutif:** Diagnosis inti (2-3 kalimat) dan Visi Solusi (1 kalimat).
2.  **Diagnosis Mendalam:** Jelaskan anomali strategis yang Anda temukan dan signifikansinya.
3.  **Analisis Akar Masalah:** Terapkan Instruksi Kunci #2 dan #3.
4.  **Cetak Biru Kebijakan:**
    - **A. Visi & Urutan Prioritas:** Jelaskan logika strategi dan prioritas (Jangka Pendek, Menengah, Panjang).
    - **B. Program Aksi Terukur:** Sajikan 3 program prioritas dengan format: Nama Program, Aksi Konkret, Target Sasaran, KPI, dan Kerangka Waktu.
"""
    return prompt

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
    
    # Panggil fungsi untuk menghasilkan penjelasan LIME
lime_exp = generate_lime_explanation(model, X, instance_idx)

# Inisialisasi variabel untuk menampung hasil
positive_features = []
negative_features = []
local_factors_for_prompt = {}

# Cek apakah LIME berhasil
if lime_exp:
    # JALUR UTAMA: LIME berhasil, proses hasilnya
    exp_list = lime_exp.as_list()
    positive_features = sorted([(f, w) for f, w in exp_list if w > 0], key=lambda i: i[1], reverse=True)
    negative_features = sorted([(f, w) for f, w in exp_list if w < 0], key=lambda i: i[1])
    
    # Siapkan data untuk prompt, LIME memberikan tuple ('feature condition', weight)
    local_factors_for_prompt = {
        'positive': positive_features, 
        'negative': negative_features
    }
else:
    # JALUR FALLBACK: LIME gagal, gunakan SHAP per instance sebagai gantinya
    st.warning("Penjelasan LIME gagal, analisis faktor lokal akan menggunakan SHAP per instance sebagai gantinya.", icon="⚠️")
    
    instance_shap_values = shap_values.values[instance_idx]
    feature_names = shap_values.feature_names
    

    shap_dict = dict(zip(feature_names, instance_shap_values))
  
    positive_features_from_shap = sorted([(f, w) for f, w in shap_dict.items() if w > 0], key=lambda i: i[1], reverse=True)
    negative_features_from_shap = sorted([(f, w) for f, w in shap_dict.items() if w < 0], key=lambda i: i[1])
    positive_features = [(f, w) for f, w in positive_features_from_shap]
    negative_features = [(f, w) for f, w in negative_features_from_shap]
    local_factors_for_prompt = {
        'positive': [(f'{feat}', val) for feat, val in positive_features_from_shap],
        'negative': [(f'{feat}', val) for feat, val in negative_features_from_shap]
    }
    
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
