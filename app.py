import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from clustering import KPrototypes, mixed_distance_matrix, prepare_data

st.set_page_config(page_title="Telco Customer Segmentation", page_icon="📊", layout="wide")

NAVY = "#123B66"
BLUE = "#2878B5"
GOLD = "#F2B544"
st.markdown("""
<style>
    .stApp {background: #f7f9fc;}
    h1, h2, h3 {color: #123B66;}
    div[data-testid="stMetric"] {background: white; border: 1px solid #e6ebf2; padding: 14px; border-radius: 12px;}
    .small-note {color: #65758b; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data(uploaded_bytes=None):
    if uploaded_bytes is None:
        return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    import io
    return pd.read_csv(io.BytesIO(uploaded_bytes))

@st.cache_data(show_spinner="Menyiapkan data...")
def prep_cached(df):
    return prepare_data(df)

@st.cache_resource(show_spinner=False)
def fit_cached(num, cat, k, gamma, seed, n_init):
    model = KPrototypes(n_clusters=k, gamma=gamma, n_init=n_init, max_iter=100, random_state=seed)
    labels = model.fit_predict(num, cat)
    return model, labels

def fmt_pct(x):
    return f"{x:.1%}"

st.sidebar.title("📊 Telco Segmentation")
st.sidebar.caption("Customer analytics · K-Prototypes")
uploaded = st.sidebar.file_uploader("Upload dataset CSV (opsional)", type=["csv"])
try:
    raw = load_data(uploaded.getvalue() if uploaded else None)
except Exception as e:
    st.error(f"Dataset tidak dapat dibaca: {e}")
    st.stop()

required = {"customerID", "Churn", "tenure", "MonthlyCharges", "TotalCharges"}
if not required.issubset(raw.columns):
    st.error("Kolom wajib tidak ditemukan. Gunakan dataset Telco Customer Churn dari tugas.")
    st.stop()

st.sidebar.divider()
st.sidebar.subheader("Pengaturan model")
k = st.sidebar.slider("Jumlah klaster (k)", 2, 8, 4)
gamma = st.sidebar.slider("Bobot kategorik (γ)", 0.1, 2.0, 0.5, 0.1)
n_init = st.sidebar.select_slider("Jumlah inisialisasi", options=[1, 3, 5, 10], value=5)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
run = st.sidebar.button("Jalankan / perbarui clustering", type="primary", use_container_width=True)

clean, numeric_cols, categorical_cols, num_scaled, cat_values, prep_info = prep_cached(raw)
if run:
    fit_cached.clear()
model, labels = fit_cached(num_scaled, cat_values, k, gamma, int(seed), int(n_init))
data = clean.copy()
data["Cluster"] = labels.astype(int) + 1

st.title("Telco Customer Segmentation")
st.markdown("**Segmentasi pelanggan telekomunikasi menggunakan algoritma K-Prototypes**")
st.markdown('<p class="small-note">Unsupervised learning · Data numerik + kategorik · Dashboard interaktif</p>', unsafe_allow_html=True)

with st.expander("Tentang dataset & preprocessing", expanded=False):
    a, b, c = st.columns(3)
    a.metric("Jumlah observasi", f"{len(data):,}")
    b.metric("Jumlah variabel awal", f"{raw.shape[1]}")
    c.metric("Variabel pembentuk klaster", f"{len(numeric_cols)+len(categorical_cols)}")
    st.write("ID pelanggan dan variabel target Churn tidak digunakan dalam pembentukan klaster. Churn hanya dipakai setelah clustering untuk profiling/evaluasi.")
    st.write(f"Fitur numerik: {', '.join(numeric_cols)}.")
    st.write(f"Fitur kategorik: {', '.join(categorical_cols)}.")
    st.write("TotalCharges dikonversi ke numerik; 11 nilai kosong/spasi diimputasi dengan median. Fitur numerik distandardisasi menggunakan Z-score.")

sizes = data["Cluster"].value_counts().sort_index()
churn_rate = data.groupby("Cluster")["Churn"].apply(lambda s: (s == "Yes").mean())
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total pelanggan", f"{len(data):,}")
c2.metric("Jumlah klaster", str(k))
c3.metric("Biaya akhir model", f"{model.cost_:,.2f}")
c4.metric("Rata-rata churn", fmt_pct((data["Churn"]=="Yes").mean()))

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Ringkasan", "Profil Klaster", "Evaluasi Model", "Eksplorasi Pelanggan", "Metodologi"])

with tab1:
    st.subheader("Distribusi segmen pelanggan")
    left, right = st.columns([1,1])
    size_df = sizes.rename_axis("Cluster").reset_index(name="Jumlah pelanggan")
    size_df["Persentase"] = size_df["Jumlah pelanggan"] / len(data)
    with left:
        fig = px.bar(size_df, x="Cluster", y="Jumlah pelanggan", color="Cluster", text="Jumlah pelanggan",
                     color_discrete_sequence=px.colors.qualitative.Safe, title="Ukuran setiap klaster")
        fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.pie(size_df, names="Cluster", values="Jumlah pelanggan", hole=.55,
                     title="Proporsi pelanggan per klaster", color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Peta karakteristik numerik")
    numeric_profile = data.groupby("Cluster")[numeric_cols].mean().reset_index()
    long = numeric_profile.melt(id_vars="Cluster", var_name="Variabel", value_name="Rata-rata")
    fig = px.line(long, x="Variabel", y="Rata-rata", color="Cluster", markers=True,
                  title="Rata-rata variabel numerik per klaster")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Churn rate per segmen (variabel evaluasi)")
    cr = churn_rate.rename("Churn rate").reset_index()
    cr["Churn rate"] *= 100
    fig = px.bar(cr, x="Cluster", y="Churn rate", color="Cluster", text=cr["Churn rate"].map(lambda x:f"{x:.1f}%"),
                 title="Persentase pelanggan churn pada setiap klaster",
                 color_discrete_sequence=px.colors.qualitative.Safe)
    fig.update_layout(yaxis_title="Churn rate (%)", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Karakteristik setiap klaster")
    st.caption("Numerik ditampilkan sebagai rata-rata; kategorik ditampilkan sebagai kategori paling dominan (modus).")
    num_prof = data.groupby("Cluster")[numeric_cols].mean().round(2)
    cat_prof = data.groupby("Cluster")[categorical_cols].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A")
    count_prof = data.groupby("Cluster").size().rename("Jumlah")
    summary = pd.concat([count_prof, num_prof, cat_prof], axis=1)
    st.dataframe(summary, use_container_width=True)
    selected = st.selectbox("Pilih klaster untuk melihat rincian", sorted(data["Cluster"].unique()))
    seg = data[data["Cluster"] == selected]
    st.markdown(f"#### Klaster {selected} · {len(seg):,} pelanggan ({len(seg)/len(data):.1%})")
    x,y,z = st.columns(3)
    x.metric("Rata-rata tenure", f"{seg['tenure'].mean():.1f} bulan")
    y.metric("Rata-rata biaya bulanan", f"${seg['MonthlyCharges'].mean():.2f}")
    z.metric("Churn rate", fmt_pct((seg["Churn"]=="Yes").mean()))
    for col in categorical_cols:
        freq = seg[col].value_counts(normalize=True).head(5).mul(100).rename_axis(col).reset_index(name="Persentase")
        with st.expander(f"Distribusi {col}"):
            fig = px.bar(freq, x=col, y="Persentase", text=freq["Persentase"].map(lambda v:f"{v:.1f}%"))
            st.plotly_chart(fig, use_container_width=True)
    st.info("Interpretasi nama segmen sebaiknya didasarkan pada pola dominan di tabel ini, bukan hanya nomor klaster.")

with tab3:
    st.subheader("Evaluasi dan pemilihan jumlah klaster")
    st.write("Silhouette dihitung menggunakan jarak campuran pada sampel data agar komputasi tetap efisien. Skor lebih tinggi menunjukkan pemisahan klaster yang relatif lebih baik.")
    if st.button("Hitung evaluasi k = 2 sampai 6"):
        rows=[]
        rng=np.random.default_rng(int(seed))
        idx=rng.choice(len(num_scaled), size=min(900,len(num_scaled)), replace=False)
        dist=mixed_distance_matrix(num_scaled[idx], cat_values[idx], gamma)
        for kk in range(2,7):
            mm=KPrototypes(n_clusters=kk, gamma=gamma, n_init=3, max_iter=100, random_state=int(seed))
            ll=mm.fit_predict(num_scaled, cat_values)
            sil=silhouette_score(dist, ll[idx], metric="precomputed")
            rows.append({"k":kk,"Cost":mm.cost_,"Silhouette":sil})
        ev=pd.DataFrame(rows)
        st.session_state["evaluation"]=ev
    if "evaluation" in st.session_state:
        ev=st.session_state["evaluation"]
        e1,e2=st.columns(2)
        with e1:
            st.plotly_chart(px.line(ev,x="k",y="Cost",markers=True,title="Elbow method — total cost"),use_container_width=True)
        with e2:
            st.plotly_chart(px.line(ev,x="k",y="Silhouette",markers=True,title="Silhouette score (mixed distance)"),use_container_width=True)
        st.dataframe(ev.round(4),use_container_width=True)
        st.caption("Pilih k dengan mempertimbangkan elbow, silhouette, ukuran segmen, dan kemudahan interpretasi; tidak otomatis memilih skor tertinggi.")
    else:
        st.info("Klik tombol evaluasi untuk membandingkan k=2–6. Penghitungan dapat memerlukan waktu beberapa saat.")

with tab4:
    st.subheader("Visualisasi dan pencarian pelanggan")
    st.caption("PCA menggunakan fitur hasil one-hot encoding untuk visualisasi dua dimensi; proyeksi ini bukan bagian dari algoritma K-Prototypes.")
    cat_df=pd.DataFrame(cat_values,columns=categorical_cols,index=data.index)
    encoded=pd.concat([pd.DataFrame(num_scaled,columns=numeric_cols,index=data.index),pd.get_dummies(cat_df,dtype=float)],axis=1)
    pca=PCA(n_components=2,random_state=int(seed))
    coords=pca.fit_transform(encoded)
    vis=pd.DataFrame({"PC1":coords[:,0],"PC2":coords[:,1],"Cluster":data["Cluster"].astype(str)})
    fig=px.scatter(vis.sample(min(2500,len(vis)),random_state=int(seed)),x="PC1",y="PC2",color="Cluster",
                   opacity=.65,title="PCA projection of mixed customer features",
                   color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(fig,use_container_width=True)
    f1,f2=st.columns(2)
    with f1:
        cl=st.multiselect("Filter klaster",sorted(data["Cluster"].unique()),default=sorted(data["Cluster"].unique()))
    with f2:
        ch=st.selectbox("Filter churn",["Semua","Yes","No"])
    view=data[data["Cluster"].isin(cl)]
    if ch!="Semua": view=view[view["Churn"]==ch]
    search=st.text_input("Cari customer ID")
    if search: view=view[view["customerID"].str.contains(search,case=False,na=False)]
    st.write(f"Menampilkan {len(view):,} pelanggan")
    st.dataframe(view.head(1000),use_container_width=True)
    st.download_button("Download hasil segmentasi (CSV)",data=data.to_csv(index=False).encode("utf-8"),
                       file_name="telco_customer_segments.csv",mime="text/csv")

with tab5:
    st.subheader("Metodologi K-Prototypes")
    st.markdown(r"""
**K-Prototypes** menggabungkan komponen K-Means untuk fitur numerik dan K-Modes untuk fitur kategorik. Prototipe klaster berisi rata-rata fitur numerik dan modus fitur kategorik.

Jarak objek ke prototipe:
\[
d(x_i,q_l)=\sum_{r=1}^{p}(x_{ir}-q_{lr})^2+
\gamma\sum_{s=1}^{m}\delta(x_{is},q_{ls})
\]
dengan \(\delta=0\) jika kategori sama dan 1 jika berbeda. Bobot \(\gamma\) mengatur kontribusi fitur kategorik.

**Tahapan:** inisialisasi prototipe → penugasan ke prototipe terdekat → pembaruan rata-rata dan modus → iterasi hingga keanggotaan stabil.

**Evaluasi:** total cost (semakin kecil semakin baik untuk model dengan k yang sama), elbow, silhouette score berbasis jarak campuran, ukuran klaster, serta interpretasi profil.
""")
    st.warning("Churn merupakan variabel eksternal untuk profiling. Jangan memasukkannya ke fitur clustering karena dapat menyebabkan target leakage.")
    st.markdown("**Referensi materi:** PPT *Analisis Klaster untuk Data Metrik, Non-Metrik, dan Campuran*, khususnya halaman 35–41 (K-Prototypes), serta halaman 45 (pemilihan algoritma dan evaluasi).")

st.divider()
st.caption("Telco Customer Segmentation · Statistical Learning Project")
