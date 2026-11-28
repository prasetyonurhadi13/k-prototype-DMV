"""
Dashboard Segmentasi Pelanggan Telco dengan K-Prototypes
Mata kuliah Data Mining - Analisis Klaster Data Campuran
Prasetyo Nurhadi (5003241015)

Alasan memilih K-Prototypes (materi: "Memilih Algoritma: Mulai dari Tipe Data"):
data Telco berisi fitur numerik (tenure, MonthlyCharges, TotalCharges) DAN fitur
kategorik (Contract, InternetService, PaymentMethod, dst). Untuk data campuran,
algoritma yang tepat adalah K-Prototypes (Huang, 1997/1998):
    d(x, q) = sum (x_r - q_r)^2  +  gamma * sum delta(x_s, q_s)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from kmodes.kprototypes import KPrototypes
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --------------------------------------------------------------------------
# Konfigurasi halaman & gaya
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Segmentasi Pelanggan Telco | K-Prototypes",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = ["#2563EB", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6",
           "#0EA5E9", "#EC4899", "#84CC16"]
DEFAULT_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1250px;}
      h1, h2, h3 {letter-spacing: -0.01em;}
      .hero {background: linear-gradient(120deg, #0B3D91 0%, #2563EB 60%, #38BDF8 100%);
             color: #fff; padding: 1.6rem 1.8rem; border-radius: 16px; margin-bottom: 1.2rem;}
      .hero h1 {color: #fff; margin: 0 0 .25rem 0; font-size: 1.9rem;}
      .hero p {color: #E0ECFF; margin: 0; font-size: 1rem;}
      .kpi {background: var(--secondary-background-color); border-radius: 14px;
            padding: .9rem 1.1rem; border: 1px solid rgba(128,128,128,.18);}
      .kpi .lbl {font-size: .78rem; opacity: .7; text-transform: uppercase; letter-spacing: .04em;}
      .kpi .val {font-size: 1.6rem; font-weight: 700; line-height: 1.25;}
      .kpi .sub {font-size: .8rem; opacity: .65;}
      .seg-card {border-radius: 14px; padding: 1rem 1.1rem; margin-bottom: .6rem;
                 border: 1px solid rgba(128,128,128,.2); background: var(--secondary-background-color);}
      .seg-card h4 {margin: 0 0 .15rem 0;}
      .pill {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem;
             font-weight: 600; color: #fff; margin-right: 6px;}
      .note {border-left: 4px solid #F59E0B; background: rgba(245,158,11,.10);
             padding: .7rem 1rem; border-radius: 6px; font-size: .92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="kpi"><div class="lbl">{label}</div>'
            f'<div class="val">{value}</div><div class="sub">{sub}</div></div>')


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
NUM_ALL = ["tenure", "MonthlyCharges", "TotalCharges"]
CAT_ALL = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
           "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
           "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
           "Contract", "PaperlessBilling", "PaymentMethod"]


@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes | None) -> pd.DataFrame:
    if file_bytes is not None:
        from io import BytesIO
        df = pd.read_csv(BytesIO(file_bytes))
    else:
        path = Path(__file__).parent / DEFAULT_FILE
        df = pd.read_csv(path)
    return df


@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pembersihan: TotalCharges kosong (pelanggan baru, tenure 0) -> 0; SeniorCitizen -> Yes/No."""
    d = df.copy()
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce")
    d["TotalCharges"] = d["TotalCharges"].fillna(0.0)
    if d["SeniorCitizen"].dtype != object and str(d["SeniorCitizen"].dtype) != "str":
        d["SeniorCitizen"] = d["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return d


# --------------------------------------------------------------------------
# Model K-Prototypes
# --------------------------------------------------------------------------
def build_matrix(d: pd.DataFrame, num_cols, cat_cols):
    """Numerik di-z-score (std populasi, ddof=0 seperti di materi) lalu digabung dengan kategorik."""
    num = d[num_cols].astype(float)
    mu, sd = num.mean(), num.std(ddof=0).replace(0, 1.0)
    z = (num - mu) / sd
    X = np.concatenate([z.to_numpy(dtype=object), d[cat_cols].astype(str).to_numpy(dtype=object)], axis=1)
    cat_idx = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
    return X, z, cat_idx


@st.cache_data(show_spinner=False)
def fit_kproto(d: pd.DataFrame, num_cols: tuple, cat_cols: tuple, k: int,
               gamma: float, init: str, n_init: int, seed: int):
    X, z, cat_idx = build_matrix(d, list(num_cols), list(cat_cols))
    kp = KPrototypes(n_clusters=k, init=init, n_init=n_init, gamma=gamma,
                     random_state=seed, n_jobs=1, max_iter=100)
    labels = kp.fit_predict(X, categorical=cat_idx)
    # urutkan label berdasarkan ukuran klaster (Segmen 1 = terbesar)
    order = pd.Series(labels).value_counts().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[v] for v in labels])
    return labels, float(kp.cost_), int(kp.n_iter_)


@st.cache_data(show_spinner=False)
def elbow_table(d: pd.DataFrame, num_cols: tuple, cat_cols: tuple, k_max: int,
                gamma: float, init: str, n_init: int, seed: int, sil_n: int):
    """Biaya (elbow) dan silhouette (jarak Huang, pada sampel) untuk k = 2..k_max."""
    X, z, cat_idx = build_matrix(d, list(num_cols), list(cat_cols))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(d), size=min(sil_n, len(d)), replace=False)
    zs = z.to_numpy()[idx]
    cs = d[list(cat_cols)].astype(str).to_numpy()[idx]
    D = cdist(zs, zs, "sqeuclidean")
    mism = np.zeros_like(D)
    for j in range(cs.shape[1]):
        mism += (cs[:, j][:, None] != cs[:, j][None, :])
    D = D + gamma * mism
    rows = []
    for k in range(2, k_max + 1):
        kp = KPrototypes(n_clusters=k, init=init, n_init=n_init, gamma=gamma,
                         random_state=seed, n_jobs=1, max_iter=100)
        lab = kp.fit_predict(X, categorical=cat_idx)
        sil = silhouette_score(D, lab[idx], metric="precomputed") if len(set(lab[idx])) > 1 else np.nan
        rows.append({"k": k, "cost": float(kp.cost_), "silhouette": float(sil)})
    return pd.DataFrame(rows)


def segment_label(row_num: pd.Series, mode_row: pd.Series, overall: pd.Series) -> str:
    """Nama segmen otomatis dari karakteristik dominan (supaya mudah ditafsirkan)."""
    parts = []
    inet = mode_row.get("InternetService")
    if inet == "Fiber optic":
        parts.append("Fiber")
    elif inet == "DSL":
        parts.append("DSL")
    elif inet == "No":
        parts.append("Tanpa Internet")
    if "MonthlyCharges" in row_num:
        r = row_num["MonthlyCharges"] / overall["MonthlyCharges"]
        parts.append("Tagihan Tinggi" if r > 1.12 else "Tagihan Rendah" if r < 0.8 else "Tagihan Sedang")
    if "tenure" in row_num:
        r = row_num["tenure"] / overall["tenure"]
        parts.append("Pelanggan Lama" if r > 1.25 else "Pelanggan Baru" if r < 0.7 else "Masa Langganan Menengah")
    return " · ".join(parts) if parts else "Segmen"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    up = st.file_uploader("Dataset (opsional, CSV Telco)", type="csv",
                          help="Kosongkan untuk memakai dataset Telco bawaan di repositori.")

    try:
        raw = load_data(up.getvalue() if up is not None else None)
    except FileNotFoundError:
        st.error(f"File `{DEFAULT_FILE}` tidak ditemukan. Upload CSV lewat tombol di atas.")
        st.stop()
    data = clean_data(raw)

    st.markdown("**Fitur untuk klastering**")
    num_cols = st.multiselect("Numerik (di-z-score)", NUM_ALL, default=NUM_ALL)
    cat_cols = st.multiselect("Kategorik (modus)", CAT_ALL, default=CAT_ALL)

    st.markdown("**Parameter K-Prototypes**")
    k = st.slider("Jumlah klaster (k)", 2, 8, 3)
    gamma_mode = st.radio("Bobot γ", ["Bawaan kmodes (0,5 × std)", "Atur manual"],
                          help="Bawaan: γ = 0,5 × rata-rata simpangan baku fitur numerik. "
                               "Karena numerik sudah di-z-score, nilainya 0,5.")
    gamma = 0.5 if gamma_mode.startswith("Bawaan") else st.slider("γ", 0.0, 5.0, 0.5, 0.1)
    init = st.selectbox("Inisialisasi", ["Cao", "Huang"],
                        help="Cao deterministik (berbasis kepadatan); Huang acak berbasis frekuensi.")
    n_init = st.slider("n_init (jumlah inisialisasi)", 1, 10, 3 if init == "Huang" else 1,
                       help="Dijalankan beberapa kali; dipilih biaya terkecil. Cao deterministik, cukup 1.")
    seed = st.number_input("Random state", 0, 9999, 42, step=1)
    st.caption("Tab *Pemilihan k* memakai pengaturan yang sama.")

if len(num_cols) == 0 or len(cat_cols) == 0:
    st.warning("Pilih minimal satu fitur numerik dan satu fitur kategorik agar K-Prototypes bermakna "
               "(kalau hanya satu jenis, cukup K-Means atau K-Modes).")
    st.stop()

num_t, cat_t = tuple(num_cols), tuple(cat_cols)

# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>📡 Segmentasi Pelanggan Telco</h1>
      <p>Analisis klaster data campuran dengan <b>K-Prototypes</b> · Data Mining ·
      Prasetyo Nurhadi (5003241015)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Jalankan model
# --------------------------------------------------------------------------
with st.spinner("Menjalankan K-Prototypes..."):
    labels, cost, n_iter = fit_kproto(data, num_t, cat_t, k, gamma, init, n_init, int(seed))

res = data.copy()
res["Segmen"] = labels + 1
seg_ids = sorted(res["Segmen"].unique())
color_map = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(seg_ids)}
color_str = {str(s): c for s, c in color_map.items()}
res["SegmenStr"] = res["Segmen"].astype(str)

has_churn = "Churn" in res.columns
if has_churn:
    res["ChurnBin"] = (res["Churn"].astype(str) == "Yes").astype(int)

overall_num = res[num_cols].mean()
prof_num = res.groupby("Segmen")[num_cols].mean()
prof_mode = res.groupby("Segmen")[cat_cols].agg(lambda s: s.value_counts().index[0])
seg_names = {s: f"Segmen {s}: " + segment_label(prof_num.loc[s], prof_mode.loc[s], overall_num)
             for s in seg_ids}
res["NamaSegmen"] = res["Segmen"].map(seg_names)

tabs = st.tabs(["🏠 Ringkasan", "🔎 Eksplorasi Data", "📐 Pemilihan k",
                "🧩 Hasil Klaster", "👥 Profil Segmen", "🆕 Prediksi Pelanggan", "📥 Unduh"])

# ========================== Ringkasan ==========================
with tabs[0]:
    c = st.columns(5)
    c[0].markdown(kpi("Pelanggan", f"{len(res):,}", "baris data"), unsafe_allow_html=True)
    c[1].markdown(kpi("Fitur dipakai", f"{len(num_cols)} + {len(cat_cols)}", "numerik + kategorik"),
                  unsafe_allow_html=True)
    c[2].markdown(kpi("Jumlah segmen", str(k), f"γ = {gamma:g} · init {init}"), unsafe_allow_html=True)
    c[3].markdown(kpi("Biaya akhir", f"{cost:,.0f}", f"{n_iter} iterasi"), unsafe_allow_html=True)
    if has_churn:
        c[4].markdown(kpi("Churn keseluruhan", f"{res['ChurnBin'].mean():.1%}", "pelanggan berhenti"),
                      unsafe_allow_html=True)
    else:
        c[4].markdown(kpi("Churn", "–", "kolom tidak ada"), unsafe_allow_html=True)

    st.markdown("")
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Mengapa K-Prototypes?")
        st.markdown(
            """
Dataset Telco berisi **dua jenis variabel sekaligus**:

- **Numerik**: `tenure`, `MonthlyCharges`, `TotalCharges`
- **Kategorik**: `Contract`, `InternetService`, `PaymentMethod`, layanan tambahan, dan lainnya

Menurut materi kuliah, pemilihan algoritma dimulai dari tipe data:

| Tipe data | Algoritma |
|---|---|
| Numerik saja | K-Means |
| Kategorik saja | K-Modes |
| **Campuran** | **K-Prototypes** ✅ |

K-Means tidak cocok karena rata-rata "Fiber optic" tidak terdefinisi, dan pengodean 0/1
menghasilkan pusat yang sulit ditafsirkan. K-Prototypes menyimpan **rata-rata** untuk fitur
numerik dan **modus** untuk fitur kategorik dalam satu prototipe.
            """
        )
        st.latex(r"d(\mathbf{x},\mathbf{q}_l)=\sum_{r=1}^{p}(x_{r}-q_{lr})^2+\gamma\sum_{s=p+1}^{m}\delta(x_{s},q_{ls})")
        st.caption("Jarak Huang: Euclidean² untuk numerik + γ × ketidakcocokan untuk kategorik.")
    with right:
        st.subheader("Ukuran tiap segmen")
        sz = res["Segmen"].value_counts().sort_index()
        fig = go.Figure(go.Pie(labels=[f"Segmen {s}" for s in sz.index], values=sz.values,
                               hole=.58, marker=dict(colors=[color_map[s] for s in sz.index]),
                               textinfo="label+percent", sort=False))
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Temuan utama")
    for s in seg_ids:
        n_s = int((res["Segmen"] == s).sum())
        line = (f"**{seg_names[s]}** — {n_s:,} pelanggan ({n_s/len(res):.1%}), "
                f"rata-rata tenure {prof_num.loc[s].get('tenure', float('nan')):.0f} bulan, "
                f"tagihan ${prof_num.loc[s].get('MonthlyCharges', float('nan')):.1f}/bulan")
        if has_churn:
            line += f", churn {res.loc[res['Segmen']==s, 'ChurnBin'].mean():.1%}"
        st.markdown(f"- {line}")

# ========================== Eksplorasi ==========================
with tabs[1]:
    st.subheader("Cuplikan & kualitas data")
    a, b = st.columns([2, 1])
    with a:
        st.dataframe(raw.head(15), width="stretch", height=330)
    with b:
        miss = (raw["TotalCharges"].astype(str).str.strip() == "").sum() if "TotalCharges" in raw else 0
        st.markdown(f"- Baris: **{raw.shape[0]:,}**, kolom: **{raw.shape[1]}**")
        st.markdown(f"- `TotalCharges` kosong: **{miss}** baris (pelanggan baru, tenure 0) → diisi 0")
        st.markdown("- `customerID` tidak dipakai (identitas)")
        st.markdown("- `Churn` **tidak** dipakai untuk klastering; hanya untuk memprofilkan segmen")
        st.markdown("- `SeniorCitizen` (0/1) diperlakukan sebagai **kategorik**")
        st.markdown("- Numerik distandarisasi dengan z-score (std populasi)")

    st.subheader("Distribusi fitur numerik")
    cols = st.columns(len(num_cols))
    for col, nc in zip(cols, num_cols):
        f = px.histogram(res, x=nc, nbins=40, color_discrete_sequence=[PALETTE[0]])
        f.update_layout(height=270, margin=dict(l=0, r=0, t=30, b=0), title=nc, yaxis_title=None)
        col.plotly_chart(f, width="stretch")

    st.subheader("Distribusi fitur kategorik")
    pick = st.selectbox("Pilih fitur", cat_cols, index=min(cat_cols.index("Contract") if "Contract" in cat_cols else 0, len(cat_cols) - 1))
    vc = res[pick].value_counts().reset_index()
    vc.columns = [pick, "Jumlah"]
    f = px.bar(vc, x=pick, y="Jumlah", color_discrete_sequence=[PALETTE[0]], text="Jumlah")
    f.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(f, width="stretch")

    if len(num_cols) >= 2:
        st.subheader("Korelasi antar fitur numerik")
        cm = res[num_cols].corr().round(2)
        f = px.imshow(cm, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        f.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(f, width="stretch")
        if "TotalCharges" in num_cols and "tenure" in num_cols:
            st.markdown('<div class="note"><b>Catatan:</b> <code>TotalCharges</code> berkorelasi kuat dengan '
                        '<code>tenure</code> (≈ tenure × tagihan bulanan), sehingga membobotkan dimensi '
                        '"lama berlangganan" dua kali. Coba lepas salah satunya di sidebar dan bandingkan hasilnya.</div>',
                        unsafe_allow_html=True)

# ========================== Pemilihan k ==========================
with tabs[2]:
    st.subheader("Memilih jumlah klaster k")
    st.markdown("Sesuai materi: pilih k dengan **elbow**, **silhouette**, dan pertimbangan substantif. "
                "Silhouette dihitung pada sampel acak dengan jarak Huang (numerik² + γ × ketidakcocokan).")
    cA, cB = st.columns(2)
    k_max = cA.slider("k maksimum", 4, 10, 8)
    sil_n = cB.select_slider("Ukuran sampel silhouette", [1000, 1500, 2000, 3000], value=2000)
    if st.button("▶️ Hitung elbow & silhouette", type="primary"):
        with st.spinner(f"Menjalankan K-Prototypes untuk k = 2..{k_max} (bisa 1–3 menit)..."):
            st.session_state["elbow"] = elbow_table(data, num_t, cat_t, k_max, gamma, init, n_init,
                                                    int(seed), int(sil_n))
    if "elbow" in st.session_state:
        et = st.session_state["elbow"]
        g1, g2 = st.columns(2)
        f1 = go.Figure(go.Scatter(x=et["k"], y=et["cost"], mode="lines+markers",
                                  line=dict(color=PALETTE[0], width=3)))
        f1.update_layout(title="Elbow: biaya (cost) vs k", xaxis_title="k", yaxis_title="Biaya",
                         height=340, margin=dict(l=0, r=0, t=40, b=0))
        g1.plotly_chart(f1, width="stretch")
        f2 = go.Figure(go.Scatter(x=et["k"], y=et["silhouette"], mode="lines+markers",
                                  line=dict(color=PALETTE[1], width=3)))
        f2.update_layout(title="Silhouette (jarak Huang) vs k", xaxis_title="k", yaxis_title="Silhouette",
                         height=340, margin=dict(l=0, r=0, t=40, b=0))
        g2.plotly_chart(f2, width="stretch")
        best = et.loc[et["silhouette"].idxmax()]
        st.success(f"Silhouette tertinggi pada **k = {int(best['k'])}** ({best['silhouette']:.3f}). "
                   f"Pertimbangkan juga titik siku pada grafik biaya dan kemudahan tafsir segmen. "
                   f"k yang sedang dipakai: **{k}**.")
        st.dataframe(et.round(4), width="stretch", hide_index=True)
    else:
        st.info("Klik tombol di atas untuk menghitung. Hasil dihitung sekali lalu disimpan.")

# ========================== Hasil Klaster ==========================
with tabs[3]:
    st.subheader("Sebaran pelanggan per segmen")
    n_plot = res
    if len(num_cols) >= 2:
        xa, ya = st.columns(2)
        xv = xa.selectbox("Sumbu X", num_cols, index=0)
        yv = ya.selectbox("Sumbu Y", num_cols, index=1)
        samp = n_plot.sample(min(len(n_plot), 3500), random_state=1)
        f = px.scatter(samp, x=xv, y=yv, color="SegmenStr", color_discrete_map=color_str,
                       category_orders={"SegmenStr": [str(s) for s in seg_ids]},
                       opacity=.65, labels={"SegmenStr": "Segmen"},
                       hover_data=[c for c in ["Contract", "InternetService", "PaymentMethod"] if c in res.columns])
        f.update_traces(marker=dict(size=6))
        f.update_layout(height=470, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(f, width="stretch")
        st.caption("Ditampilkan sampel maksimal 3.500 titik agar ringan.")

    st.subheader("Proyeksi 2D data campuran (PCA pada numerik z-score + kategorik one-hot)")
    X_num = (res[num_cols] - res[num_cols].mean()) / res[num_cols].std(ddof=0).replace(0, 1)
    X_cat = pd.get_dummies(res[cat_cols].astype(str)) * np.sqrt(gamma)
    P = PCA(n_components=2, random_state=0).fit_transform(np.hstack([X_num.to_numpy(), X_cat.to_numpy(dtype=float)]))
    pp = pd.DataFrame({"PC1": P[:, 0], "PC2": P[:, 1], "SegmenStr": res["SegmenStr"]})
    pp = pp.sample(min(len(pp), 3500), random_state=1)
    f = px.scatter(pp, x="PC1", y="PC2", color="SegmenStr", color_discrete_map=color_str,
                   category_orders={"SegmenStr": [str(s) for s in seg_ids]},
                   opacity=.6, labels={"SegmenStr": "Segmen"})
    f.update_traces(marker=dict(size=5))
    f.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(f, width="stretch")
    st.caption("Hanya alat bantu visual; klaster tetap dihitung dengan jarak Huang di ruang aslinya.")

    st.subheader("Sebaran fitur numerik per segmen")
    nv = st.selectbox("Fitur", num_cols, key="boxnum")
    f = px.box(res, x="SegmenStr", y=nv, color="SegmenStr", color_discrete_map=color_str,
               category_orders={"SegmenStr": [str(s) for s in seg_ids]},
               labels={"SegmenStr": "Segmen"})
    f.update_layout(height=380, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(f, width="stretch")

# ========================== Profil Segmen ==========================
with tabs[4]:
    st.subheader("Profil dan interpretasi segmen")
    st.markdown("Prototipe tiap klaster = **rata-rata** fitur numerik + **modus** fitur kategorik "
                "(Huang, 1998). Tafsirkan profilnya, bukan sekadar label.")

    for s in seg_ids:
        sub = res[res["Segmen"] == s]
        mt = prof_mode.loc[s]
        bits = []
        for cc in ["Contract", "InternetService", "PaymentMethod"]:
            if cc in mt.index:
                share = (sub[cc] == mt[cc]).mean()
                bits.append(f"{cc}: <b>{mt[cc]}</b> ({share:.0%})")
        num_bits = " · ".join(f"{nc}: <b>{prof_num.loc[s, nc]:,.1f}</b>" for nc in num_cols)
        churn_bit = f' <span class="pill" style="background:#EF4444">Churn {sub["ChurnBin"].mean():.1%}</span>' if has_churn else ""
        st.markdown(
            f'<div class="seg-card" style="border-left:6px solid {color_map[s]}">'
            f'<h4>{seg_names[s]}</h4>'
            f'<span class="pill" style="background:{color_map[s]}">{len(sub):,} pelanggan · {len(sub)/len(res):.1%}</span>'
            f'{churn_bit}<br><span style="font-size:.9rem">{num_bits}</span><br>'
            f'<span style="font-size:.9rem">{" · ".join(bits)}</span></div>',
            unsafe_allow_html=True)

    st.subheader("Tabel prototipe")
    proto = pd.concat([prof_num.round(2), prof_mode], axis=1)
    proto.index = [f"Segmen {s}" for s in proto.index]
    proto = proto.astype(str)
    st.dataframe(proto.T, width="stretch", height=560)

    if has_churn:
        st.subheader("Tingkat churn per segmen")
        ch = res.groupby("Segmen")["ChurnBin"].mean().reset_index()
        ch["Segmen"] = ch["Segmen"].map(lambda s: f"Segmen {s}")
        f = px.bar(ch, x="Segmen", y="ChurnBin", text=ch["ChurnBin"].map("{:.1%}".format),
                   color="Segmen", color_discrete_sequence=[color_map[s] for s in seg_ids])
        f.add_hline(y=res["ChurnBin"].mean(), line_dash="dash", annotation_text="rata-rata keseluruhan")
        f.update_layout(height=340, showlegend=False, yaxis_tickformat=".0%", yaxis_title="Churn",
                        margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(f, width="stretch")

    st.subheader("Komposisi fitur kategorik per segmen")
    cv = st.selectbox("Fitur kategorik", cat_cols,
                      index=cat_cols.index("Contract") if "Contract" in cat_cols else 0, key="catprof")
    ct = pd.crosstab(res["Segmen"], res[cv], normalize="index") * 100
    ct.index = [f"Segmen {s}" for s in ct.index]
    f = px.imshow(ct.round(1), text_auto=".0f", color_continuous_scale="Blues", aspect="auto",
                  labels=dict(color="%"))
    f.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(f, width="stretch")
    st.caption("Angka = persentase pelanggan dalam segmen (tiap baris berjumlah 100%).")

    st.subheader("Pembeda utama antar segmen (selisih terhadap rata-rata populasi)")
    overall_modes = {}
    diffs = []
    for cc in cat_cols:
        pop = res[cc].value_counts(normalize=True)
        for s in seg_ids:
            seg_share = res.loc[res["Segmen"] == s, cc].value_counts(normalize=True)
            for lv in pop.index:
                diffs.append({"Segmen": s, "Fitur": f"{cc} = {lv}",
                              "Selisih (poin %)": (seg_share.get(lv, 0) - pop[lv]) * 100})
    dd = pd.DataFrame(diffs)
    sel = st.selectbox("Segmen", seg_ids, format_func=lambda s: seg_names[s])
    top = dd[dd["Segmen"] == sel].reindex(dd[dd["Segmen"] == sel]["Selisih (poin %)"].abs()
                                          .sort_values(ascending=False).index).head(12)
    top = top.sort_values("Selisih (poin %)")
    f = px.bar(top, x="Selisih (poin %)", y="Fitur", orientation="h",
               color=top["Selisih (poin %)"] > 0,
               color_discrete_map={True: PALETTE[2], False: PALETTE[3]})
    f.update_layout(height=420, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(f, width="stretch")
    st.caption("Hijau = lebih sering muncul di segmen ini dibanding populasi; merah = lebih jarang.")

# ========================== Prediksi ==========================
with tabs[5]:
    st.subheader("Tentukan segmen untuk pelanggan baru")
    st.markdown("Isi profil pelanggan; aplikasi menghitung jarak Huang ke prototipe setiap segmen "
                "dan memilih yang terdekat.")
    inp = {}
    cols = st.columns(3)
    for i, nc in enumerate(num_cols):
        lo, hi = float(res[nc].min()), float(res[nc].max())
        inp[nc] = cols[i % 3].number_input(nc, lo, hi, float(round(res[nc].median(), 1)))
    for i, cc in enumerate(cat_cols):
        opts = sorted(res[cc].astype(str).unique())
        default = prof_mode[cc].mode().iloc[0] if cc in prof_mode else opts[0]
        inp[cc] = cols[(i + len(num_cols)) % 3].selectbox(cc, opts, index=opts.index(str(default)))

    if st.button("🎯 Tentukan segmen", type="primary"):
        mu, sd = res[num_cols].mean(), res[num_cols].std(ddof=0).replace(0, 1.0)
        zx = np.array([(inp[nc] - mu[nc]) / sd[nc] for nc in num_cols])
        dists = {}
        for s in seg_ids:
            zq = ((prof_num.loc[s] - mu) / sd).to_numpy()
            d_num = float(((zx - zq) ** 2).sum())
            d_cat = sum(inp[cc] != prof_mode.loc[s, cc] for cc in cat_cols)
            dists[s] = (d_num + gamma * d_cat, d_num, d_cat)
        best = min(dists, key=lambda s: dists[s][0])
        st.success(f"Pelanggan ini paling dekat dengan **{seg_names[best]}**")
        dt = pd.DataFrame([{"Segmen": f"Segmen {s}", "Jarak total": v[0], "Bagian numerik": v[1],
                            "Ketidakcocokan kategorik": v[2]} for s, v in dists.items()])
        st.dataframe(dt.round(3), width="stretch", hide_index=True)

# ========================== Unduh ==========================
with tabs[6]:
    st.subheader("Unduh hasil")
    out = res.drop(columns=[c for c in ["SegmenStr", "ChurnBin"] if c in res.columns])
    st.dataframe(out.head(50), width="stretch", height=300)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Data + label segmen (CSV)", out.to_csv(index=False).encode("utf-8"),
                       "telco_segmentasi.csv", "text/csv")
    pr = proto.copy()
    d2.download_button("⬇️ Prototipe segmen (CSV)", pr.T.to_csv().encode("utf-8"),
                       "prototipe_segmen.csv", "text/csv")
    st.caption(f"Parameter: k = {k}, γ = {gamma:g}, init = {init}, n_init = {n_init}, random_state = {int(seed)}.")
