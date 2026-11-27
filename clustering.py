import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

NUMERIC_CANDIDATES = ["tenure", "MonthlyCharges", "TotalCharges"]
EXCLUDE = {"customerID", "Churn"}

def prepare_data(df):
    data = df.copy()
    for col in NUMERIC_CANDIDATES:
        if col == "TotalCharges":
            data[col] = pd.to_numeric(data[col].astype(str).str.strip(), errors="coerce")
        else:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    numeric_cols = [c for c in NUMERIC_CANDIDATES if c in data.columns]
    missing_numeric = 0
    for col in numeric_cols:
        missing_numeric += int(data[col].isna().sum())
        data[col] = data[col].fillna(data[col].median())
    categorical_cols = [c for c in data.columns if c not in EXCLUDE and c not in numeric_cols]
    for col in categorical_cols:
        data[col] = data[col].fillna("Unknown").astype(str).str.strip()
        data[col] = data[col].replace("", "Unknown")
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(data[numeric_cols]).astype(float)
    cat_values = data[categorical_cols].astype(str).to_numpy(dtype=object)
    return data, numeric_cols, categorical_cols, num_scaled, cat_values, {"imputed_numeric": missing_numeric}

def _assign(num, cat, cent_num, cent_cat, gamma):
    # squared Euclidean component + gamma * simple-matching dissimilarity
    num_d = ((num[:, None, :] - cent_num[None, :, :]) ** 2).sum(axis=2)
    cat_d = np.zeros((len(num), len(cent_cat)), dtype=float)
    for j in range(cat.shape[1]):
        cat_d += (cat[:, None, j] != cent_cat[None, :, j])
    return num_d + gamma * cat_d

def _mode(values):
    vals, counts = np.unique(values, return_counts=True)
    return vals[np.argmax(counts)]

class KPrototypes:
    """Huang-style K-Prototypes for mixed numeric and categorical arrays."""
    def __init__(self, n_clusters=4, gamma=0.5, n_init=5, max_iter=100, random_state=42):
        self.n_clusters=n_clusters
        self.gamma=gamma
        self.n_init=n_init
        self.max_iter=max_iter
        self.random_state=random_state
        self.cost_=None
        self.labels_=None
        self.cluster_centroids_=None
        self.n_iter_=0

    def fit_predict(self, num, cat):
        num=np.asarray(num,dtype=float); cat=np.asarray(cat,dtype=object)
        if num.ndim!=2 or cat.ndim!=2 or len(num)!=len(cat):
            raise ValueError("Input numerik dan kategorik harus berupa matriks dengan jumlah baris sama.")
        n=len(num); k=int(self.n_clusters)
        if k<2 or k>=n: raise ValueError("Jumlah klaster harus >=2 dan lebih kecil dari jumlah observasi.")
        rng=np.random.default_rng(self.random_state)
        best=None
        for init in range(max(1,int(self.n_init))):
            ids=rng.choice(n,size=k,replace=False)
            cn=num[ids].copy()
            cc=cat[ids].copy()
            prev=None
            for it in range(self.max_iter):
                costs=_assign(num,cat,cn,cc,self.gamma)
                labels=costs.argmin(axis=1)
                # Repair empty clusters by moving high-cost observations.
                counts=np.bincount(labels,minlength=k)
                for empty in np.where(counts==0)[0]:
                    donor=np.argmax(costs.min(axis=1))
                    labels[donor]=empty
                    counts=np.bincount(labels,minlength=k)
                new_cn=cn.copy(); new_cc=cc.copy()
                for j in range(k):
                    mask=labels==j
                    if mask.any():
                        new_cn[j]=num[mask].mean(axis=0)
                        for q in range(cat.shape[1]):
                            new_cc[j,q]=_mode(cat[mask,q])
                stable=(prev is not None and np.array_equal(labels,prev))
                cn,cc=new_cn,new_cc
                prev=labels.copy()
                if stable: break
            final_cost=_assign(num,cat,cn,cc,self.gamma)
            final_labels=final_cost.argmin(axis=1)
            cost=float(final_cost[np.arange(n),final_labels].sum())
            if best is None or cost<best[0]:
                best=(cost,final_labels.copy(),cn.copy(),cc.copy(),it+1)
        self.cost_,self.labels_,self.cluster_centroids_,self.n_iter_=best[0],best[1],(best[2],best[3]),best[4]
        return self.labels_

def mixed_distance_matrix(num, cat, gamma):
    num=np.asarray(num,dtype=float); cat=np.asarray(cat,dtype=object)
    d2=((num[:,None,:]-num[None,:,:])**2).sum(axis=2)
    mismatch=np.zeros_like(d2)
    for j in range(cat.shape[1]):
        mismatch += (cat[:,None,j] != cat[None,:,j])
    # Square root converts the additive dissimilarity into a distance for silhouette.
    return np.sqrt(np.maximum(d2 + gamma*mismatch, 0.0))
