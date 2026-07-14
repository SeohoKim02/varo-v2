"""
점포 클러스터링 (Store Clustering)
K-means 알고리즘으로 점포를 위치·재고 특성 기반으로 군집화한다.
"""
import os

# Windows loky may parse localized command output while importing sklearn.
# An explicit bounded count prevents that known environment warning without
# hiding unrelated warnings globally.
_cpu_count = os.cpu_count() or 2
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, _cpu_count - 1)))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

_K_MIN = 3
_K_MAX = 8
_RANDOM_STATE = 42
_EARTH_RADIUS_KM = 6371.0
_CLUSTER_ROLE_LABELS = {
    "HIGH_SALES":  "고회전 클러스터",
    "BALANCED":    "균형 클러스터",
    "LOW_SALES":   "저회전 클러스터",
    "STOCK_HEAVY": "재고 과잉 클러스터",
}

def _haversine(lat1, lon1, lat2, lon2):
    r = _EARTH_RADIUS_KM
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * r * np.arcsin(np.sqrt(a))

def _safe(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)

def _build_store_features(stores_df, inventory_df):
    store_col = next((c for c in ["type","store_type"] if c in stores_df.columns), None)
    if store_col:
        stores = stores_df[stores_df[store_col].str.upper() != "DC"].copy()
    else:
        stores = stores_df.copy()

    name_col = next((c for c in ["store_name","name"] if c in stores.columns), None)
    if name_col and name_col != "store_name":
        stores = stores.rename(columns={name_col: "store_name"})

    stores["latitude"]  = _safe(stores.get("latitude",  pd.Series([37.5]*len(stores))), 37.5)
    stores["longitude"] = _safe(stores.get("longitude", pd.Series([127.0]*len(stores))), 127.0)

    inv = inventory_df.copy()
    inv_name = next((c for c in ["store_name"] if c in inv.columns), None)
    pname_col = "inventory_product_name" if "inventory_product_name" in inv.columns else (
        "product_name" if "product_name" in inv.columns else None)

    if inv_name:
        agg = {"avg_daily_sales": ("avg_daily_sales","mean"),
               "stock_qty":       ("stock_qty","mean"),
               "dead_stock_qty":  ("dead_stock_qty","mean"),
               "days_to_expiry":  ("days_to_expiry","mean")}
        if pname_col:
            agg["product_count"] = (pname_col, "nunique")
        grp = inv.groupby(inv_name).agg(**agg).reset_index().rename(columns={inv_name:"store_name"})
        stores = stores.merge(grp, on="store_name", how="left")
    else:
        for col in ["avg_daily_sales","stock_qty","dead_stock_qty","days_to_expiry","product_count"]:
            stores[col] = 0.0

    for col in ["avg_daily_sales","stock_qty","dead_stock_qty","days_to_expiry","product_count"]:
        if col in stores.columns:
            stores[col] = _safe(stores[col], 0.0)
        else:
            stores[col] = 0.0

    return stores.reset_index(drop=True)

def _choose_k(X_scaled):
    n = len(X_scaled)
    unique_count = len(np.unique(X_scaled, axis=0))
    k_max = min(_K_MAX, n - 1, unique_count)
    k_min = min(_K_MIN, k_max)
    if k_max < 2:
        return 1
    sil_scores = []
    ks = list(range(k_min, k_max + 1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=_RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        if len(set(labels)) >= 2:
            sil_scores.append(silhouette_score(X_scaled, labels))
        else:
            sil_scores.append(-1)
    return ks[int(np.argmax(sil_scores))]

def _assign_cluster_role(cluster_stats):
    sales_med = cluster_stats["avg_daily_sales"].median()
    dead_med  = cluster_stats["dead_stock_qty"].median()
    role_map = {}
    for _, row in cluster_stats.iterrows():
        cid = row["cluster_id"]
        hi_s = row["avg_daily_sales"] >= sales_med * 1.1
        hi_d = row["dead_stock_qty"]  >= dead_med  * 1.2
        if hi_s and not hi_d:    role_map[cid] = "HIGH_SALES"
        elif hi_d and not hi_s:  role_map[cid] = "STOCK_HEAVY"
        elif hi_s and hi_d:      role_map[cid] = "BALANCED"
        else:                    role_map[cid] = "LOW_SALES"
    return role_map

def analyze_store_clustering(stores_df, inventory_df, n_clusters=None):
    """점포 클러스터링 수행. (store_features_df, cluster_summary_df, cluster_map) 반환."""
    if stores_df is None or stores_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    feat_df = _build_store_features(stores_df, inventory_df)
    if len(feat_df) < 2:
        feat_df["cluster_id"] = 0; feat_df["cluster_label"] = "클러스터 1"
        feat_df["cluster_role"] = "BALANCED"; feat_df["cluster_size"] = 1
        feat_df["cluster_center_distance"] = 0.0
        return feat_df, pd.DataFrame(), {}

    feature_cols = ["latitude","longitude","avg_daily_sales","stock_qty","dead_stock_qty","days_to_expiry"]
    avail = [c for c in feature_cols if c in feat_df.columns]
    X = feat_df[avail].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    unique_count = len(np.unique(X_scaled, axis=0))
    if unique_count < 2:
        feat_df["cluster_id"] = 0
        feat_df["cluster_label"] = "클러스터 1"
        feat_df["cluster_role"] = "BALANCED"
        feat_df["cluster_size"] = len(feat_df)
        feat_df["cluster_center_distance"] = 0.0
        return feat_df, pd.DataFrame(), feat_df.set_index("store_name")["cluster_id"].to_dict()

    k = n_clusters if n_clusters else _choose_k(X_scaled)
    k = max(2, min(k, len(feat_df) - 1, unique_count))

    km = KMeans(n_clusters=k, random_state=_RANDOM_STATE, n_init=10)
    feat_df["cluster_id"] = km.fit_predict(X_scaled)

    # 중심 거리
    centers_orig = scaler.inverse_transform(km.cluster_centers_)
    lat_i = avail.index("latitude")  if "latitude"  in avail else None
    lon_i = avail.index("longitude") if "longitude" in avail else None
    if lat_i is not None and lon_i is not None:
        clat = centers_orig[:, lat_i]; clon = centers_orig[:, lon_i]
        feat_df["cluster_center_distance"] = feat_df.apply(
            lambda r: round(_haversine(r["latitude"], r["longitude"],
                                       clat[int(r["cluster_id"])], clon[int(r["cluster_id"])]), 2), axis=1)
    else:
        feat_df["cluster_center_distance"] = 0.0

    feat_df["cluster_size"] = feat_df["cluster_id"].map(feat_df["cluster_id"].value_counts())

    # 요약
    summary = feat_df.groupby("cluster_id").agg(
        점포수           =("store_name",              "count"),
        평균일판매        =("avg_daily_sales",         "mean"),
        평균재고          =("stock_qty",               "mean"),
        평균악성재고      =("dead_stock_qty",           "mean"),
        평균유통기한잔여일=("days_to_expiry",           "mean"),
        중심거리_평균km  =("cluster_center_distance",  "mean"),
    ).reset_index()
    summary["avg_daily_sales"] = summary["평균일판매"]
    summary["dead_stock_qty"]  = summary["평균악성재고"]

    role_map = _assign_cluster_role(summary)
    feat_df["cluster_role"] = feat_df["cluster_id"].map(role_map)
    feat_df["cluster_label"] = feat_df.apply(
        lambda r: f"클러스터 {r['cluster_id']+1} ({_CLUSTER_ROLE_LABELS.get(r['cluster_role'],'')})", axis=1)
    summary["역할"] = summary["cluster_id"].map(
        lambda c: _CLUSTER_ROLE_LABELS.get(role_map.get(c, "BALANCED"), "균형"))
    summary["cluster_label"] = summary["cluster_id"].apply(lambda c: f"클러스터 {c+1}")
    summary = summary.drop(columns=["avg_daily_sales","dead_stock_qty"], errors="ignore")

    # 수치 반올림
    for col in ["평균일판매","평균재고","평균악성재고","평균유통기한잔여일","중심거리_평균km"]:
        if col in summary.columns:
            summary[col] = summary[col].round(1)

    cluster_map = feat_df.set_index("store_name")["cluster_id"].to_dict()
    return feat_df, summary, cluster_map


def add_cluster_to_recommendations(final_recommendations, cluster_map):
    """final_recommendations에 source_cluster / target_cluster / is_same_cluster 추가."""
    if final_recommendations is None or final_recommendations.empty or not cluster_map:
        return final_recommendations
    df = final_recommendations.copy()
    if "source_store" in df.columns:
        df["source_cluster"] = df["source_store"].map(cluster_map).fillna(-1).astype(int)
    if "target_store" in df.columns:
        df["target_cluster"] = df["target_store"].map(cluster_map).fillna(-1).astype(int)
    if "source_cluster" in df.columns and "target_cluster" in df.columns:
        df["is_same_cluster"] = (df["source_cluster"] == df["target_cluster"]) & (df["source_cluster"] >= 0)
    return df


def get_cluster_transfer_efficiency(final_recommendations):
    """클러스터 간 이동 효율성 요약."""
    if final_recommendations is None or "is_same_cluster" not in final_recommendations.columns:
        return pd.DataFrame()
    total = len(final_recommendations)
    same  = final_recommendations["is_same_cluster"].sum()
    diff  = total - same
    return pd.DataFrame({
        "구분":  ["동일 클러스터 이동", "클러스터 간 이동"],
        "건수":  [int(same), int(diff)],
        "비율":  [f"{same/total*100:.1f}%", f"{diff/total*100:.1f}%"],
        "특징":  ["비용·시간 효율 높음", "비용 추가 발생 가능"],
    })
