"""Coverage-consistent VHS candidate; production scoring remains untouched.

Each raw dimension is normalized separately before blending. A dimension is
eligible only when finite for every compared row and nonconstant. Missing
evidence is not zero or a favourable neutral score. Weights are declared fixed
reference weights, not learned weights. Candidate IDs resolve score ties.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from services.vhs_score_engine import BASE_WEIGHTS

SOURCES = {
    "savings_score": (("expected_saving", True), ("avoided_disposal_cost", True), ("recovered_margin", True)),
    "disposal_risk_score": (("days_to_expiry", False), ("expiry_days", False)),
    "demand_fit_score": (("demand_forecast_7d", True), ("sales_30d", True)),
    "inventory_balance_score": (("recommended_qty", True),),
    "route_cost_score": (("move_cost", False),),
    "promotion_score": (("promotion_effect", True),),
}


def normalized_vhs_candidate(frame: pd.DataFrame):
    if "route_id" not in frame or frame.route_id.isna().any() or frame.route_id.duplicated().any():
        raise ValueError("unique nonmissing route_id required")
    result=frame.copy()
    active, components, audit = {}, {}, []
    for component, fields in SOURCES.items():
        dimensions=[]
        for field,higher in fields:
            values=pd.to_numeric(frame[field],errors="coerce") if field in frame else pd.Series(np.nan,index=frame.index)
            valid=np.isfinite(values)
            reason=None
            if not valid.all(): reason="MISSING_OR_NONFINITE_FOR_SOME_CANDIDATES"
            elif values.nunique()<=1: reason="CONSTANT_NONDISCRIMINATING"
            if reason is None:
                score=(values-values.min())/(values.max()-values.min())
                dimensions.append(score if higher else 1-score)
            audit.append({"component":component,"field":field,"present_count":int(valid.sum()),
                          "total_count":len(frame),"used":reason is None,"reason":reason})
        if dimensions:
            components[component]=pd.concat(dimensions,axis=1).mean(axis=1)
            active[component]=BASE_WEIGHTS[component]
    if not active:
        result["candidate_vhs_score"]=np.nan
        result["candidate_vhs_rank"]=pd.Series(pd.NA,index=result.index,dtype="Int64")
        return result,{"status":"INPUT_REQUIRED","features":audit,"active_weights":{}}
    total=sum(active.values())
    active={k:v/total for k,v in active.items()}
    result["candidate_vhs_score"]=sum(components[k]*w for k,w in active.items())*100
    order=result.sort_values(["candidate_vhs_score","route_id"],ascending=[False,True],kind="stable").index
    result["candidate_vhs_rank"]=pd.Series(range(1,len(order)+1),index=order).reindex(result.index)
    return result,{"status":"CANDIDATE_PROXY_FEATURE_SCORE","features":audit,"active_weights":active,
                   "missing_policy":"exclude incomplete dimension for entire comparison population",
                   "normalization_context":"same supplied candidate set only",
                   "weights_learned":False,"actual_risk_calibrated":False,"production_approved":False}
