"""Third-batch research experiments: explicit information, no production writes.

CLI outputs are exclusive new directories. All models are named candidates;
latest, operational DB, raw inputs and previous evaluation files are read-only.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from services import forecast_candidate as forecast
from services.candidate_selection import Candidate, AllocationOption, SelectionProblem, evaluate, _service

SEEDS = (3, 7, 17, 29, 43)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(clean(value), handle, ensure_ascii=False, indent=2, allow_nan=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def allocation_problem(seed, size=3):
    """New numerical scenarios, not a claim of independent business data."""
    rng = random.Random(seed)
    needs = {f"T{k}": rng.choice((4, 6, 8)) for k in range(size)}
    caps = {**needs, **{f"S{k}": rng.randint(3, 9) for k in range(size)}, "VEHICLE": rng.randint(7, 16)}
    rows = []
    for s in range(size):
        for t in range(size):
            source, target = f"S{s}", f"T{t}"
            fixed, variable = rng.randint(1, 10), rng.randint(1, 4)
            rows.append(Candidate(f"{source}-{target}", "P", source, target, "CHILLED", "L", "EA",
                {source: 1, target: 1, "VEHICLE": 1}, target,
                tuple(AllocationOption(q, fixed+q*variable) for q in (1, 2, 3, 4)),
                vhs=0, cost_basis="SCENARIO_FIXED_PLUS_VARIABLE"))
    return SelectionProblem(f"THIRD-{seed}-{size}", caps, needs, tuple(rows), min(size+1, 5), "SCENARIO",
                            policy={"cost_model": "declared scenario fixed plus variable"})


def service_cost_order(problem):
    """Use the actual declared service metric, not raw quantity as a surrogate."""
    return [c.candidate_id for c in sorted(problem.candidates,
        key=lambda c: (-_service(problem, c, c.options[-1]), c.options[-1].cost, c.candidate_id))]


def inspect_unmasked_proposal(model, problem):
    """Disable only ranking mask, retain the executor's mandatory safety check.

    An invalid first proposal terminates this diagnostic, never applies it.
    This measures mask dependence, not the value of an executable unsafe policy.
    """
    import torch
    from services import candidate_dqn as dqn
    env, trace = dqn.SelectionEnvironment(problem), []
    started = time.perf_counter()
    with torch.no_grad():
        while not env.done:
            # Already chosen IDs stay excluded from the decision vocabulary.
            actions = [i for i, (c, _) in enumerate(env.actions) if c is None or c.candidate_id not in env.plan]
            values = model.network(torch.from_numpy(np.array([env.features(i) for i in actions]))).flatten().numpy()
            chosen = actions[int(np.argmax(values))]
            valid = env.valid_actions()
            trace.append({"action": chosen, "is_valid": chosen in valid, "valid_count": len(valid),
                          "candidate_count": len(actions)})
            if chosen not in valid:
                break
            env.step(chosen)
    return {"plan": env.plan, "evaluation": evaluate(problem, env.plan), "trace": trace,
            "invalid_proposals": sum(not t["is_valid"] for t in trace),
            "seconds": time.perf_counter()-started, "unsafe_action_applied": False}


def forecast_economic_replay(predicted, observed, history_mean):
    """Frozen transfer replays in the existing economic engine, SCENARIO only.

    Normalized sales are explicitly scaled by 10 into synthetic EA. Initial
    stock, prices and costs are declared artificial, not a recovered ledger.
    Observed sales are a scenario demand proxy; latent demand is unavailable.
    """
    from services.candidate_economics import PriceOption, solve_economics
    from services.candidate_economics_experiments import amount, demo_problem
    predicted, observed = float(predicted), float(observed)
    if any(not np.isfinite(x) or x < 0 for x in (predicted, observed, history_mean)):
        raise ValueError("finite nonnegative scenario inputs required")
    stock = max(1, int(np.ceil(history_mean*7*10)))
    p = demo_problem("transfer")
    source = replace(p.markets[0], options=(PriceOption("normal", amount(6), {"base": stock}),))
    target = replace(p.markets[1], options=(PriceOption("normal", amount(10), {"base": predicted*10}),))
    p = replace(p, problem_id="FORECAST_BRIDGE_SCENARIO", lots=(replace(p.lots[0], stock=stock),),
                routes=(replace(p.routes[0], vehicle_capacity=stock, fixed_per_vehicle=amount(2, None)),),
                markets=(source, target), storage_capacity={"S": stock, "T": stock})
    planned = solve_economics(p)
    if planned["status"] != "OPTIMAL":
        raise AssertionError(planned)
    quantity = sum(x["quantity"] for x in planned["transfers"])
    realized = replace(p, markets=(source, replace(target, options=(
        PriceOption("normal", amount(10), {"base": observed*10}),))))
    result = solve_economics(realized, fixed_transfers={(p.lots[0].lot_id, p.routes[0].route_id, 0): quantity})
    if result["status"] != "OPTIMAL":
        raise AssertionError(result)
    return {"planned_transfer": quantity, "initial_stock": stock,
            "planned_expected_net": planned["expected_net_value"],
            "replayed_scenario_net": result["expected_net_value"],
            "constraint_violations": result["constraint_violations"], "actual_savings": None,
            "evidence_class": "NORMALIZED_SALES_AS_DEMAND_SCENARIO", "future_used_in_initial_plan": False}


def run_forecast(root, output, data_root):
    import pyarrow.dataset as ds
    source = data_root / "05_FreshRetailNet/processed/canonical_demand_series.parquet"
    old = root / "outputs/upgrade_20261016/w4/final/forecast_report.json"
    second = root / "outputs/second_upgrade_20261016/r4_r5/forecast/exposure_log.json"
    exposed = set(json.loads(old.read_text(encoding="utf-8"))["lineage"]["series"])
    exposed.update(json.loads(second.read_text(encoding="utf-8"))["selected_series"])
    origins = [69, 76, 83, 90]
    write(output / "preregistration.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "excluded_previously_exposed_series": sorted(exposed), "origins": origins, "horizon": 7,
        "selection": "first 69 days quartile volume x stockout burden half, six lowest salted hashes per stratum",
        "salt": "VARO_THIRD_20261017", "model_rule": "EWMA20 default; train-only 5% MAE gain and 2/3 inner-window wins",
        "promotion_gate": "2% lower WAPE vs EWMA20 and OLD_V2; no uncensored worsening; median case MAE and frozen-economic replay not worse than EWMA20",
        "economic_subset": "first 8 salted selected IDs, origin69, all 7 days summed; fixed declared scenario economy",
        "production_approved": False, "external_generalization_fail": "UNCHANGED"})
    dataset = ds.dataset(str(source), format="parquet")
    aggregates = []
    for batch in dataset.to_batches(columns=["date", "location_id", "product_id", "sales_qty", "stockout_hours"],
                                   filter=ds.field("date") < "2024-06-05"):
        frame = batch.to_pandas()
        frame["series"] = frame.location_id.astype(str)+":"+frame.product_id.astype(str)
        frame["censored"] = frame.stockout_hours.gt(0).astype(int)
        aggregates.append(frame.groupby("series").agg(total=("sales_qty", "sum"), count=("sales_qty", "count"),
                                                       censored=("censored", "sum")))
    stats = pd.concat(aggregates).groupby(level=0).sum().reset_index()
    stats = stats[~stats.series.isin(exposed)].copy()
    stats["mean"] = stats.total/stats["count"]
    stats["burden"] = stats.censored/stats["count"]
    stats = stats.sort_values(["mean", "series"])
    stats["scale_quartile"] = np.arange(len(stats))*4//len(stats)
    for _, group in stats.groupby("scale_quartile"):
        ids = group.sort_values(["burden", "series"]).index
        stats.loc[ids, "burden_half"] = np.arange(len(ids))*2//len(ids)
    stats["sample_hash"] = stats.series.map(lambda x: hashlib.sha256(("VARO_THIRD_20261017:"+x).encode()).hexdigest())
    chosen = stats.sort_values("sample_hash").groupby(["scale_quartile", "burden_half"]).head(6)
    chosen.to_csv(output/"selected_training_only_sample.csv", index=False, encoding="utf-8-sig")
    keys = chosen.series.tolist()
    write(output/"exposure_log.json", {"selected_series": keys, "overlap_prior": len(set(keys)&exposed),
          "outer_labels_read_yet": False, "same_dataset_not_external_holdout": True})
    parts = []
    for batch in dataset.to_batches(columns=["date", "location_id", "product_id", "sales_qty", "stockout_hours"]):
        frame = batch.to_pandas()
        frame["series"] = frame.location_id.astype(str)+":"+frame.product_id.astype(str)
        parts.append(frame[frame.series.isin(keys)])
    data = pd.concat(parts, ignore_index=True)
    rows, decisions, economic = [], [], []
    started = time.perf_counter()
    for series, group in data.groupby("series", sort=True):
        group = group.sort_values("date")
        dates = pd.to_datetime(group.date)
        if group.date.duplicated().any() or len(group) < max(origins)+7 or not dates.reset_index(drop=True).equals(
                pd.Series(pd.date_range(dates.min(), dates.max()))):
            raise ValueError("complete daily window covering all declared origins required")
        h = group.sales_qty.to_numpy(float)
        stockout = group.stockout_hours.to_numpy(float)
        for origin in origins:
            selection = forecast.select_third_train_only(h[:origin], ~np.isfinite(stockout[:origin]) | (stockout[:origin]>0))
            decisions.append({"series": series, "origin": origin, **selection})
            if selection["status"] != "CANDIDATE":
                continue
            predictions = {m: forecast.forecast(h[:origin], m) for m in ("OLD_V2", *forecast.THIRD_METHODS)}
            predictions["THIRD_CANDIDATE"] = predictions[selection["selected"]["method"]]
            censor_history = ~np.isfinite(stockout[:origin]) | (stockout[:origin]>0)
            second_choice = forecast.select_second_train_only(h[:origin], censor_history)
            if second_choice["status"] != "CANDIDATE":
                raise AssertionError("common inner eligibility differs")
            second_h = h[:origin].copy()
            if second_choice["selected"]["mask_censored_training"]:
                second_h[censor_history] = np.nan
            predictions["SECOND_CANDIDATE"] = forecast.forecast(second_h, second_choice["selected"]["method"])
            for method, values in predictions.items():
                for step, value in enumerate(values):
                    rows.append({"series": series, "origin": origin, "date": str(group.date.iloc[origin+step]),
                        "method": method, "forecast": value, "observed_sales": h[origin+step],
                        "target_observed": np.isfinite(h[origin+step]), "target_uncensored": stockout[origin+step] == 0,
                        "true_demand_observed": False})
                if series in keys[:8] and origin == 69 and method in {"OLD_V2", "EWMA20", "THIRD_CANDIDATE"}:
                    economic.append({"series": series, "method": method, **forecast_economic_replay(
                        values.sum(), h[origin:origin+7].sum(), float(np.mean(h[:origin])))})
    frame = pd.DataFrame(rows)
    frame.to_csv(output/"predictions.csv", index=False, encoding="utf-8-sig")
    write(output/"decisions.json", decisions)
    write(output/"economic_replay.json", economic)
    case = frame.assign(error=lambda x: (x.forecast-x.observed_sales).abs()).groupby(["series", "origin", "method"]).error.mean().unstack("method")
    case.to_csv(output/"case_errors.csv", encoding="utf-8-sig")
    summary = forecast.summarize(frame)
    lookup = {(r["scope"], r["method"]): r["wape"] for r in summary}
    net = pd.DataFrame(economic).groupby("method").replayed_scenario_net.sum().to_dict()
    all_scope, unc = "ALL_OBSERVED_SALES", "CONFIRMED_NO_STOCKOUT_SALES"
    checks = {"wape_2pct_vs_ewma": lookup[(all_scope, "THIRD_CANDIDATE")] <= .98*lookup[(all_scope, "EWMA20")],
        "wape_2pct_vs_v2": lookup[(all_scope, "THIRD_CANDIDATE")] <= .98*lookup[(all_scope, "OLD_V2")],
        "uncensored_not_worse": lookup[(unc, "THIRD_CANDIDATE")] <= lookup[(unc, "EWMA20")],
        "median_case_not_worse": (case.THIRD_CANDIDATE-case.EWMA20).median() <= 0,
        "economic_not_worse": net["THIRD_CANDIDATE"] >= net["EWMA20"]}
    report = {"series": len(keys), "cases": len(case), "previous_series_overlap": 0, "summary": summary,
        "better_vs_ewma": int((case.THIRD_CANDIDATE < case.EWMA20).sum()),
        "worse_vs_ewma": int((case.THIRD_CANDIDATE > case.EWMA20).sum()),
        "equal_vs_ewma": int((case.THIRD_CANDIDATE == case.EWMA20).sum()), "checks": checks,
        "candidate_passed_preregistered_gate": all(checks.values()), "economic_replay_net": net,
        "default_retention": sum(d.get("reference_retained", False) for d in decisions),
        "seconds_evaluation_including_economic": time.perf_counter()-started,
        "source_sha256": sha(source), "latent_demand": "UNOBSERVED", "production_approved": False,
        "external_generalization_fail": "UNCHANGED", "new_external_dataset": False}
    write(output/"report.json", report)
    return report


def run_dqn(root, output, *, episodes=120):
    import torch
    from services import candidate_dqn as dqn
    from services.learning_policy_audit import inspect_policy, rolling_allocation_bridge
    old_root = root/"outputs/upgrade_20261016/w4/final"
    old_paths = [old_root/f"candidate_seed_{s}.pt" for s in SEEDS]
    before = {str(p): sha(p) for p in old_paths}
    write(output/"preregistration.json", {"seeds": SEEDS, "episodes": episodes,
        "training_seeds": list(range(3100, 3124)), "test_seeds": list(range(4100, 4112)),
        "train_sizes": [2, 3], "test_sizes": [3, 4], "policy_comparators":
        ["UNTRAINED_DQN", "RETAINED_DQN", "THIRD_DQN", "RANDOM_VALID", "GREEDY_IMMEDIATE", "MILP"],
        "mask_ablation": "ranking mask off, invalid proposal terminates without applying", "old_models_unchanged": True,
        "multiperiod_scope": "persistent inventory bridge only, not a trained multiperiod MDP"})
    train = [allocation_problem(s, 2+s%2) for s in range(3100, 3124)]
    test = [allocation_problem(s, 3+s%2) for s in range(4100, 4112)]
    fingerprints = {dqn.decision_fingerprint(p) for p in train}
    rows, training, masks, rolling = [], [], [], []
    torch.set_num_threads(1)
    for seed, path in zip(SEEDS, old_paths):
        blob = torch.load(path, map_location="cpu", weights_only=True)
        network = dqn._network(); network.load_state_dict(blob["state_dict"]); network.eval()
        old = dqn.TrainedCandidate(network, blob["metadata"])
        new = dqn.train(train, seed=seed, episodes=episodes)
        dqn.save_candidate(new, output/f"third_candidate_seed_{seed}.pt")
        training.append(new.metadata)
        for p in test:
            policies = ["UNTRAINED_DQN", "RETAINED_DQN", "THIRD_DQN", "RANDOM_VALID"]
            if seed == SEEDS[0]:
                policies += ["GREEDY_IMMEDIATE", "MILP"]
            for policy in policies:
                r = inspect_policy(p, policy="TRAINED_DQN" if policy in {"RETAINED_DQN", "THIRD_DQN"} else policy,
                                   seed=seed, model=new if policy == "THIRD_DQN" else old)
                e = r["evaluation"]
                rows.append({"case_id": p.problem_id, "policy": policy, "seed": seed, "service": e["service"],
                    "cost": e["cost"], "violations": len(e["violations"]), "seconds": r["seconds"],
                    "training_exact_overlap": dqn.decision_fingerprint(p) in fingerprints,
                    "plan": json.dumps(r["plan"], sort_keys=True), "evidence_class": "SCENARIO"})
            mask = inspect_unmasked_proposal(new, p)
            masks.append({"case_id": p.problem_id, "seed": seed, **mask})
        for p in test[:4]:
            rng = random.Random(int(p.problem_id.split("-")[1])+50000)
            observations = [{k: rng.randint(0, 8) for k in p.needs} for _ in range(3)]
            for label, model, policy in (("THIRD_DQN", new, "TRAINED_DQN"), ("RETAINED_DQN", old, "TRAINED_DQN")):
                r = rolling_allocation_bridge(model, p, observations, policy=policy)
                rolling.append({"case_id": p.problem_id, "seed": seed, "policy": label, **r})
            if seed == SEEDS[0]:
                for policy in ("GREEDY_IMMEDIATE", "MILP"):
                    rolling.append({"case_id": p.problem_id, "seed": seed, "policy": policy,
                        **rolling_allocation_bridge(None, p, observations, policy=policy)})
        print(f"DQN third seed {seed} complete; training {new.metadata['seconds']:.2f}s", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output/"policy_cases.csv", index=False, encoding="utf-8-sig")
    write(output/"training_diagnostics.json", training)
    write(output/"mask_ablation.json", masks)
    write(output/"rolling_bridge.json", rolling)
    summary = frame.groupby("policy").agg(mean_service=("service", "mean"), mean_cost=("cost", "mean"),
        violations=("violations", "sum"), mean_seconds=("seconds", "mean"), max_seconds=("seconds", "max"),
        exact_train_overlap=("training_exact_overlap", "sum")).reset_index().to_dict("records")
    report = {"summary": summary, "training_seconds": sum(t["seconds"] for t in training),
        "model_seeds": list(SEEDS), "episodes_per_seed": episodes,
        "new_model_hashes": {f"third_candidate_seed_{s}.pt": sha(output/f"third_candidate_seed_{s}.pt") for s in SEEDS},
        "training_scope_change": "New mixed 2x2/3x3 scenarios and 120-episode default; retained models used other 2x2 data and 350 episodes. Differences are not isolated causal attribution.",
        "training_gradient_updates": sum(d["gradient_updates"] for t in training for d in t["training_diagnostics"]),
        "mask_invalid_proposals": sum(m["invalid_proposals"] for m in masks), "mask_diagnostic_cases": len(masks),
        "unsafe_action_applied": False, "rolling_summary": pd.DataFrame(rolling).groupby("policy").agg(
            mean_sales_proxy=("total_sales_proxy", "mean"), mean_cost=("total_cost", "mean")).reset_index().to_dict("records"),
        "old_models_before": before, "old_models_after": {str(p): sha(p) for p in old_paths},
        "old_models_unchanged": before == {str(p): sha(p) for p in old_paths}, "production_approved": False,
        "latest_accessed": False, "multiperiod_training": False, "independent_field_generalization": False}
    write(output/"report.json", report)
    return report


def run_aux(root, output, data_root):
    """Expanded score coverage plus exact finite mathematical/period checks."""
    from services.candidate_vhs_normalization import normalized_vhs_candidate, SOURCES
    from services.vhs_score_engine import apply_auto_vhs
    from services.pareto_service import frontier_indices
    from services import inventory_metrics_candidate as metrics
    from services import inventory_metric_integrity as legacy
    universe_path = data_root/"_CANDIDATE_GENERATION_VALIDATION/candidate_universe_rows.parquet"
    universe = pd.read_parquet(universe_path)
    before = sha(universe_path)
    actual_coverage = []
    for day, group in universe[universe.pool_ALL_VALID.eq("IN_POOL")].groupby("snapshot_date"):
        frame = group.rename(columns={"transport_cost": "move_cost"}).copy()
        # route_id can repeat between candidate derivations; preserve unique
        # candidate ID rather than falsely combining different rows.
        frame["route_id"] = frame.candidate_id
        _, audit = normalized_vhs_candidate(frame)
        actual_coverage.append({"date": day, "candidate_count": len(group),
            "active_components": list(audit["active_weights"]), "features": audit["features"],
            "full_vhs_evidence": False, "generator_score_substituted": False})
    write(output/"expanded_real_feature_coverage.json", actual_coverage)
    write(output/"preregistration.json", {"synthetic_cases": "seeds5100..5119, 4x4 routes with full declared raw feature fields",
        "pareto": "exact service/cost/saving non-dominance; no portfolio optimum claim",
        "vhs": "original full VHS vs coverage-consistent six-component research score; no generator substitute",
        "inventory": "7/30/90-day turnover hand values; currency/physical unit scaling; EOQ stationary unconstrained only",
        "real_full_vhs": "NOT_RUN_REQUIRED_FEATURES_ABSENT"})
    rows, plans = [], []
    for seed in range(5100, 5120):
        p = allocation_problem(seed, 4)
        rng = random.Random(seed+10000)
        frame = pd.DataFrame([{"route_id": c.candidate_id, "recommended_qty": c.options[-1].quantity,
            "move_cost": c.options[-1].cost, "expected_saving": rng.randint(1, 100),
            "avoided_disposal_cost": rng.randint(1, 60), "recovered_margin": rng.randint(1, 70),
            "days_to_expiry": rng.randint(1, 15), "demand_forecast_7d": p.needs[c.need_key],
            "sales_30d": rng.randint(5, 60), "promotion_effect": rng.random()*100,
            "distance_km": rng.randint(1, 30), "travel_time_min": rng.randint(5, 60),
            "cutline_passed": True, "time_window_status": "OK", "cold_storage_available": True,
            "route_type": "DIRECT", "final_recommendation": "재고 이동"} for c in p.candidates])
        full = apply_auto_vhs(frame).frame
        normalized, audit = normalized_vhs_candidate(frame)
        # Compare equal finite options with one shared independent feasibility
        # evaluator; this is a declared greedy ordering experiment, not optimal.
        lookup = {c.candidate_id: c for c in p.candidates}
        selections = {}
        for label, ordered in (("FULL_VHS", full.sort_values(["vhs_rank", "route_id"])),
            ("COVERAGE_NORMALIZED_RESEARCH", normalized.sort_values(["candidate_vhs_rank", "route_id"])),
            ("SERVICE_COST_FIRST", frame.set_index("route_id").loc[service_cost_order(p)].reset_index())):
            plan = {}
            for cid in ordered.route_id:
                candidate = lookup[cid]
                trial = {**plan, cid: candidate.options[-1].quantity}
                if not evaluate(p, trial)["violations"]:
                    plan = trial
            e = evaluate(p, plan)
            selections[label] = plan
            rows.append({"case": seed, "strategy": label, "service": e["service"], "cost": e["cost"],
                         "violations": len(e["violations"]), "selected_ids": sorted(plan), "scope": "FULL_FEATURE_SCENARIO"})
        records = frame.to_dict("records")
        points = [(r["recommended_qty"], -r["move_cost"], r["expected_saving"]) for r in records]
        exact = {i for i, x in enumerate(points) if not any(j != i and all(a >= b for a, b in zip(y, x))
                 and any(a > b for a, b in zip(y, x)) for j, y in enumerate(points))}
        plans.append({"case": seed, "pareto_exact": exact == set(frontier_indices(records)),
                      "full_vs_service_cost_selection_changed": selections["FULL_VHS"] != selections["SERVICE_COST_FIRST"],
                      "raw_feature_count": len(frame.columns), "normalized_active_components": list(audit["active_weights"])})
    pd.DataFrame(rows).to_csv(output/"vhs_full_feature_scenarios.csv", index=False, encoding="utf-8-sig")
    write(output/"vhs_pareto_cases.json", plans)
    checks = []
    for days in (7, 30, 90):
        end = pd.Timestamp("2026-07-01")+pd.Timedelta(days=days-1)
        dates = pd.date_range("2026-07-01", end)
        sales = legacy.aggregate_daily_window([{"date": str(d.date()), "quantity": 10} for d in dates],
            window_end=str(end.date()), window_days=days, dataset_id="THIRD_SCENARIO", signal_type=legacy.OBSERVED_RETAIL_SALES,
            quantity_unit="EA", zero_semantics=legacy.ZERO_RECORDED, semantics_evidence=legacy.SOURCE_DOCUMENTED)
        inventory = legacy.average_inventory([{"date": str(d.date()), "quantity": 100} for d in dates],
            period_start="2026-07-01", period_end=str(end.date()), dataset_id="THIRD_SCENARIO", quantity_unit="EA")
        for obj in (sales, inventory):
            obj.update(network_id="N", location_id="S", product_id="P", product_state="CHILLED", lot_id="L")
        r = metrics.aligned_turnover(sales, inventory)
        checks.append({"metric": "PHYSICAL_TURNOVER", "days": days, "expected": days*.1,
                       "actual": r["turnover_value"], "pass": bool(np.isclose(r["turnover_value"], days*.1)),
                       "annualization_applied": False})
    for factor in (.1, 1, 1000):
        r = metrics.eoq(annual_demand=10000, order_cost=500*factor, annual_holding_cost_per_unit=100*factor,
                        quantity_unit="EA", currency="SCENARIO_CURRENCY", stationary_demand=True, perishability=False)
        checks.append({"metric": "EOQ_CURRENCY_SCALE", "scale": factor, "expected": np.sqrt(100000),
                       "actual": r["value"], "pass": bool(np.isclose(r["value"], np.sqrt(100000))),
                       "period": "annual demand and annual holding, no implicit daily/annual mixing"})
    for days in (7, 30, 90):
        end = str((pd.Timestamp("2026-07-01")+pd.Timedelta(days=days-1)).date())
        abc_rows = [{"product_key": key, "network_id": "N", "location_scope": "S",
            "period_start": "2026-07-01", "period_end": end, "quantity_unit": "EA",
            "signal_type": legacy.OBSERVED_RETAIL_SALES, "sales_qty": daily*days,
            "unit_price": 2, "price_quantity_unit": "EA", "currency": "SCENARIO_CURRENCY"}
            for key, daily in (("P", 80), ("Q", 15), ("R", 5))]
        result = metrics.abc_explicit_basis(abc_rows, basis="SALES_REVENUE")
        grades = {k: v["grade"] for k, v in result["grades"].items()}
        checks.append({"metric": "ABC_COMMON_PERIOD", "days": days, "actual": grades,
                       "expected": {"P": "A", "Q": "B", "R": "C"},
                       "pass": grades == {"P": "A", "Q": "B", "R": "C"}})
    from statistics import NormalDist
    for lead in (1, 4, 9):
        result = metrics.safety_stock(daily_mean=10, daily_std=2, lead_time_days=lead,
            cycle_service_level=.95, quantity_unit="EA", demand_basis="SCENARIO_DEMAND",
            stationary_independent=True, constant_lead_time=True)
        expected = NormalDist().inv_cdf(.95)*2*np.sqrt(lead)
        checks.append({"metric": "SAFETY_STOCK_LEAD_PERIOD", "days": lead, "actual": result["value"],
                       "expected": expected, "pass": bool(np.isclose(result["value"], expected)),
                       "assumptions": "Normal IID daily demand, constant lead time; cycle service, not fill rate"})
    for days in (0, 1, 7):
        r = metrics.expiry_exposure(on_hand=50, days_to_expiry=days, daily_demand=8, quantity_unit="EA")
        checks.append({"metric": "EXPIRY_EXPOSURE", "days": days, "actual": r["value"],
                       "expected": max(0, 50-8*days), "pass": r["value"] == max(0, 50-8*days),
                       "observed_waste": False, "probability_available": r["probability"]["value"] is not None})
    write(output/"period_unit_hand_checks.json", checks)
    report = {"expanded_dates": len(actual_coverage), "real_full_vhs": "NOT_RUN_MISSING_EXPIRY_PRICE_PROMOTION_EVIDENCE",
        "actual_source_before": before, "actual_source_after": sha(universe_path),
        "full_feature_synthetic_cases": len(plans), "pareto_exact_pass": sum(p["pareto_exact"] for p in plans),
        "full_vhs_selection_changed_cases": sum(p["full_vs_service_cost_selection_changed"] for p in plans),
        "summary": pd.DataFrame(rows).groupby("strategy").agg(mean_service=("service", "mean"), mean_cost=("cost", "mean"),
            violations=("violations", "sum")).reset_index().to_dict("records"),
        "period_unit_checks": len(checks), "period_unit_checks_pass": sum(c["pass"] for c in checks),
        "actual_waste_probability": "UNVERIFIED", "production_approved": False}
    write(output/"report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("forecast", "dqn", "aux"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("C:/VARO_V2_REAL_DATA"))
    parser.add_argument("--episodes", type=int, default=120)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.mode == "forecast":
        result = run_forecast(root, args.output_dir, args.data_root)
    elif args.mode == "aux":
        result = run_aux(root, args.output_dir, args.data_root)
    else:
        result = run_dqn(root, args.output_dir, episodes=args.episodes)
    print(json.dumps(clean(result), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
