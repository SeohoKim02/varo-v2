import json
import pandas as pd
import pytest

from services.second_upgrade_evidence import LONG_FIELDS, COMPARISON_FIELDS, build


def test_absent_evidence_stays_not_run_and_output_is_exclusive(tmp_path):
    root = tmp_path / "batch"; root.mkdir()
    output = tmp_path / "evidence1"
    result = build(root, output)
    frame = pd.read_csv(output / "evidence_longform.csv", keep_default_na=False)
    assert result["longform_rows"] == 6
    assert list(frame.columns[:20]) == LONG_FIELDS
    assert set(frame.solver_status) == {"NOT_RUN"}
    assert set(frame.metric_value) == {"NOT_RUN"}
    assert not frame.runtime_seconds.astype(str).str.contains("0.0").any()
    assert list(pd.read_csv(output / "comparison_17_columns.csv").columns) == COMPARISON_FIELDS
    before = (output / "evidence_manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        build(root, output)
    assert (output / "evidence_manifest.json").read_bytes() == before


def test_risk_tradeoff_is_two_objectives_not_unconditional_gain(tmp_path):
    root = tmp_path / "batch"; r2 = root / "r2/run1"; r2.mkdir(parents=True)
    report = {"fair_baselines": {}, "adaptive_policy_cases": {}, "risk_tradeoff": {
        "expected": {"objective": "EXPECTED_NET_CASH_MAX", "expected_net_value": 51,
                     "worst_net_value": 26, "constraint_violations": 0, "runtime_seconds": .1},
        "worst_case": {"objective": "WORST_NET_CASH_MAX", "expected_net_value": 30,
                       "worst_net_value": 30, "constraint_violations": 0, "runtime_seconds": .1}}}
    path = r2 / "economic_second_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    pd.DataFrame(columns=LONG_FIELDS).to_csv(r2 / "economic_long_form.csv", index=False)
    output = tmp_path / "evidence"
    build(root, output)
    frame = pd.read_csv(output / "comparison_17_columns.csv")
    risk = frame[frame["문제/데이터"].str.contains("uncertainty")]
    assert len(risk) == 2
    assert set(risk["목적함수"]) == {"EXPECTED_NET_CASH_MAX", "WORST_NET_CASH_MAX"}
    assert set(risk["원시 결과"]) == {"기대 51; 최악 26", "기대 30; 최악 30"}
    assert risk["최종 선택 결과"].str.contains("trade-off").all()
