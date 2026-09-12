"""Validation and candidate generation for an explicit direct network."""
from __future__ import annotations

import pandas as pd

from services.candidate_generator import generate_candidates
from services.data_validator import validate_workbook_data
from tests.fixtures import sample_workbook


def _workbook_without_dc(*, direct_network: bool) -> dict[str, pd.DataFrame]:
    data = sample_workbook()
    data.pop("recommendations")
    data["stores"] = data["stores"].loc[
        data["stores"]["node_type"] == "STORE"
    ].copy()
    if direct_network:
        data["stores"]["network_mode"] = "DIRECT_NETWORK"
    data["inventory"] = data["inventory"].loc[
        data["inventory"]["store_id"] != "DC01"
    ].copy()
    data["routes"] = data["routes"].loc[
        (data["routes"]["source_id"] != "DC01")
        & (data["routes"]["target_id"] != "DC01")
    ].copy()
    return data


def test_direct_network_without_dc_passes_validation():
    data = _workbook_without_dc(direct_network=True)
    data["recommendations"], _ = generate_candidates(data)

    report = validate_workbook_data(data)

    assert not report.has_errors, report.messages
    assert report.summary["dc_count"] == 0


def test_direct_network_without_dc_generates_direct_candidates():
    frame, info = generate_candidates(
        _workbook_without_dc(direct_network=True)
    )

    assert info["generated"]
    assert frame is not None and not frame.empty
    assert info["direct_count"] == len(frame)
    assert info["via_dc_count"] == 0
    assert set(frame["route_type"]) == {"DIRECT"}


def test_normal_network_without_dc_fails_validation():
    data = _workbook_without_dc(direct_network=False)
    data["recommendations"], _ = generate_candidates(data)

    report = validate_workbook_data(data)

    assert report.has_errors
    assert any(
        message.message == "DC가 1개 이상 필요합니다."
        for message in report.messages
    )
