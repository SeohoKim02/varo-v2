import numpy as np
import pandas as pd
import pytest
from services import forecast_candidate as f


def frame():
    return pd.DataFrame({"series": "controlled", "date": pd.date_range("2025-01-01", periods=100),
                         "sales_qty": np.resize([2., 3., 4., 2., 3., 8., 9.], 100), "stockout_hours": 0.0})


@pytest.mark.parametrize("method", [*f.METHODS, "OLD_V1", "OLD_V2"])
def test_constant_forecast(method):
    values = f.forecast(np.full(70, 3.), method)
    assert np.allclose(values, 3.)


def test_future_changes_do_not_change_selection_or_forecast():
    data = frame()
    first, first_selection = f.evaluate_rolling(data, [70])
    data.loc[70:, "sales_qty"] = 9999
    data.loc[70:, "stockout_hours"] = 24
    second, second_selection = f.evaluate_rolling(data, [70])
    assert first_selection == second_selection
    assert first["forecast"].tolist() == second["forecast"].tolist()
    assert second["target_uncensored"].eq(False).all()
    assert second["true_demand_observed"].eq(False).all()


def test_zero_missing_and_no_history_are_different():
    assert f.forecast([0.] * 42, "MA7").sum() == 0
    with pytest.raises(ValueError, match="no observed"):
        f.forecast([np.nan] * 42, "MA7")
    with pytest.raises(ValueError, match="returns"):
        f.forecast([1, -1, 3], "MA7")


def test_duplicate_days_not_silently_merged():
    data = frame()
    with pytest.raises(ValueError, match="duplicate"):
        f.evaluate_rolling(pd.concat([data, data.head(1)]), [70])


def test_all_stockout_training_cannot_select_on_fake_labels():
    result = f.select_train_only(np.ones(70), np.ones(70, bool))
    assert result["status"] == "INSUFFICIENT_UNCENSORED_VALIDATION"


def test_uncensored_scope_and_zero_denominator_explicit():
    data = frame()
    data["sales_qty"] = 0
    data.loc[70:72, "stockout_hours"] = 5
    rows, _ = f.evaluate_rolling(data, [70])
    summary = f.summarize(rows)
    assert all(r["wape"] is None for r in summary)
    assert all(r["rows"] == (7 if r["scope"] == "ALL_OBSERVED_SALES" else 4) for r in summary)


def test_series_row_order_does_not_change_results():
    a, _ = f.evaluate_rolling(frame(), [70])
    b, _ = f.evaluate_rolling(frame().sample(frac=1, random_state=4), [70])
    pd.testing.assert_frame_equal(a, b)


def test_unknown_availability_is_not_uncensored_validation():
    import numpy as np
    import pandas as pd
    from services.forecast_candidate import evaluate_rolling
    frame = pd.DataFrame({"series": "s", "date": pd.date_range("2026-01-01", periods=56),
                          "sales_qty": np.ones(56), "stockout_hours": np.full(56, np.nan)})
    rows, decisions = evaluate_rolling(frame, [49])
    assert rows.empty
    assert decisions[0]["status"] == "INSUFFICIENT_UNCENSORED_VALIDATION"
