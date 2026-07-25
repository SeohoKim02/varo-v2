"""Tests for Varo V2 network layout and simulation contracts."""
import unittest

from simulation.network_layout import calculate_network_layout
from simulation.network_simulator import (
    advance_simulation,
    create_simulation,
    pause_simulation,
    restart_simulation,
    start_simulation,
)
from simulation.route_animation import build_route_legs
from simulation.simulation_state import AT_DC, COMPLETED, DC_DWELL_TICKS, MOVING, PAUSED, READY, SPEED_STEPS


NODES = [
    {"node_id": "DC01", "node_name": "서울 서북권 물류센터", "node_type": "DC"},
    {"node_id": "S01", "node_name": "연신내점", "node_type": "STORE"},
    {"node_id": "S02", "node_name": "응암점", "node_type": "STORE"},
    {"node_id": "S03", "node_name": "홍제점", "node_type": "STORE"},
]

DIRECT_ROUTE = {
    "route_id": "R001",
    "product_name": "냉동만두",
    "source_id": "S01",
    "target_id": "S03",
    "dc_id": None,
    "route_type": "DIRECT",
    "recommended_qty": 12,
    "transport_type": "냉동/냉장 탑차",
    "status": "READY",
}

VIA_DC_ROUTE = {
    "route_id": "R002",
    "product_name": "샐러드",
    "source_id": "S02",
    "target_id": "S03",
    "dc_id": "DC01",
    "route_type": "VIA_DC",
    "recommended_qty": 8,
    "transport_type": "냉동/냉장 탑차",
    "status": "READY",
}


class NetworkLayoutTests(unittest.TestCase):
    def test_dc_one_and_multiple_stores_layout(self):
        result = calculate_network_layout(NODES)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.dc["node_id"], "DC01")
        self.assertEqual(len(result.stores), 3)
        self.assertEqual(result.dc["x"], result.canvas["width"] / 2)
        self.assertEqual(result.dc["y"], result.canvas["height"] / 2)
        self.assertGreaterEqual(result.canvas["height"], 650)
        self.assertGreater(result.dc["width"], result.stores[0]["width"])

    def test_store_count_matches_coordinate_count(self):
        nodes = NODES + [{"node_id": "S04", "node_name": "불광점", "node_type": "STORE"}]
        result = calculate_network_layout(nodes)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.stores), 4)

    def test_missing_dc_returns_error(self):
        result = calculate_network_layout([node for node in NODES if node["node_type"] != "DC"])
        self.assertFalse(result.is_valid)
        self.assertIn("DC", result.errors[0])

    def test_two_dcs_returns_error(self):
        nodes = NODES + [{"node_id": "DC02", "node_name": "보조 센터", "node_type": "DC"}]
        result = calculate_network_layout(nodes)
        self.assertFalse(result.is_valid)
        self.assertIn("2개", result.errors[0])


class RouteParsingTests(unittest.TestCase):
    def test_direct_route_parsing(self):
        legs = build_route_legs(DIRECT_ROUTE)
        self.assertEqual(len(legs), 1)
        self.assertEqual(legs[0]["from_node_id"], "S01")
        self.assertEqual(legs[0]["to_node_id"], "S03")
        self.assertEqual(legs[0]["phase"], "DIRECT")

    def test_via_dc_route_parsing(self):
        legs = build_route_legs(VIA_DC_ROUTE)
        self.assertEqual(len(legs), 2)
        self.assertEqual(legs[0]["to_node_id"], "DC01")
        self.assertEqual(legs[1]["from_node_id"], "DC01")

    def test_unsupported_route_type_validation(self):
        route = dict(DIRECT_ROUTE, route_id="R099", route_type="UNKNOWN")
        with self.assertRaises(ValueError):
            build_route_legs(route)


class SimulationTests(unittest.TestCase):
    def test_max_five_routes(self):
        routes = [dict(DIRECT_ROUTE, route_id=f"R00{i}") for i in range(1, 8)]
        snapshot = create_simulation(routes)
        self.assertEqual(len(snapshot.routes), 5)

    def test_direct_never_uses_at_dc(self):
        snapshot = start_simulation(create_simulation([DIRECT_ROUTE]))
        snapshot = advance_simulation(snapshot, step=1.0)
        state = snapshot.route_states["R001"]
        self.assertEqual(state.status, COMPLETED)
        self.assertNotEqual(state.status, AT_DC)

    def test_via_dc_pauses_before_second_leg(self):
        snapshot = start_simulation(create_simulation([VIA_DC_ROUTE]))
        self.assertEqual(snapshot.route_states["R002"].status, MOVING)
        snapshot = advance_simulation(snapshot, step=1.0)
        self.assertEqual(snapshot.route_states["R002"].status, AT_DC)
        for expected_tick in range(1, DC_DWELL_TICKS + 1):
            snapshot = advance_simulation(snapshot, step=1.0)
            state = snapshot.route_states["R002"]
            self.assertEqual(state.status, AT_DC)
            self.assertEqual(state.dc_wait_ticks, expected_tick)
        snapshot = advance_simulation(snapshot, step=1.0)
        self.assertEqual(snapshot.route_states["R002"].status, MOVING)
        snapshot = advance_simulation(snapshot, step=1.0)
        self.assertEqual(snapshot.route_states["R002"].status, COMPLETED)

    def test_pause_and_restart_preserve_expected_positions(self):
        snapshot = start_simulation(create_simulation([DIRECT_ROUTE]))
        snapshot = advance_simulation(snapshot, step=0.2)
        moving_progress = snapshot.route_states["R001"].progress
        snapshot = pause_simulation(snapshot)
        self.assertEqual(snapshot.route_states["R001"].status, PAUSED)
        self.assertEqual(snapshot.route_states["R001"].progress, moving_progress)
        snapshot = advance_simulation(snapshot, step=0.2)
        self.assertEqual(snapshot.route_states["R001"].progress, moving_progress)
        snapshot = restart_simulation(snapshot)
        self.assertEqual(snapshot.route_states["R001"].status, READY)
        self.assertEqual(snapshot.route_states["R001"].progress, 0.0)
        self.assertEqual(snapshot.logs, [])

    def test_speed_steps_match_slow_normal_fast_order(self):
        self.assertLess(SPEED_STEPS["느림"], SPEED_STEPS["보통"])
        self.assertLess(SPEED_STEPS["보통"], SPEED_STEPS["빠름"])
        expected_seconds = {"느림": (24.0, 27.0), "보통": (14.0, 18.0), "빠름": (8.0, 12.0)}
        for label, bounds in expected_seconds.items():
            snapshot = start_simulation(create_simulation([DIRECT_ROUTE], speed_label=label))
            ticks = 0
            while snapshot.is_running and ticks < 200:
                snapshot = advance_simulation(snapshot)
                ticks += 1
            seconds = ticks * 0.35
            self.assertGreaterEqual(seconds, bounds[0])
            self.assertLessEqual(seconds, bounds[1])

    def test_duplicate_route_id_returns_error(self):
        snapshot = create_simulation([DIRECT_ROUTE, dict(VIA_DC_ROUTE, route_id="R001")])
        self.assertTrue(snapshot.errors)
        self.assertIn("중복 route_id", snapshot.errors[0])


if __name__ == "__main__":
    unittest.main()