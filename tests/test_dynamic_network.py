"""Tests for the data-driven multi-DC home network."""
import unittest

from simulation.dynamic_network import (
    build_network_nodes,
    build_route_segments,
    compute_dynamic_layout,
)


class DynamicNodeTests(unittest.TestCase):
    def test_store_count_changes_rendered_node_count(self):
        for count in (5, 10, 30, 35):
            nodes = [{"node_id": "HUB", "node_name": "중앙 물류센터", "node_type": "DC"}]
            nodes.extend(
                {"node_id": f"ST{i:02d}", "node_name": f"점포 {i:02d}", "node_type": "STORE"}
                for i in range(count)
            )
            layout = compute_dynamic_layout(nodes)
            self.assertTrue(layout.is_valid)
            self.assertEqual(len(layout.stores), count)
            self.assertEqual(layout.canvas["store_count"], count)

    def test_one_two_and_four_dcs_are_all_laid_out(self):
        for dc_count in (1, 2, 4):
            nodes = [
                {"node_id": f"HUB{i}", "node_name": f"물류센터 {i}", "node_type": "DC"}
                for i in range(dc_count)
            ] + [{"node_id": "STORE1", "node_name": "테스트점", "node_type": "STORE"}]
            layout = compute_dynamic_layout(nodes)
            self.assertTrue(layout.is_valid)
            self.assertEqual(len(layout.dcs), dc_count)
            self.assertEqual(len({(node["x"], node["y"]) for node in layout.dcs}), dc_count)

    def test_radial_layout_is_deterministic_and_centers_the_dc(self):
        nodes = [
            {"node_id": "HUB", "node_name": "중앙 센터", "node_type": "DC", "lat": 37.55, "lon": 126.95},
            {"node_id": "A", "node_name": "A점", "node_type": "STORE", "lat": 37.50, "lon": 126.90},
            {"node_id": "B", "node_name": "B점", "node_type": "STORE", "lat": 37.60, "lon": 127.00},
        ]
        first = compute_dynamic_layout(nodes)
        second = compute_dynamic_layout(list(reversed(nodes)))
        positions1 = {node["node_id"]: (node["x"], node["y"]) for node in first.dcs + first.stores}
        positions2 = {node["node_id"]: (node["x"], node["y"]) for node in second.dcs + second.stores}
        # Placement ignores lat/lon: it is deterministic and DC-centric, not geographic.
        self.assertEqual(first.canvas["layout_mode"], "radial")
        self.assertEqual(positions1, positions2)
        hub = first.dcs[0]
        self.assertAlmostEqual(hub["x"], first.canvas["width"] / 2.0, delta=1.0)
        self.assertAlmostEqual(hub["y"], first.canvas["height"] / 2.0, delta=1.0)

    def test_aliases_and_referenced_nodes_are_normalized(self):
        data = {"stores": [
            {"store_id": "A", "store_name": "A점", "store_type": "STORE"},
            {"store_id": "H1", "store_name": "동부 물류센터", "store_type": "DC"},
        ]}
        routes = [{"source_id": "A", "target_id": "B", "target_name": "B점", "dc_id": "H2", "dc_name": "서부 센터"}]
        nodes = build_network_nodes(data, routes)
        self.assertEqual(len([node for node in nodes if node["node_type"] == "DC"]), 2)
        self.assertIn("B", {node["node_id"] for node in nodes})

    def test_inventory_can_supply_store_nodes_when_store_sheet_is_empty(self):
        data = {"inventory": [
            {"store_id": "S1", "store_name": "첫째점", "product_id": "P1"},
            {"store_id": "S1", "store_name": "첫째점", "product_id": "P2"},
            {"store_id": "S2", "store_name": "둘째점", "product_id": "P1"},
        ]}
        routes = [{"source_id": "S1", "target_id": "S2", "dc_id": "HUB", "dc_name": "중앙 센터"}]
        nodes = build_network_nodes(data, routes)
        self.assertEqual({node["node_id"] for node in nodes}, {"S1", "S2", "HUB"})


class DynamicRouteTests(unittest.TestCase):
    def setUp(self):
        self.nodes = [
            {"node_id": "DC_A", "node_name": "서부 센터", "node_type": "DC"},
            {"node_id": "DC_B", "node_name": "동부 센터", "node_type": "DC"},
            {"node_id": "S1", "node_name": "출발점", "node_type": "STORE"},
            {"node_id": "S2", "node_name": "도착점", "node_type": "STORE"},
        ]

    def test_direct_route_builds_one_segment(self):
        route = {"source_id": "S1", "target_id": "S2", "route_type": "직접 이동"}
        self.assertEqual(build_route_segments(route, self.nodes), [
            {"from_node_id": "S1", "to_node_id": "S2", "phase": "DIRECT"}
        ])

    def test_via_route_uses_its_own_dc_not_fixed_dc(self):
        route = {"source_id": "S1", "target_id": "S2", "route_type": "VIA_DC", "dc_id": "DC_B"}
        segments = build_route_segments(route, self.nodes)
        self.assertEqual(segments[0]["to_node_id"], "DC_B")
        self.assertEqual(segments[1]["from_node_id"], "DC_B")
        self.assertNotIn("DC_A", {value for segment in segments for value in segment.values()})

    def test_via_route_can_resolve_dc_name(self):
        route = {"source_id": "S1", "target_id": "S2", "route_type": "DC 경유", "dc_name": "서부 센터"}
        segments = build_route_segments(route, self.nodes)
        self.assertEqual(segments[0]["to_node_id"], "DC_A")

    def test_via_route_without_dc_uses_nearest_available_dc_deterministically(self):
        nodes = [
            {"node_id": "DC_A", "node_name": "서부 센터", "node_type": "DC", "latitude": 37.50, "longitude": 126.90},
            {"node_id": "DC_B", "node_name": "동부 센터", "node_type": "DC", "latitude": 37.80, "longitude": 127.30},
            {"node_id": "S1", "node_name": "출발점", "node_type": "STORE", "latitude": 37.51, "longitude": 126.91},
            {"node_id": "S2", "node_name": "도착점", "node_type": "STORE", "latitude": 37.60, "longitude": 127.00},
        ]
        route = {"source_id": "S1", "target_id": "S2", "route_type": "VIA_DC"}
        first = build_route_segments(route, nodes)
        second = build_route_segments(route, list(reversed(nodes)))
        self.assertEqual(first[0]["to_node_id"], "DC_A")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
