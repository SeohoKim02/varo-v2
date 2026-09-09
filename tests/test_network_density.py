"""What the 재고 운영 network must still do when the plan gets big.

The bundled sample workbooks draw 11 nodes. These checks push the same drawing
code to 60 with :mod:`tests.network_fixtures` and hold it to the things a user
depends on at that size:

* the selected move is always fully readable — its stores, its 물류센터, its role
  labels and its ``planned_qty`` — however dense the rest of the picture is,
* no two boxes, names or quantity chips are drawn on top of each other,
* nothing drops below ~10px once the SVG is scaled into the centre column,
* the same plan always produces the same picture, whatever order its rows arrive
  in, and picking a different move never moves a node.

The screen sizes are not invented. Each ``COLUMN_WIDTH`` below was measured in
headless Chrome against the running app at that viewport, with the sidebar in the
stated state; the SVG is scaled by ``column width / CANVAS_WIDTH``, so those
ratios are what turns a user unit into a pixel.
"""
from __future__ import annotations

import re
import time
import unittest

from components import workspace_network as wsn
from components.workspace_network import (
    ROLE_DC,
    ROLE_SOURCE,
    ROLE_TARGET,
    SCOPE_ALL,
    SCOPE_PLAN,
    SCOPE_SELECTED,
    build_workspace_network,
)
from simulation.flow_layout import (
    CANVAS_WIDTH,
    DC_BAND,
    MAX_HEIGHT,
    MIN_HEIGHT,
    compute_flow_layout,
    dc_dimensions,
    store_dimensions,
)
from tests.network_fixtures import NETWORK_SIZES, SHAPES, dense_case

#: Centre-column width per supported desktop, measured in Chrome against the
#: running app. 1366 with the sidebar expanded is the narrowest the workspace
#: ever gets, so it is the one every legibility floor is judged against.
COLUMN_WIDTH = {
    "1920": 755.0,
    "1920+sidebar": 735.0,
    "1600": 832.0,
    "1600+sidebar": 787.0,
    "1366": 763.0,
    "1366+sidebar": 646.0,
}
NARROWEST = min(COLUMN_WIDTH.values())
#: Below this a label stops being readable on screen. Measured, not styled.
MIN_SCREEN_PX = 10.0

_NODE_RE = re.compile(
    r'<g class="ws-node([^"]*)" transform="translate\(([-\d.]+) ([-\d.]+)\)">(.*?)</g>'
)
_RECT_RE = re.compile(r'<rect x="([-\d.]+)" y="([-\d.]+)" width="([\d.]+)" height="([\d.]+)"')
_CHIP_RE = re.compile(
    r'<rect class="ws-edge-chip" x="([-\d.]+)" y="([-\d.]+)" width="([\d.]+)" height="([\d.]+)"'
)
_NAME_RE = re.compile(r'<text class="ws-node-name"[^>]*style="font-size:([\d.]+)px">([^<]*)</text>')
_PILL_RE = re.compile(r'font-size="([\d.]+)" font-weight="700">([^<]*)</text>')
_VIEWBOX_RE = re.compile(r'viewBox="0 0 ([\d.]+) ([\d.]+)"')

Rect = tuple[float, float, float, float]  # left, top, width, height


def _hits(first: Rect, second: Rect) -> bool:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


class Picture:
    """The drawn SVG, parsed back into the boxes and sizes a reader sees."""

    def __init__(self, html: str) -> None:
        self.html = html
        viewbox = _VIEWBOX_RE.search(html)
        self.width = float(viewbox.group(1)) if viewbox else 0.0
        self.height = float(viewbox.group(2)) if viewbox else 0.0
        self.nodes: list[dict] = []
        self.name_fonts: list[float] = []
        self.pill_fonts: list[float] = []
        for match in _NODE_RE.finditer(html):
            classes, x, y, body = match.group(1), float(match.group(2)), float(match.group(3)), match.group(4)
            rect = _RECT_RE.search(body)
            if not rect:
                continue
            left, top, width, height = (float(value) for value in rect.groups())
            names = _NAME_RE.findall(body)
            self.nodes.append({
                "box": (x + left, y + top, width, height),
                "focus": "focus" in classes,
                "context": "context" in classes,
                "dc": "dc" in classes,
                "names": [text for _size, text in names],
            })
            self.name_fonts += [float(size) for size, _text in names]
            self.pill_fonts += [float(size) for size, _text in _PILL_RE.findall(body)]
        self.chips: list[Rect] = [
            tuple(float(value) for value in match.groups())  # type: ignore[misc]
            for match in _CHIP_RE.finditer(html)
        ]
        self.chip_labels = re.findall(r'class="ws-edge-label"[^>]*>([^<]*)</text>', html)
        self.roles = re.findall(r'class="ws-node-role"[^>]*>([^<]*)</text>', html)

    @property
    def fonts(self) -> list[float]:
        return self.name_fonts + self.pill_fonts + [wsn.EDGE_FONT]

    def named(self, name: str) -> dict | None:
        return next((node for node in self.nodes if name in node["names"]), None)


def _draw(case: dict, scope: str = SCOPE_PLAN) -> Picture:
    network = build_workspace_network(
        case["data"], case["items"], case["selected_route_id"],
        scope=scope, store_states=case["states"],
    )
    assert network["ok"], network["message"]
    return Picture(network["html"])


class DenseLayoutTests(unittest.TestCase):
    """A · every size the picture is asked to draw, it draws in full."""

    def test_every_node_is_drawn_up_to_sixty(self):
        for size in NETWORK_SIZES:
            with self.subTest(nodes=size):
                case = dense_case(size, dc_count=2 if size >= 16 else 1)
                picture = _draw(case, scope=SCOPE_ALL)
                self.assertEqual(len(picture.nodes), size)

    def test_no_two_node_boxes_overlap(self):
        for size in NETWORK_SIZES:
            for shape in SHAPES:
                with self.subTest(nodes=size, shape=shape):
                    picture = _draw(dense_case(size, dc_count=2, shape=shape), scope=SCOPE_ALL)
                    boxes = [node["box"] for node in picture.nodes]
                    for first in range(len(boxes)):
                        for second in range(first + 1, len(boxes)):
                            self.assertFalse(
                                _hits(boxes[first], boxes[second]),
                                f"{boxes[first]} overlaps {boxes[second]}",
                            )

    def test_nothing_is_drawn_outside_the_canvas(self):
        for size in NETWORK_SIZES:
            with self.subTest(nodes=size):
                picture = _draw(dense_case(size, dc_count=2), scope=SCOPE_ALL)
                for node in picture.nodes:
                    left, top, width, height = node["box"]
                    self.assertGreaterEqual(left, -1)
                    self.assertLessEqual(left + width, picture.width + 1)
                    # The role label sits above the box; keep its room too.
                    self.assertGreaterEqual(top - 24, -1)
                    self.assertLessEqual(top + height, picture.height + 1)
                for left, top, width, height in picture.chips:
                    self.assertGreaterEqual(left, -1)
                    self.assertLessEqual(left + width, picture.width + 1)
                    self.assertGreaterEqual(top, -1)
                    self.assertLessEqual(top + height, picture.height + 1)

    def test_the_canvas_grows_in_height_only_and_within_bounds(self):
        """Width is fixed, so a denser plan never shrinks the picture sideways."""
        heights = []
        for size in NETWORK_SIZES:
            picture = _draw(dense_case(size, dc_count=2), scope=SCOPE_ALL)
            self.assertEqual(picture.width, CANVAS_WIDTH)
            self.assertGreaterEqual(picture.height, MIN_HEIGHT)
            self.assertLessEqual(picture.height, MAX_HEIGHT)
            heights.append(picture.height)
        self.assertEqual(heights, sorted(heights), "height must not fall as nodes are added")
        self.assertEqual(heights[0], MIN_HEIGHT, "a small plan keeps the original canvas")

    def test_a_plan_beyond_the_canvas_says_what_it_left_out(self):
        # Far more stores than the canvas can hold, but only a handful of moves:
        # the stores those moves touch are the ones that must survive.
        case = dense_case(90, dc_count=2, move_count=4)
        network = build_workspace_network(
            case["data"], case["items"], case["selected_route_id"],
            scope=SCOPE_ALL, store_states=case["states"],
        )
        self.assertTrue(network["ok"])
        self.assertIn("표시하지 않은 점포", network["message"])
        picture = Picture(network["html"])
        # Whatever is dropped, it is never a store the plan actually moves.
        for item in case["items"]:
            for name in (item["source_name"], item["target_name"]):
                self.assertIsNotNone(picture.named(name), f"{name} was dropped")


class DeterminismTests(unittest.TestCase):
    """B · the same plan is always the same picture."""

    def _layout(self, stores, dcs, edges):
        return compute_flow_layout(stores, dcs, edges, keep=stores)

    def test_the_same_input_gives_the_same_positions(self):
        case = dense_case(40, dc_count=2)
        first, second = _draw(case), _draw(case)
        self.assertEqual(first.html, second.html)

    def test_shuffled_rows_give_the_same_positions(self):
        import random

        stores = [f"S{index:03d}" for index in range(38)]
        dcs = ["DC01", "DC02"]
        edges = [
            (stores[index], dcs[index % 2]) for index in range(0, 18)
        ] + [
            (dcs[index % 2], stores[19 + index]) for index in range(0, 18)
        ]
        expected = self._layout(stores, dcs, edges)
        for seed in range(6):
            shuffler = random.Random(seed)
            shuffled_stores, shuffled_dcs, shuffled_edges = list(stores), list(dcs), list(edges)
            shuffler.shuffle(shuffled_stores)
            shuffler.shuffle(shuffled_dcs)
            shuffler.shuffle(shuffled_edges)
            with self.subTest(seed=seed):
                self.assertEqual(
                    self._layout(shuffled_stores, shuffled_dcs, shuffled_edges).positions,
                    expected.positions,
                )

    def test_shuffled_plan_rows_give_the_same_picture(self):
        import random

        case = dense_case(24, dc_count=2)
        boxes = {
            tuple(node["names"]): node["box"] for node in _draw(case).nodes
        }
        for seed in range(4):
            shuffled = dict(case)
            items = list(case["items"])
            random.Random(seed).shuffle(items)
            shuffled["items"] = items
            with self.subTest(seed=seed):
                self.assertEqual(
                    {tuple(node["names"]): node["box"] for node in _draw(shuffled).nodes},
                    boxes,
                )

    def test_changing_the_selection_never_moves_a_node(self):
        """Picking another move re-emphasises the picture; it does not redraw it."""
        case = dense_case(40, dc_count=2)
        def positions(route_id: str) -> dict:
            network = build_workspace_network(
                case["data"], case["items"], route_id,
                scope=SCOPE_PLAN, store_states=case["states"],
            )
            picture = Picture(network["html"])
            # A focused node is drawn wider on purpose, so compare centres.
            return {
                tuple(node["names"]): round(node["box"][0] + node["box"][2] / 2, 2)
                for node in picture.nodes
            }
        first = positions(case["items"][0]["route_id"])
        for item in case["items"][1:6]:
            with self.subTest(route=item["route_id"]):
                self.assertEqual(positions(item["route_id"]), first)


class SelectedMoveTests(unittest.TestCase):
    """C · whatever else is on screen, the chosen move can be read."""

    def _selected(self, size: int, shape: str, dc_count: int = 2) -> tuple[dict, Picture]:
        case = dense_case(size, dc_count=dc_count, shape=shape)
        return case, _draw(case)

    def test_direct_and_via_dc_both_keep_their_roles_and_quantity(self):
        for size in (16, 24, 40, 60):
            for shape in ("direct", "via_dc"):
                with self.subTest(nodes=size, shape=shape):
                    case, picture = self._selected(size, shape)
                    selected = case["items"][0]
                    self.assertIn(ROLE_SOURCE, picture.roles)
                    self.assertIn(ROLE_TARGET, picture.roles)
                    if shape == "via_dc":
                        self.assertIn(ROLE_DC, picture.roles)
                        self.assertIn("ws-edge-ribbon", picture.html)
                    self.assertIn(f"{selected['planned_qty']:,}개", picture.chip_labels)

    def test_the_selected_stores_are_drawn_larger_than_their_neighbours(self):
        for size in (24, 40, 60):
            with self.subTest(nodes=size):
                _case, picture = self._selected(size, "direct")
                focus = [n for n in picture.nodes if n["focus"] and not n["dc"]]
                plain = [n for n in picture.nodes if not n["focus"] and not n["dc"]]
                self.assertTrue(focus)
                self.assertGreater(min(n["box"][2] for n in focus), max(n["box"][2] for n in plain))

    def test_the_selected_stores_keep_the_written_state_badge_when_others_lose_it(self):
        """Dense pictures swap the badge for an outline marker — except on the
        move the user is looking at."""
        _case, picture = self._selected(60, "direct")
        self.assertIn("ws-node-marker", picture.html)
        focus_bodies = [
            body for classes, _x, _y, body in _NODE_RE.findall(picture.html) if "focus" in classes
        ]
        self.assertTrue(focus_bodies)
        for body in focus_bodies:
            self.assertTrue(_PILL_RE.search(body), "the selected move lost its state badge")

    def test_a_move_through_either_dc_names_the_one_it_uses(self):
        case = dense_case(24, dc_count=2, shape="via_dc")
        by_dc = {}
        for item in case["items"]:
            by_dc.setdefault(item["dc_id"], item)
        self.assertEqual(sorted(by_dc), ["DC01", "DC02"])
        for dc_id, item in by_dc.items():
            with self.subTest(dc=dc_id):
                network = build_workspace_network(
                    case["data"], case["items"], item["route_id"],
                    scope=SCOPE_PLAN, store_states=case["states"],
                )
                picture = Picture(network["html"])
                self.assertIn(ROLE_DC, picture.roles)
                # With both 물류센터 on screen, each is told apart by its code as
                # well as its name, and only the one this move uses is marked.
                for code in ("DC01", "DC02"):
                    self.assertIn(f">{code}<", picture.html)
                marked = re.search(
                    r'<g class="ws-node ws-node-dc"[^>]*>(?:(?!</g>).)*?경유 DC(?:(?!</g>).)*?</g>',
                    picture.html, re.S,
                )
                self.assertIsNotNone(marked)
                self.assertIn(f">{dc_id}<", marked.group(0))

    def test_only_the_selected_move_is_drawn_when_the_scope_narrows_to_it(self):
        case = dense_case(24, dc_count=2, shape="via_dc")
        picture = _draw(case, scope=SCOPE_SELECTED)
        self.assertEqual(len(picture.chips), 1)
        self.assertIn(ROLE_DC, picture.roles)

    def test_the_selected_quantity_survives_a_plan_of_thirty_moves(self):
        case = dense_case(60, dc_count=2, move_count=29)
        picture = _draw(case)
        self.assertIn(f"{case['items'][0]['planned_qty']:,}개", picture.chip_labels)


class OverlapTests(unittest.TestCase):
    """D · nothing readable is drawn on top of anything else readable."""

    def test_quantity_chips_clear_the_node_boxes_and_each_other(self):
        for size in NETWORK_SIZES:
            for shape in SHAPES:
                with self.subTest(nodes=size, shape=shape):
                    picture = _draw(dense_case(size, dc_count=2, shape=shape))
                    boxes = [node["box"] for node in picture.nodes] + list(picture.chips)
                    for first in range(len(picture.chips)):
                        for second in range(len(boxes)):
                            if boxes[second] is picture.chips[first]:
                                continue
                            self.assertFalse(
                                _hits(picture.chips[first], boxes[second]),
                                f"chip {picture.chips[first]} overlaps {boxes[second]}",
                            )

    def test_every_name_stays_inside_its_own_box(self):
        for size in NETWORK_SIZES:
            with self.subTest(nodes=size):
                picture = _draw(dense_case(size, dc_count=2), scope=SCOPE_ALL)
                for classes, _x, _y, body in _NODE_RE.findall(picture.html):
                    rect = _RECT_RE.search(body)
                    if not rect:
                        continue
                    width = float(rect.group(3))
                    for size_text, text in _NAME_RE.findall(body):
                        self.assertLessEqual(
                            wsn._glyph_width(text, float(size_text)), width - 12.0 + 0.01,
                            f"{text} is wider than its box ({classes})",
                        )

    def test_a_long_store_name_is_shortened_rather_than_spilled(self):
        picture = _draw(dense_case(60, dc_count=2), scope=SCOPE_ALL)
        drawn = [name for node in picture.nodes for name in node["names"]]
        self.assertTrue(any(name.endswith("…") for name in drawn))
        # Shortening never goes as far as leaving only an identifier behind.
        for name in drawn:
            self.assertGreater(len(name.rstrip("…")), 1)


class LegibilityTests(unittest.TestCase):
    """E/F · what a person actually reads, at each supported window size."""

    def test_no_label_falls_under_ten_pixels_on_any_supported_desktop(self):
        for size in NETWORK_SIZES:
            picture = _draw(dense_case(size, dc_count=2), scope=SCOPE_ALL)
            smallest = min(picture.fonts)
            for name, column in COLUMN_WIDTH.items():
                with self.subTest(nodes=size, viewport=name):
                    self.assertGreaterEqual(
                        smallest * column / CANVAS_WIDTH, MIN_SCREEN_PX,
                        f"{smallest} units is unreadable in a {column}px column",
                    )

    def test_the_selected_quantity_reads_above_eleven_pixels_everywhere(self):
        self.assertGreaterEqual(wsn.EDGE_FONT * NARROWEST / CANVAS_WIDTH, 11.0)

    def test_the_state_badge_stays_readable_at_every_node_count(self):
        for size in NETWORK_SIZES + (90,):
            _width, height = store_dimensions(size)
            pill_h = max(13.0, min(20.0, height * 0.35))
            pill_font = min(wsn.STATE_PILL_FONT_MAX, pill_h * 0.84)
            with self.subTest(nodes=size):
                self.assertGreaterEqual(
                    pill_font * NARROWEST / CANVAS_WIDTH, MIN_SCREEN_PX,
                )

    def test_a_dc_box_is_always_bigger_than_a_store_box(self):
        """Shape, not colour, is what tells a 물류센터 from a 점포."""
        for size in NETWORK_SIZES + (90,):
            store = store_dimensions(size)
            dc = dc_dimensions(store)
            with self.subTest(nodes=size):
                self.assertGreater(dc[0], store[0])
                self.assertGreater(dc[1], store[1])

    def test_state_is_never_carried_by_colour_alone(self):
        """Dense pictures drop the written badge, so each state keeps its shape."""
        self.assertEqual(sorted(wsn.STATE_MARKERS), sorted(wsn.STATE_STYLES))
        self.assertEqual(len(set(wsn.STATE_MARKERS.values())), len(wsn.STATE_MARKERS))
        picture = _draw(dense_case(60, dc_count=2), scope=SCOPE_ALL)
        for path in wsn.STATE_MARKERS.values():
            if path == wsn.STATE_MARKERS["이동 대상"]:
                continue
            self.assertIn(path, picture.html)


class PerformanceTests(unittest.TestCase):
    """G · a big plan must not stall the screen it is drawn on."""

    #: Generous next to the ~5ms measured here, so a loaded machine cannot make
    #: this flake; it still catches an order-of-magnitude regression.
    BUDGET_MS = 400.0

    def test_layout_and_drawing_stay_fast_at_every_size(self):
        for size in NETWORK_SIZES:
            case = dense_case(size, dc_count=2, move_count=max(1, (size - 2) // 2))
            start = time.perf_counter()
            build_workspace_network(
                case["data"], case["items"], case["selected_route_id"],
                scope=SCOPE_ALL, store_states=case["states"],
            )
            elapsed = (time.perf_counter() - start) * 1000
            with self.subTest(nodes=size):
                self.assertLess(elapsed, self.BUDGET_MS, f"{size} nodes took {elapsed:.0f}ms")

    def test_the_layout_itself_is_cheap_to_recompute(self):
        stores = [f"S{index:03d}" for index in range(58)]
        dcs = ["DC01", "DC02"]
        edges = [(stores[index], dcs[index % 2]) for index in range(29)]
        edges += [(dcs[index % 2], stores[29 + index]) for index in range(29)]
        start = time.perf_counter()
        for _ in range(10):
            compute_flow_layout(stores, dcs, edges, keep=stores)
        self.assertLess((time.perf_counter() - start) * 1000 / 10, 60.0)


class FlowMeaningTests(unittest.TestCase):
    """The geometry has to say what the plan says: 출발 왼쪽, DC 가운데, 도착 오른쪽."""

    def test_donors_sit_left_receivers_right_and_the_dc_between_them(self):
        case = dense_case(24, dc_count=2, shape="via_dc")
        stores = [f"S{index:03d}" for index in range(22)]
        edges: list[tuple[str, str]] = []
        for item in case["items"]:
            edges += [(item["source_id"], item["dc_id"]), (item["dc_id"], item["target_id"])]
        layout = compute_flow_layout(stores, ["DC01", "DC02"], edges, keep=stores)
        sources = {item["source_id"] for item in case["items"]}
        targets = {item["target_id"] for item in case["items"]}
        left = max(layout.positions[node][0] for node in sources)
        right = min(layout.positions[node][0] for node in targets)
        centre = [layout.positions[node][0] for node in ("DC01", "DC02")]
        self.assertLess(left, min(centre))
        self.assertGreater(right, max(centre))
        for node in sources:
            self.assertEqual(layout.bands[node], "source")
        for node in targets:
            self.assertEqual(layout.bands[node], "target")

    def test_a_store_that_both_gives_and_receives_stays_one_node(self):
        stores = ["S001", "S002", "S003"]
        # S002 receives from S001 and sends on to S003.
        layout = compute_flow_layout(stores, [], [("S001", "S002"), ("S002", "S003")])
        self.assertEqual(len(layout.positions), 3)
        self.assertEqual(len(set(layout.positions.values())), 3)

    def test_a_dc_keeps_its_own_band_however_many_there_are(self):
        for dc_count in (1, 2, 3, 4):
            stores = [f"S{index:03d}" for index in range(12)]
            dcs = [f"DC{index + 1:02d}" for index in range(dc_count)]
            edges = [(stores[index], dcs[index % dc_count]) for index in range(6)]
            edges += [(dcs[index % dc_count], stores[6 + index]) for index in range(6)]
            layout = compute_flow_layout(stores, dcs, edges, keep=stores)
            with self.subTest(dcs=dc_count):
                self.assertEqual(
                    {layout.bands[dc] for dc in dcs}, {DC_BAND},
                )
                positions = [layout.positions[dc] for dc in dcs]
                self.assertEqual(len(set(positions)), dc_count)


if __name__ == "__main__":
    unittest.main()
