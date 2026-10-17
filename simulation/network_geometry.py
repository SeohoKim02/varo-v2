"""Presentation-only, deterministic node-boundary routing in SVG coordinates."""
from __future__ import annotations

import heapq
import math


def boundary_point(center, toward, size, offset=0.0):
    """Intersect the lane ray with the actual card, never translate a clipped end."""
    dx, dy = toward[0] - center[0], toward[1] - center[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        raise ValueError("경로 정보 확인 필요")
    half_w, half_h = size[0] / 2, size[1] / 2
    offset = max(-min(half_w, half_h) * .65, min(min(half_w, half_h) * .65, offset))
    ox, oy = -dy / length * offset, dx / length * offset
    tx = ((half_w if dx > 0 else -half_w) - ox) / dx if dx else math.inf
    ty = ((half_h if dy > 0 else -half_h) - oy) / dy if dy else math.inf
    t = min(tx, ty)
    x, y = ox + dx * t, oy + dy * t
    # Avoid the rounded corner rather than docking on its invisible square corner.
    if tx <= ty:
        y = max(-half_h + 12, min(half_h - 12, y))
    else:
        x = max(-half_w + 12, min(half_w - 12, x))
    return center[0] + x, center[1] + y


def _crosses(a, b, rect):
    """True only when the open segment enters the rectangle interior."""
    lo, hi = 0.0, 1.0
    for p, q, low, high in ((a[0], b[0], rect[0], rect[2]), (a[1], b[1], rect[1], rect[3])):
        low, high = low + 1e-6, high - 1e-6
        if abs(q - p) < 1e-9:
            if not low < p < high:
                return False
        else:
            t1, t2 = sorted(((low - p) / (q - p), (high - p) / (q - p)))
            lo, hi = max(lo, t1), min(hi, t2)
            if lo >= hi:
                return False
    return hi > 0 and lo < 1


def _clear_path(start, end, obstacles):
    if not any(_crosses(start, end, r) for r in obstacles):
        return [start, end]
    vertices = [start, end]
    for left, top, right, bottom in obstacles:
        vertices.extend(((left, top), (right, top), (right, bottom), (left, bottom)))
    costs, previous, queue = {0: 0.0}, {}, [(0.0, 0)]
    while queue:
        cost, i = heapq.heappop(queue)
        if cost != costs[i]:
            continue
        if i == 1:
            path = [end]
            while i:
                i = previous[i]
                path.append(vertices[i])
            return path[::-1]
        for j, v in enumerate(vertices):
            if j == i or any(_crosses(vertices[i], v, r) for r in obstacles):
                continue
            candidate = cost + math.dist(vertices[i], v)
            if candidate < costs.get(j, math.inf):
                costs[j], previous[j] = candidate, i
                heapq.heappush(queue, (candidate, j))
    raise ValueError("경로 정보 확인 필요")


def route_legs(segments, positions, dimensions, lane_offset=0.0):
    """One visible boundary-to-boundary leg per data-backed segment.

    Intermediate hubs deliberately have separate incoming/outgoing paths: no
    invisible path crosses the hub card. No inventory or route data is changed.
    """
    legs = []
    for segment in segments:
        source, target = str(segment["from_node_id"]), str(segment["to_node_id"])
        if source == target or source not in positions or target not in positions:
            raise ValueError("경로 정보 확인 필요")
        a, b = positions[source], positions[target]
        # Use a canonical lane sign so opposite-direction routes separate too.
        offset = lane_offset if source < target else -lane_offset
        start = boundary_point(a, b, dimensions[source], offset)
        end = boundary_point(b, a, dimensions[target], -offset)
        obstacles = []
        for node, (x, y) in positions.items():
            w, h = dimensions[node]
            pad = 0 if node in (source, target) else 18
            obstacles.append((x-w/2-pad, y-h/2-pad, x+w/2+pad, y+h/2+pad))
        points = _clear_path(start, end, obstacles)
        path = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in points)
        legs.append({**segment, "points": points, "path": path})
    return legs


def point_and_angle(points, fraction):
    lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
    distance = max(0., min(1., fraction)) * sum(lengths)
    for i, (a, b, length) in enumerate(zip(points, points[1:], lengths)):
        if distance <= length or i == len(lengths)-1:
            ratio = distance / length if length else 0
            return (a[0]+(b[0]-a[0])*ratio, a[1]+(b[1]-a[1])*ratio,
                    math.degrees(math.atan2(b[1]-a[1], b[0]-a[0])))
        distance -= length
    return (*points[0], 0.)
