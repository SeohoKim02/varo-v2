"""Node-edge invariants and read-only presentation regression checks."""
from copy import deepcopy
import math
import xml.etree.ElementTree as ET

import pytest

from simulation.dynamic_network import build_route_segments, compute_dynamic_layout
from simulation.network_geometry import boundary_point, route_legs, point_and_angle, _crosses
from pages.overview import _network_markup_cached


@pytest.mark.parametrize("end", [(300, 0), (-300, 0), (0, 300), (0, -300), (300, 300), (-300, 200)])
@pytest.mark.parametrize("offset", [-54, 0, 54])
def test_boundary_intersection_all_directions(end, offset):
    x, y = boundary_point((0, 0), end, (100, 60), offset)
    assert abs(x) <= 50 and abs(y) <= 30
    assert abs(abs(x)-50) < 1e-7 or abs(abs(y)-30) < 1e-7


def fixtures(count=4):
    nodes = [{"node_id": f"S{i}", "node_name": "아주 긴 점포 이름 " + str(i), "node_type": "STORE",
              "network_mode": "DIRECT_NETWORK"} for i in range(count)]
    nodes.append({"node_id": "DC", "node_name": "물류센터", "node_type": "DC"})
    return nodes


@pytest.mark.parametrize("pairs", [[(0, 1)], [(0, 1), (0, 2), (0, 3)],
                                   [(0, 3), (1, 3), (2, 3)], [(0, 2), (1, 3), (2, 0)]])
@pytest.mark.parametrize("via", [False, True])
def test_direct_via_fan_in_out_crossing_and_shared_nodes(pairs, via):
    nodes = fixtures()
    routes = [dict(source_id=f"S{a}", target_id=f"S{b}", dc_id="DC",
                   route_type="VIA_DC" if via else "DIRECT") for a,b in pairs]
    layout = compute_dynamic_layout(nodes, routes, width=1280, height=800)
    all_nodes = layout.dcs + layout.stores
    pos = {n["node_id"]:(n["x"],n["y"]) for n in all_nodes}
    sizes = {n["node_id"]:(n["width"],n["height"]) for n in all_nodes}
    for idx, route in enumerate(routes):
        segments = build_route_segments(route, nodes, require_explicit_dc=True)
        legs = route_legs(segments, pos, sizes, [-54,0,54][idx])
        assert len(legs) == (2 if via else 1)
        for leg in legs:
            for node_id, point in ((leg["from_node_id"],leg["points"][0]),(leg["to_node_id"],leg["points"][-1])):
                x,y = pos[node_id]; w,h = sizes[node_id]
                dx,dy = abs(point[0]-x),abs(point[1]-y)
                assert dx <= w/2+1e-6 and dy <= h/2+1e-6
                assert abs(dx-w/2)<1e-6 or abs(dy-h/2)<1e-6
            for a,b in zip(leg["points"],leg["points"][1:]):
                for n in all_nodes:
                    assert not _crosses(a,b,(n['x']-n['width']/2,n['y']-n['height']/2,
                                             n['x']+n['width']/2,n['y']+n['height']/2))
            assert all(0<=x<=1280 and 0<=y<=800 for x,y in leg["points"])


@pytest.mark.parametrize("missing", [{}, {"dc_id":"UNKNOWN"}, {"dc_name":"없는 센터"}])
def test_display_never_invents_a_hub(missing):
    route = dict(source_id="S0",target_id="S1",route_type="VIA_DC",**missing)
    with pytest.raises(ValueError):
        build_route_segments(route, fixtures(), require_explicit_dc=True)


def test_move_resize_and_rerun_recompute_docking_without_mutation():
    segments=[dict(from_node_id="A",to_node_id="B")]
    pos={"A":(100,100),"B":(600,100)}; sizes={"A":(100,60),"B":(100,60)}
    before=deepcopy((segments,pos,sizes))
    first=route_legs(segments,pos,sizes)
    assert first == route_legs(segments,pos,sizes)
    assert (segments,pos,sizes)==before
    moved=route_legs(segments,{**pos,"B":(700,200)},sizes)
    resized=route_legs(segments,pos,{**sizes,"B":(200,60)})
    assert first != moved and first != resized
    assert resized[0]['points'][-1][0]==500


@pytest.mark.parametrize("qty", [0,46])
@pytest.mark.parametrize("playing", [False,True])
def test_svg_arrows_vehicles_and_quantity_preserve_payload(qty,playing):
    nodes=fixtures(2)
    routes=[dict(route_id="R",source_id="S0",target_id="S1",route_type="VIA_DC",dc_id="DC",recommended_qty=50)]
    scenario={"transitions":[dict(route_id="R",feasible=qty>0,applied_quantity=qty,requested_quantity=50)]}
    before=deepcopy((nodes,routes,scenario))
    markup=_network_markup_cached(f"geometry-{qty}-{playing}",nodes,routes,[],playing,14,False,"R",scenario)['html']
    svg=ET.fromstring(markup[markup.index('<svg'):markup.index('</svg>')+6])
    ns={'s':'http://www.w3.org/2000/svg'}
    paths=svg.findall('.//s:path[@data-route]',ns)
    assert len(paths)==2 and all(p.get('marker-end') for p in paths)
    assert [(p.get('data-from'),p.get('data-to')) for p in paths]==[('S0','DC'),('DC','S1')]
    motion=svg.findall('.//s:animateMotion',ns)
    assert len(motion)==(1 if qty and playing else 0)
    assert markup.count('class="v2-vehicle ')==1
    assert all(m.get('rotate')=='auto' for m in motion)
    if qty and playing:
        visibility=svg.findall('.//s:animate[@attributeName="opacity"]',ns)
        assert [a.get('values') for a in visibility]==['1;0;1;1']
        assert motion[0].get('path').count('M ')==2
        assert motion[0].get('keyTimes')=='0;0.43;0.57;1'
    if not qty:
        assert 'v2-vehicle-disabled' in markup and '이동 없음' in markup
    assert (nodes,routes,scenario)==before


def test_missing_connection_is_visible_not_silently_omitted():
    nodes=fixtures(2)
    routes=[dict(route_id='R',source_id='S0',target_id='S1',route_type='VIA_DC')]
    markup=_network_markup_cached('missing-hub',nodes,routes,[],False,14,False,'R')['html']
    assert '경로 정보 확인 필요' in markup and 'data-route=' not in markup


def test_stationary_vehicle_uses_same_path_and_direction():
    points=[(0,0),(100,0),(100,100)]
    assert point_and_angle(points,.75)==(100,50,90)
    assert math.isclose(point_and_angle([(100,0),(0,0)],.5)[2],180)


def test_inventory_46_of_50_is_unchanged_by_network_rendering():
    from services.inventory_transition_service import run_inventory_scenario
    import pandas as pd
    data={"inventory":pd.DataFrame([
        dict(store_id="S0",product_id="P",stock_qty=149,avg_daily_sales=10),
        dict(store_id="S1",product_id="P",stock_qty=136,avg_daily_sales=26)])}
    routes=[dict(route_id="R",source_id="S0",target_id="S1",product_id="P",recommended_qty=50,route_type="DIRECT")]
    scenario=run_inventory_scenario(data,routes)
    before=deepcopy(scenario)
    result=_network_markup_cached('inventory-46',fixtures(2),routes,[],False,14,False,'R',scenario)
    assert result['ok']
    assert scenario==before==run_inventory_scenario(data,routes)
    t=scenario['transitions'][0]
    assert t['requested_quantity']==50 and t['applied_quantity']==46
    assert (t['source']['before_stock'],t['source']['after_stock'])==(149,103)
    assert (t['target']['before_stock'],t['target']['after_stock'])==(136,182)
