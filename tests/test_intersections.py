"""闭合轮廓交叉检查的几何单元测试。"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.intersections import compute_intersections  # noqa: E402
from app.svggeo import analyze_contours  # noqa: E402


def _check(d, fill_rule="nonzero"):
    contours, _ = analyze_contours(d, fill_rule)
    return compute_intersections(contours)


# ---------------------------------------------------------------------------
# 自交（同一闭合子路径的非相邻边）
# ---------------------------------------------------------------------------

def test_simple_square_has_no_intersections():
    res = _check("M0,0h10v10h-10Z")
    assert res["checked_contour_count"] == 1
    assert res["skipped_contours"] == []
    assert res["intersection_count"] == 0
    assert res["intersections"] == []


def test_triangle_has_no_non_adjacent_pairs():
    # 三角形所有边两两相邻，不可能自交
    assert _check("M0,0 L10,0 L5,10 Z")["intersections"] == []


def test_self_crossing_single_point():
    # 五边形 e0 (0,0)-(10,10) 与 e2 (10,0)-(0,10) 在 (5,5) 内-内交叉
    res = _check("M0,0 L10,10 L10,0 L0,10 L-5,5 Z")
    assert res["intersection_count"] == 1
    item = res["intersections"][0]
    assert item["type"] == "cross"
    assert item["point"] == {"x": 5.0, "y": 5.0}
    assert item["edges"] == [{"contour": 0, "edge": 0},
                             {"contour": 0, "edge": 2}]


def test_self_touch_vertex_on_edge_merges_edge_indices():
    # 顶点 (5,0) 落在 e0 内部：e0×e2 与 e0×e3 两对同坐标同类型 -> 合并
    res = _check("M0,0 L10,0 L10,10 L5,0 L0,10 Z")
    assert res["intersection_count"] == 1
    item = res["intersections"][0]
    assert item["type"] == "touch"
    assert item["point"] == {"x": 5.0, "y": 0.0}
    assert item["edges"] == [{"contour": 0, "edge": 0},
                             {"contour": 0, "edge": 2},
                             {"contour": 0, "edge": 3}]


def test_adjacent_wrap_around_pair_not_reported():
    # 末边与首边相邻共享起点：即使共线往返（尖刺）也不按相邻对报告；
    # 这里验证普通闭合环的首尾相邻边共享端点不产生任何条目
    res = _check("M0,0 L10,0 L10,10 L0,10 Z")
    assert res["intersections"] == []


# ---------------------------------------------------------------------------
# 轮廓间相交
# ---------------------------------------------------------------------------

def test_cross_contour_crossings_sorted_by_edge_index():
    # 两个错位正方形：e1×f0 交于 (10,5)，e2×f3 交于 (5,10)
    res = _check("M0,0h10v10h-10Z M5,5h10v10h-10Z")
    assert res["checked_contour_count"] == 2
    assert res["intersection_count"] == 2
    first, second = res["intersections"]
    assert first["type"] == "cross"
    assert first["point"] == {"x": 10.0, "y": 5.0}
    assert first["edges"] == [{"contour": 0, "edge": 1},
                              {"contour": 1, "edge": 0}]
    assert second["type"] == "cross"
    assert second["point"] == {"x": 5.0, "y": 10.0}
    assert second["edges"] == [{"contour": 0, "edge": 2},
                               {"contour": 1, "edge": 3}]
    # 按涉及边索引稳定升序
    keys = [tuple((e["contour"], e["edge"]) for e in it["edges"])
            for it in res["intersections"]]
    assert keys == sorted(keys)


def test_corner_touch_between_contours_merges_all_pairs():
    # 两正方形仅共享角点 (10,10)：4 个边对同坐标同类型 -> 合并为一条
    res = _check("M0,0h10v10h-10Z M10,10h10v10h-10Z")
    assert res["intersection_count"] == 1
    item = res["intersections"][0]
    assert item["type"] == "touch"
    assert item["point"] == {"x": 10.0, "y": 10.0}
    assert item["edges"] == [{"contour": 0, "edge": 1},
                             {"contour": 0, "edge": 2},
                             {"contour": 1, "edge": 0},
                             {"contour": 1, "edge": 3}]


def test_collinear_overlap_reports_both_endpoints():
    # 两矩形在 y=0 上共线重叠 (5,0)-(10,0)；重叠端点同时引发两条端点接触
    res = _check("M0,0 L10,0 L10,5 L0,5 Z M5,0 L15,0 L15,-5 L5,-5 Z")
    assert res["intersection_count"] == 3
    overlap, touch_lo, touch_hi = res["intersections"]
    assert overlap["type"] == "overlap"
    assert overlap["start"] == {"x": 5.0, "y": 0.0}
    assert overlap["end"] == {"x": 10.0, "y": 0.0}
    assert overlap["edges"] == [{"contour": 0, "edge": 0},
                                {"contour": 1, "edge": 0}]
    assert touch_lo["type"] == "touch"
    assert touch_lo["point"] == {"x": 5.0, "y": 0.0}
    assert touch_lo["edges"] == [{"contour": 0, "edge": 0},
                                 {"contour": 1, "edge": 3}]
    assert touch_hi["type"] == "touch"
    assert touch_hi["point"] == {"x": 10.0, "y": 0.0}
    assert touch_hi["edges"] == [{"contour": 0, "edge": 1},
                                 {"contour": 1, "edge": 0}]


def test_identical_contours_overlap_every_edge():
    # 两条全等正方形：4 条边各自共线重叠，4 个角点各自端点接触
    d = "M0,0h10v10h-10Z M0,0h10v10h-10Z"
    res = _check(d)
    assert res["intersection_count"] == 8
    overlaps = [it for it in res["intersections"] if it["type"] == "overlap"]
    touches = [it for it in res["intersections"] if it["type"] == "touch"]
    assert len(overlaps) == 4
    assert len(touches) == 4
    first = overlaps[0]
    assert first["start"] == {"x": 0.0, "y": 0.0}
    assert first["end"] == {"x": 10.0, "y": 0.0}
    assert first["edges"] == [{"contour": 0, "edge": 0},
                              {"contour": 1, "edge": 0}]
    # 每个角点接触合并了两条轮廓的各两条邻边
    corner = next(it for it in touches if it["point"] == {"x": 0.0, "y": 0.0})
    assert corner["edges"] == [{"contour": 0, "edge": 0},
                               {"contour": 0, "edge": 3},
                               {"contour": 1, "edge": 0},
                               {"contour": 1, "edge": 3}]


def test_separate_contours_no_intersections():
    res = _check("M0,0h10v10h-10Z M30,0h10v10h-10Z")
    assert res["intersection_count"] == 0
    assert res["checked_contour_count"] == 2


# ---------------------------------------------------------------------------
# 参与规则 / 细分 / 确定性
# ---------------------------------------------------------------------------

def test_open_and_degenerate_contours_skipped():
    d = ("M0,0h40v40h-40Z "
         "M5,5 L10,5 L5,5 Z "          # 退化（面积 0）
         "M20,20 L25,20 L25,25")        # 开放折线
    res = _check(d)
    assert res["checked_contour_count"] == 1
    assert res["intersections"] == []
    assert res["skipped_contours"] == [
        {"original_index": 1, "role": "degenerate", "reason": "degenerate"},
        {"original_index": 2, "role": "open", "reason": "open"}]


def test_empty_path_returns_empty_list():
    res = _check("")
    assert res["checked_contour_count"] == 0
    assert res["skipped_contours"] == []
    assert res["intersection_count"] == 0
    assert res["intersections"] == []


def test_curves_flattened_before_check():
    # 圆弧拼成的整圆（圆心 (5,5) 半径 5）与竖直长条 x∈[4,6]：
    # 长条左右两条边各与圆的细分折线交叉两次
    d = ("M0,5 A5,5 0 1,1 10,5 A5,5 0 1,1 0,5 Z "
         "M4,-10 L6,-10 L6,20 L4,20 Z")
    res = _check(d)
    assert res["intersection_count"] == 4
    for item in res["intersections"]:
        assert item["type"] == "cross"
        contours = {e["contour"] for e in item["edges"]}
        assert contours == {0, 1}
        assert -5.0 < item["point"]["y"] < 15.0


def test_both_fill_rules_agree():
    d = "M0,0 L10,10 L10,0 L0,10 L-5,5 Z"
    assert _check(d, "nonzero") == _check(d, "evenodd")


def test_results_deterministic():
    d = ("M0,0 L10,10 L10,0 L0,10 L-5,5 Z "
         "M20,0h10v10h-10Z M25,5h10v10h-10Z")
    outs = []
    for _ in range(3):
        contours, _ = analyze_contours(d, "nonzero")
        outs.append(compute_intersections(contours))
    assert outs[0] == outs[1] == outs[2]
