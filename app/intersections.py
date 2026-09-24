"""闭合轮廓交叉检查（intersections，只读即时计算）。

对一份已存 SVG path 即时解析并完成轮廓包含层级分类后（与 metrics /
clearance / scanlines / cutplan / tabs 同一份解析与曲线细分折线，见
:func:`app.svggeo.analyze_contours`），在**加工前**检查闭合轮廓细分
折线的边相交情况并输出交叉清单：

- **自交**：同一闭合子路径内所有**非相邻**边对逐一检查
  （相邻边仅共享端点属于正常，不参与检查；含首尾闭合边与首边
  这一对相邻边）；
- **轮廓间相交**：不同闭合子路径的所有边对逐一检查；
- 相交类型：
    - ``cross``   交叉：单点相交且交点严格位于两条边的内部；
    - ``touch``   端点接触：单点相交且交点至少是一条边的端点；
    - ``overlap`` 共线重叠：两边共线且重叠区间长度为正，
      给出重叠区间的两个端点；
- **合并**：坐标相同（量化到 1e-9）且类型相同的交点合并为一条，
  其涉及的 ``(contour, edge)`` 边索引取并集；
- **排序**：结果按涉及边的 ``(contour, edge)`` 索引升序、再按类型
  稳定排列，多次调用结果一致。

参与规则：仅闭合且非退化轮廓（``outer`` / ``hole``）参与；开放轮廓
（未以 ``Z`` 闭合）与退化轮廓（面积为 0 或顶点不足）跳过并列入
``skipped_contours``；空路径返回空清单。

整个计算只读、即时完成：不依赖已保存的规范化结果，也不写库。
"""

from __future__ import annotations

import math
from typing import Any

from .svggeo import Contour, point_segment_dist2

# 相交判定的相对容差：吸收细分折线与求交运算的浮点表示误差
_EPS_REL = 1e-9
# 平行判定的角度容差（|sin θ| 阈值）
_EPS_PARALLEL = 1e-12
# 交点合并键与输出坐标的量化位数
_DECIMALS = 9

# 边引用：(contour_original_index, edge_index)
EdgeRef = tuple[int, int]
# 细分线段：两个端点
Segment = tuple[tuple[float, float], tuple[float, float]]


def compute_intersections(contours: list[Contour]) -> dict[str, Any]:
    """对已分类的轮廓列表检查闭合轮廓细分折线的边相交。

    参数:
        contours: :func:`app.svggeo.analyze_contours` 返回的轮廓
            （已带 ``role`` / ``closed`` / ``degenerate``）。

    返回:
        ``{checked_contour_count, skipped_contours,
        intersection_count, intersections}``；``intersections`` 每项含
        ``type``（``cross`` / ``touch`` / ``overlap``）、交点坐标
        （``point``，重叠为 ``start`` / ``end`` 两端点）与涉及的
        ``edges``（``{contour, edge}`` 列表，按索引升序）。
    """
    checked = [c for c in contours if c.role in ("outer", "hole")]
    checked.sort(key=lambda c: c.original_index)

    skipped = [
        {"original_index": c.original_index, "role": c.role,
         "reason": ("open" if c.role == "open" else "degenerate")}
        for c in sorted(contours, key=lambda c: c.original_index)
        if c.role not in ("outer", "hole")
    ]

    # 每条闭合轮廓的边表：边 k 为 顶点k -> 顶点k+1（末边回首点）
    edge_map: dict[int, list[Segment]] = {c.original_index: _edges(c)
                                          for c in checked}

    merged: dict[tuple, dict[str, Any]] = {}

    # ---- 自交：同一轮廓的非相邻边对 --------------------------------------
    for c in checked:
        idx = c.original_index
        segs = edge_map[idx]
        n = len(segs)
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue  # 末边与首边相邻（共享起点），视为正常
                _check_pair(segs[i], (idx, i), segs[j], (idx, j), merged)

    # ---- 轮廓间相交：不同闭合轮廓的所有边对 ------------------------------
    for x, cx in enumerate(checked):
        for cy in checked[x + 1:]:
            # 包围盒快速剔除：不相交（含容差）则边对不可能相交
            if not _bboxes_overlap(cx.bounds, cy.bounds):
                continue
            ix, iy = cx.original_index, cy.original_index
            for i, si in enumerate(edge_map[ix]):
                for j, sj in enumerate(edge_map[iy]):
                    _check_pair(si, (ix, i), sj, (iy, j), merged)

    items = [_to_item(entry) for entry in merged.values()]
    items.sort(key=lambda it: (tuple((e["contour"], e["edge"])
                                     for e in it["edges"]), it["type"]))
    return {
        "checked_contour_count": len(checked),
        "skipped_contours": skipped,
        "intersection_count": len(items),
        "intersections": items,
    }


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _edges(c: Contour) -> list[Segment]:
    """闭合轮廓的边序列：边 k 为 顶点k -> 顶点(k+1)%n（含闭合边）。"""
    pts = c.points
    n = len(pts)
    return [(pts[k], pts[(k + 1) % n]) for k in range(n)]


def _check_pair(s1: Segment, ref1: EdgeRef,
                s2: Segment, ref2: EdgeRef,
                merged: dict[tuple, dict[str, Any]]) -> None:
    """检查一对细分线段，相交则按类型并入合并表。"""
    hit = _classify(s1[0], s1[1], s2[0], s2[1])
    if hit is None:
        return
    kind, p1, p2 = hit
    if kind == "overlap":
        assert p2 is not None
        # 端点按字典序规范化：同一重叠区间与边的方向无关
        if (p2[0], p2[1]) < (p1[0], p1[1]):
            p1, p2 = p2, p1
        key = (kind, _q(p1[0]), _q(p1[1]), _q(p2[0]), _q(p2[1]))
    else:
        key = (kind, _q(p1[0]), _q(p1[1]))
    entry = merged.get(key)
    if entry is None:
        entry = {"type": kind, "p1": p1, "p2": p2, "refs": set()}
        merged[key] = entry
    entry["refs"].add(ref1)
    entry["refs"].add(ref2)


def _classify(a: tuple[float, float], b: tuple[float, float],
              c: tuple[float, float], d: tuple[float, float]
              ) -> tuple[str, tuple[float, float],
                         tuple[float, float] | None] | None:
    """细分线段 ``ab`` 与 ``cd`` 的相交分类。

    返回 ``None``（不相交）或 ``(kind, p1, p2)``：
    ``cross`` / ``touch`` 时 ``p1`` 为交点、``p2`` 为 ``None``；
    ``overlap`` 时 ``p1`` / ``p2`` 为重叠区间两端（沿 ``ab`` 方向有序）。
    """
    rx, ry = b[0] - a[0], b[1] - a[1]
    sx, sy = d[0] - c[0], d[1] - c[1]
    len_r = math.hypot(rx, ry)
    len_s = math.hypot(sx, sy)
    if len_r == 0.0 or len_s == 0.0:
        return _classify_degenerate(a, b, c, d, len_r, len_s)
    eps = _EPS_REL * max(1.0, len_r, len_s)

    denom = rx * sy - ry * sx
    if abs(denom) <= _EPS_PARALLEL * len_r * len_s:
        # 平行：仅共线（c 到直线 ab 的距离在容差内）时才可能相交
        cross = (c[0] - a[0]) * ry - (c[1] - a[1]) * rx
        if abs(cross) > eps * len_r:
            return None
        return _classify_collinear(a, rx, ry, c, d, len_r, eps)

    t = ((c[0] - a[0]) * sy - (c[1] - a[1]) * sx) / denom
    u = ((c[0] - a[0]) * ry - (c[1] - a[1]) * rx) / denom
    t_eps = eps / len_r
    u_eps = eps / len_s
    if not (-t_eps <= t <= 1.0 + t_eps and -u_eps <= u <= 1.0 + u_eps):
        return None
    tc = min(1.0, max(0.0, t))
    p = (a[0] + tc * rx, a[1] + tc * ry)
    if t <= t_eps or t >= 1.0 - t_eps or u <= u_eps or u >= 1.0 - u_eps:
        return ("touch", p, None)
    return ("cross", p, None)


def _classify_collinear(a: tuple[float, float], rx: float, ry: float,
                        c: tuple[float, float], d: tuple[float, float],
                        len_r: float, eps: float
                        ) -> tuple[str, tuple[float, float],
                                   tuple[float, float] | None] | None:
    """共线线段的重叠区间（投影到 ``ab`` 的主导轴参数化）。"""
    k = 0 if abs(rx) >= abs(ry) else 1
    dir_k = rx if k == 0 else ry
    t_c = (c[k] - a[k]) / dir_k
    t_d = (d[k] - a[k]) / dir_k
    lo = max(0.0, min(t_c, t_d))
    hi = min(1.0, max(t_c, t_d))
    t_eps = eps / len_r
    if lo > hi + t_eps:
        return None
    if hi - lo <= t_eps:
        # 单点接触：共线时接触点必为至少一条边的端点
        tm = min(1.0, max(0.0, (lo + hi) / 2.0))
        return ("touch", (a[0] + tm * rx, a[1] + tm * ry), None)
    return ("overlap",
            (a[0] + lo * rx, a[1] + lo * ry),
            (a[0] + hi * rx, a[1] + hi * ry))


def _classify_degenerate(a: tuple[float, float], b: tuple[float, float],
                         c: tuple[float, float], d: tuple[float, float],
                         len_r: float, len_s: float
                         ) -> tuple[str, tuple[float, float],
                                    tuple[float, float] | None] | None:
    """零长度边按点处理（dedupe 后理论上不出现，防御性处理）。"""
    if len_r == 0.0 and len_s == 0.0:
        return ("touch", a, None) if a == c else None
    if len_r == 0.0:
        eps = _EPS_REL * max(1.0, len_s)
        on = point_segment_dist2(a, c, d) <= eps * eps
        return ("touch", a, None) if on else None
    eps = _EPS_REL * max(1.0, len_r)
    on = point_segment_dist2(c, a, b) <= eps * eps
    return ("touch", c, None) if on else None


def _bboxes_overlap(b1: tuple[float, float, float, float],
                    b2: tuple[float, float, float, float]) -> bool:
    """两包围盒是否相交（含边界接触，带相对容差）。"""
    eps = _EPS_REL * max(1.0, b1[2] - b1[0], b1[3] - b1[1],
                         b2[2] - b2[0], b2[3] - b2[1])
    return (b1[0] <= b2[2] + eps and b2[0] <= b1[2] + eps
            and b1[1] <= b2[3] + eps and b2[1] <= b1[3] + eps)


def _to_item(entry: dict[str, Any]) -> dict[str, Any]:
    """合并表条目 -> 输出项（边索引升序、坐标量化展示）。"""
    edges = [{"contour": ci, "edge": ei}
             for ci, ei in sorted(entry["refs"])]
    item: dict[str, Any] = {"type": entry["type"]}
    if entry["type"] == "overlap":
        p1, p2 = entry["p1"], entry["p2"]
        item["start"] = {"x": _num(p1[0]), "y": _num(p1[1])}
        item["end"] = {"x": _num(p2[0]), "y": _num(p2[1])}
    else:
        p1 = entry["p1"]
        item["point"] = {"x": _num(p1[0]), "y": _num(p1[1])}
    item["edges"] = edges
    return item


def _q(v: float) -> float:
    """交点合并键的坐标量化（吸收求交浮点噪声）。"""
    return round(v, _DECIMALS)


def _num(v: float) -> float:
    """量化展示值并消除 -0.0。"""
    r = round(v, _DECIMALS)
    return r + 0.0 if r == 0 else r
