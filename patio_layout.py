#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

GRID_MM = 10
LONG_LINE_THRESHOLD_MM = 1200
MAX_ALLOWED_LONG_LINE_MM = 1800


@dataclass(frozen=True)
class SlabType:
    name: str
    w_mm: int
    h_mm: int
    count_available: int

    def orientations(self) -> List[Tuple[int, int, bool]]:
        if self.w_mm == self.h_mm:
            return [(self.w_mm, self.h_mm, False)]
        return [(self.w_mm, self.h_mm, False), (self.h_mm, self.w_mm, True)]


@dataclass(frozen=True)
class Placement:
    slab: str
    x_mm: int
    y_mm: int
    w_mm: int
    h_mm: int
    rotated: bool

    @property
    def right(self) -> int:
        return self.x_mm + self.w_mm

    @property
    def bottom(self) -> int:
        return self.y_mm + self.h_mm

    @property
    def area(self) -> int:
        return self.w_mm * self.h_mm


@dataclass
class ScoreBreakdown:
    score_total: float
    coverage_ratio: float
    coverage_score: float
    uncovered_penalty: float
    long_line_penalty: float
    cross_penalty: float
    t_junction_penalty: float
    variety_penalty: float
    max_long_line_mm: int
    cross_count: int
    t_junction_count: int
    seams_over_threshold: int


@dataclass
class Layout:
    placements: List[Placement] = field(default_factory=list)
    used_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass(frozen=True)
class Patio:
    width_mm: int
    depth_mm: int


@dataclass(frozen=True)
class InnerRect:
    x0_mm: int
    y0_mm: int
    width_mm: int
    depth_mm: int


@dataclass
class FillStats:
    attempted_placements: int = 0
    frontier_failures: int = 0
    blocked_cells: int = 0
    end_reason: str = "unknown"


@dataclass(frozen=True)
class PlacementPolicy:
    strategy: str
    slab_weights: Dict[str, float]
    min_remaining_ratio_for_small: float
    late_stage_small_bonus: float
    small_slab_names: Set[str]
    small_area_threshold: int


@dataclass(frozen=True)
class AntiPatternConfig:
    long_line_threshold_mm: int = LONG_LINE_THRESHOLD_MM
    max_allowed_long_line_mm: int = MAX_ALLOWED_LONG_LINE_MM
    cross_reject: bool = True
    cross_reject_free_area_below: float = 0.08
    long_line_reject: bool = True
    long_line_reject_free_area_below: float = 0.10
    cross_penalty: float = 5000.0
    t_junction_penalty: float = 150.0
    long_line_penalty_factor: float = 1.0
    long_line_weight: float = 40.0
    uncovered_proxy_weight: float = 120.0
    contact_weight: float = 6.0
    stagger_bonus_weight: float = 50.0


class OccupancyGrid:
    def __init__(self, width_mm: int, depth_mm: int, grid_mm: int = GRID_MM):
        self.grid_mm = grid_mm
        self.cols = width_mm // grid_mm
        self.rows = depth_mm // grid_mm
        self.bits = [0] * self.rows
        self.row_mask_full = (1 << self.cols) - 1

    def _mask(self, c0: int, c1: int) -> int:
        width = c1 - c0
        return ((1 << width) - 1) << c0

    def is_free(self, x_mm: int, y_mm: int, w_mm: int, h_mm: int) -> bool:
        c0 = x_mm // self.grid_mm
        c1 = c0 + (w_mm // self.grid_mm)
        r0 = y_mm // self.grid_mm
        r1 = r0 + (h_mm // self.grid_mm)
        if c0 < 0 or r0 < 0 or c1 > self.cols or r1 > self.rows:
            return False
        mask = self._mask(c0, c1)
        for r in range(r0, r1):
            if self.bits[r] & mask:
                return False
        return True

    def set_rect(self, x_mm: int, y_mm: int, w_mm: int, h_mm: int, value: bool) -> None:
        c0 = x_mm // self.grid_mm
        c1 = c0 + (w_mm // self.grid_mm)
        r0 = y_mm // self.grid_mm
        r1 = r0 + (h_mm // self.grid_mm)
        mask = self._mask(c0, c1)
        for r in range(r0, r1):
            if value:
                self.bits[r] |= mask
            else:
                self.bits[r] &= ~mask

    def first_empty(self) -> Optional[Tuple[int, int]]:
        for r in range(self.rows):
            row_bits = self.bits[r]
            if row_bits != self.row_mask_full:
                free = (~row_bits) & self.row_mask_full
                lsb = free & -free
                c = lsb.bit_length() - 1
                return c * self.grid_mm, r * self.grid_mm
        return None


def mm_to_grid(mm: int) -> int:
    return (mm // GRID_MM) * GRID_MM


def clamp_to_grid(mm: int) -> int:
    return (mm // GRID_MM) * GRID_MM


def validate_dim(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return clamp_to_grid(value)


def long_line_penalty(placements: Sequence[Placement], patio_w: int, patio_h: int, threshold: int) -> Tuple[float, int, int]:
    vertical_by_x: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    horizontal_by_y: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    for p in placements:
        vertical_by_x[p.x_mm].append((p.y_mm, p.bottom))
        vertical_by_x[p.right].append((p.y_mm, p.bottom))
        horizontal_by_y[p.y_mm].append((p.x_mm, p.right))
        horizontal_by_y[p.bottom].append((p.x_mm, p.right))

    def merge(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not intervals:
            return []
        intervals = sorted(intervals)
        out = [list(intervals[0])]  # type: ignore[var-annotated]
        for a, b in intervals[1:]:
            cur = out[-1]
            if a <= cur[1]:
                cur[1] = max(cur[1], b)
            else:
                out.append([a, b])
        return [(a, b) for a, b in out]

    max_len = 0
    penalty = 0.0
    over_count = 0

    for x, spans in vertical_by_x.items():
        if x <= 0 or x >= patio_w:
            continue
        for a, b in merge(spans):
            length = b - a
            max_len = max(max_len, length)
            if length > threshold:
                over = length - threshold
                penalty += ((over / 100.0) ** 2)
                over_count += 1

    for y, spans in horizontal_by_y.items():
        if y <= 0 or y >= patio_h:
            continue
        for a, b in merge(spans):
            length = b - a
            max_len = max(max_len, length)
            if length > threshold:
                over = length - threshold
                penalty += ((over / 100.0) ** 2)
                over_count += 1

    return penalty, max_len, over_count


def corner_and_t_penalties(placements: Sequence[Placement], patio_w: int, patio_h: int) -> Tuple[int, int]:
    corners: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    edges_v: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    edges_h: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)

    for idx, p in enumerate(placements):
        for pt in [(p.x_mm, p.y_mm), (p.right, p.y_mm), (p.x_mm, p.bottom), (p.right, p.bottom)]:
            corners[pt].add(idx)
        edges_v[p.x_mm].append((p.y_mm, p.bottom, idx))
        edges_v[p.right].append((p.y_mm, p.bottom, idx))
        edges_h[p.y_mm].append((p.x_mm, p.right, idx))
        edges_h[p.bottom].append((p.x_mm, p.right, idx))

    cross = 0
    t_count = 0

    for (x, y), idxs in corners.items():
        if x <= 0 or x >= patio_w or y <= 0 or y >= patio_h:
            continue
        if len(idxs) >= 4:
            cross += 1
            continue

        is_t = False
        if len(idxs) >= 2:
            for a, b, sid in edges_v.get(x, []):
                if sid in idxs:
                    continue
                if a < y < b:
                    is_t = True
                    break
            if not is_t:
                for a, b, sid in edges_h.get(y, []):
                    if sid in idxs:
                        continue
                    if a < x < b:
                        is_t = True
                        break
        if is_t:
            t_count += 1

    return cross, t_count


@dataclass
class CandidateEval:
    placement: Placement
    placement_score: float
    delta_long_line: float
    delta_cross: int
    delta_t_junction: int
    stagger_bonus: float
    creates_cross: bool
    max_line_after_mm: int


class PatternState:
    def __init__(self, patio_w: int, patio_h: int, anti: AntiPatternConfig):
        self.patio_w = patio_w
        self.patio_h = patio_h
        self.anti = anti
        self.next_slab_id = 0
        self.corner_map: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        self.edges_v: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
        self.edges_h: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
        self.seams_v: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.seams_h: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    def _is_interior(self, x: int, y: int) -> bool:
        return 0 < x < self.patio_w and 0 < y < self.patio_h

    def _line_penalty(self, length: int) -> float:
        if length <= self.anti.long_line_threshold_mm:
            return 0.0
        over = length - self.anti.long_line_threshold_mm
        return self.anti.long_line_penalty_factor * ((over / 100.0) ** 2)

    def _line_total_penalty(self, intervals: Sequence[Tuple[int, int]]) -> float:
        return sum(self._line_penalty(b - a) for a, b in intervals)

    def _preview_add_interval(self, intervals: Sequence[Tuple[int, int]], a: int, b: int) -> List[Tuple[int, int]]:
        if not intervals:
            return [(a, b)]
        starts = [x for x, _ in intervals]
        i = bisect.bisect_left(starts, a)
        merged: List[Tuple[int, int]] = list(intervals)
        left = i
        if left > 0 and merged[left - 1][1] >= a:
            left -= 1
        na, nb = a, b
        right = left
        while right < len(merged) and merged[right][0] <= nb:
            na = min(na, merged[right][0])
            nb = max(nb, merged[right][1])
            right += 1
        return merged[:left] + [(na, nb)] + merged[right:]

    def _max_interval_len(self, intervals: Sequence[Tuple[int, int]]) -> int:
        if not intervals:
            return 0
        return max(b - a for a, b in intervals)

    def _overlap_len(self, intervals: Sequence[Tuple[int, int]], a: int, b: int) -> int:
        overlap = 0
        for ia, ib in intervals:
            if ib <= a:
                continue
            if ia >= b:
                break
            overlap += max(0, min(ib, b) - max(ia, a))
        return overlap

    def evaluate_candidate(self, candidate: Placement, occ: OccupancyGrid, inner: InnerRect, filled_area: int) -> CandidateEval:
        x0 = candidate.x_mm - inner.x0_mm
        y0 = candidate.y_mm - inner.y0_mm
        x1 = x0 + candidate.w_mm
        y1 = y0 + candidate.h_mm

        delta_cross = 0
        delta_t = 0
        corner_points = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
        for x, y in corner_points:
            existing = self.corner_map.get((x, y), set())
            new_count = len(existing) + 1
            if self._is_interior(x, y) and new_count >= 4:
                delta_cross += 1
                continue
            if self._is_interior(x, y) and new_count >= 2:
                has_t = False
                for a, b, sid in self.edges_v.get(x, []):
                    if sid in existing:
                        continue
                    if a < y < b:
                        has_t = True
                        break
                if not has_t:
                    for a, b, sid in self.edges_h.get(y, []):
                        if sid in existing:
                            continue
                        if a < x < b:
                            has_t = True
                            break
                if has_t:
                    delta_t += 1

        delta_long = 0.0
        max_line_after = 0
        v_edges = [(x0, y0, y1), (x1, y0, y1)]
        h_edges = [(y0, x0, x1), (y1, x0, x1)]
        for x, a, b in v_edges:
            if x <= 0 or x >= self.patio_w:
                continue
            old = self.seams_v.get(x, [])
            new = self._preview_add_interval(old, a, b)
            delta_long += self._line_total_penalty(new) - self._line_total_penalty(old)
            max_line_after = max(max_line_after, self._max_interval_len(new))
        for y, a, b in h_edges:
            if y <= 0 or y >= self.patio_h:
                continue
            old = self.seams_h.get(y, [])
            new = self._preview_add_interval(old, a, b)
            delta_long += self._line_total_penalty(new) - self._line_total_penalty(old)
            max_line_after = max(max_line_after, self._max_interval_len(new))

        stagger_bonus = 0.0
        for x, a, b in v_edges:
            if x <= 0 or x >= self.patio_w:
                continue
            old = self.seams_v.get(x, [])
            overlap = self._overlap_len(old, a, b)
            if overlap > 0:
                stagger_bonus -= overlap / 100.0
            else:
                stagger_bonus += max(0.5, (b - a) / 250.0)

        contact = compute_contact_score(candidate, occ, inner)
        patio_area = self.patio_w * self.patio_h
        delta_uncovered_proxy = (candidate.area / patio_area) if patio_area else 0.0
        placement_score = (
            candidate.area
            - self.anti.uncovered_proxy_weight * delta_uncovered_proxy
            - self.anti.long_line_weight * delta_long
            - self.anti.cross_penalty * delta_cross
            - self.anti.t_junction_penalty * delta_t
            + self.anti.contact_weight * contact
            + self.anti.stagger_bonus_weight * stagger_bonus
        )
        return CandidateEval(
            placement=candidate,
            placement_score=placement_score,
            delta_long_line=delta_long,
            delta_cross=delta_cross,
            delta_t_junction=delta_t,
            stagger_bonus=stagger_bonus,
            creates_cross=delta_cross > 0,
            max_line_after_mm=max_line_after,
        )

    def commit(self, placement: Placement, inner: InnerRect) -> None:
        x0 = placement.x_mm - inner.x0_mm
        y0 = placement.y_mm - inner.y0_mm
        x1 = x0 + placement.w_mm
        y1 = y0 + placement.h_mm
        sid = self.next_slab_id
        self.next_slab_id += 1
        for pt in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]:
            self.corner_map[pt].add(sid)
        self.edges_v[x0].append((y0, y1, sid))
        self.edges_v[x1].append((y0, y1, sid))
        self.edges_h[y0].append((x0, x1, sid))
        self.edges_h[y1].append((x0, x1, sid))
        if 0 < x0 < self.patio_w:
            self.seams_v[x0] = self._preview_add_interval(self.seams_v.get(x0, []), y0, y1)
        if 0 < x1 < self.patio_w:
            self.seams_v[x1] = self._preview_add_interval(self.seams_v.get(x1, []), y0, y1)
        if 0 < y0 < self.patio_h:
            self.seams_h[y0] = self._preview_add_interval(self.seams_h.get(y0, []), x0, x1)
        if 0 < y1 < self.patio_h:
            self.seams_h[y1] = self._preview_add_interval(self.seams_h.get(y1, []), x0, x1)


def score_layout(layout: Layout, patio: Patio, slab_types: Sequence[SlabType], anti: AntiPatternConfig) -> ScoreBreakdown:
    patio_area = patio.width_mm * patio.depth_mm
    covered = sum(p.area for p in layout.placements)
    coverage_ratio = covered / patio_area if patio_area else 0.0

    long_pen, max_line, over_count = long_line_penalty(layout.placements, patio.width_mm, patio.depth_mm, anti.long_line_threshold_mm)
    cross_count, t_count = corner_and_t_penalties(layout.placements, patio.width_mm, patio.depth_mm)

    counts = [layout.used_counts.get(s.name, 0) for s in slab_types]
    used_total = sum(counts)
    if used_total > 0:
        proportions = [c / used_total for c in counts if c > 0]
        entropy = -sum(p * math.log(p + 1e-12) for p in proportions)
        max_entropy = math.log(len([s for s in slab_types if s.count_available > 0]) + 1e-12)
        variety_pen = max(0.0, (max_entropy - entropy) * 20.0)
    else:
        variety_pen = 50.0

    coverage_score = coverage_ratio * 50000.0
    uncovered_penalty = (1.0 - coverage_ratio) * 30000.0
    cross_penalty = cross_count * anti.cross_penalty
    t_penalty = t_count * anti.t_junction_penalty

    total = coverage_score - uncovered_penalty - 6.0 * long_pen - cross_penalty - t_penalty - 0.35 * variety_pen

    return ScoreBreakdown(
        score_total=total,
        coverage_ratio=coverage_ratio,
        coverage_score=coverage_score,
        uncovered_penalty=uncovered_penalty,
        long_line_penalty=long_pen,
        cross_penalty=cross_penalty,
        t_junction_penalty=t_penalty,
        variety_penalty=variety_pen,
        max_long_line_mm=max_line,
        cross_count=cross_count,
        t_junction_count=t_count,
        seams_over_threshold=over_count,
    )


def has_joint_clearance(candidate: Placement, placements: Sequence[Placement], joint_mm: int) -> bool:
    for p in placements:
        no_conflict = (
            candidate.right + joint_mm <= p.x_mm
            or p.right + joint_mm <= candidate.x_mm
            or candidate.bottom + joint_mm <= p.y_mm
            or p.bottom + joint_mm <= candidate.y_mm
        )
        if not no_conflict:
            return False
    return True


def build_candidate_placements(
    x: int,
    y: int,
    inner: InnerRect,
    joint_mm: int,
    slabs: Sequence[SlabType],
    layout: Layout,
    occ: OccupancyGrid,
) -> List[Placement]:
    out: List[Placement] = []
    x_limit = inner.x0_mm + inner.width_mm
    y_limit = inner.y0_mm + inner.depth_mm

    for slab in slabs:
        if layout.used_counts.get(slab.name, 0) >= slab.count_available:
            continue
        for w, h, rotated in slab.orientations():
            if x + w > x_limit or y + h > y_limit:
                continue
            cand = Placement(slab=slab.name, x_mm=x, y_mm=y, w_mm=w, h_mm=h, rotated=rotated)
            if occ.is_free(x, y, w, h) and has_joint_clearance(cand, layout.placements, joint_mm):
                out.append(cand)
    return out


def compute_contact_score(candidate: Placement, occ: OccupancyGrid, inner: InnerRect) -> float:
    c0 = (candidate.x_mm - inner.x0_mm) // GRID_MM
    c1 = c0 + candidate.w_mm // GRID_MM
    r0 = (candidate.y_mm - inner.y0_mm) // GRID_MM
    r1 = r0 + candidate.h_mm // GRID_MM

    score = 0.0

    if c0 == 0:
        score += (r1 - r0)
    elif c0 > 0:
        left_col = c0 - 1
        for r in range(r0, r1):
            if occ.bits[r] & (1 << left_col):
                score += 1.0

    if c1 == occ.cols:
        score += (r1 - r0)
    elif c1 < occ.cols:
        right_col = c1
        for r in range(r0, r1):
            if occ.bits[r] & (1 << right_col):
                score += 1.0

    if r0 == 0:
        score += (c1 - c0)
    elif r0 > 0:
        top_row = r0 - 1
        mask = occ._mask(c0, c1)
        score += (occ.bits[top_row] & mask).bit_count()

    if r1 == occ.rows:
        score += (c1 - c0)
    elif r1 < occ.rows:
        bottom_row = r1
        mask = occ._mask(c0, c1)
        score += (occ.bits[bottom_row] & mask).bit_count()

    return score


def choose_candidate(
    cands: Sequence[Placement],
    layout: Layout,
    slabs: Sequence[SlabType],
    occ: OccupancyGrid,
    inner: InnerRect,
    policy: PlacementPolicy,
    anti: AntiPatternConfig,
    pattern_state: PatternState,
    rng: random.Random,
) -> Placement:
    slab_by_name = {s.name: s for s in slabs}
    total_area = inner.width_mm * inner.depth_mm
    filled_area = sum(p.area for p in layout.placements)
    remaining_free_fraction = 1.0 - (filled_area / total_area if total_area else 0.0)

    def is_small(c: Placement) -> bool:
        return c.slab in policy.small_slab_names or c.area <= policy.small_area_threshold

    large = [c for c in cands if not is_small(c)]
    prefiltered = list(cands)
    if remaining_free_fraction > policy.min_remaining_ratio_for_small and large:
        prefiltered = large

    evals: List[CandidateEval] = [pattern_state.evaluate_candidate(c, occ, inner, filled_area) for c in prefiltered]

    enforce_cross = anti.cross_reject and remaining_free_fraction > anti.cross_reject_free_area_below
    enforce_long = anti.long_line_reject and remaining_free_fraction > anti.long_line_reject_free_area_below

    filtered = [
        ev
        for ev in evals
        if (not enforce_cross or not ev.creates_cross)
        and (not enforce_long or ev.max_line_after_mm <= anti.max_allowed_long_line_mm)
    ]
    if not filtered:
        filtered = evals

    def slab_weight(c: Placement) -> float:
        return max(0.01, policy.slab_weights.get(c.slab, 1.0))

    def balanced_factor(c: Placement) -> float:
        slab = slab_by_name[c.slab]
        used = layout.used_counts.get(c.slab, 0)
        remaining = max(0, slab.count_available - used)
        return 0.25 + (remaining / max(1, slab.count_available))

    if policy.strategy == "largest_first":
        best = max(
            filtered,
            key=lambda ev: (
                ev.placement_score,
                ev.placement.area,
                max(ev.placement.w_mm, ev.placement.h_mm),
                -ev.placement.y_mm,
                -ev.placement.x_mm,
            ),
        )
        return best.placement

    weighted: List[Tuple[float, Placement]] = []
    for ev in filtered:
        c = ev.placement
        weight = slab_weight(c) * max(1.0, ev.placement_score)
        if policy.strategy == "balanced":
            weight *= balanced_factor(c)
        if remaining_free_fraction <= policy.min_remaining_ratio_for_small and is_small(c):
            weight *= max(0.01, policy.late_stage_small_bonus)
        weighted.append((max(0.001, weight), c))

    total = sum(w for w, _ in weighted)
    pick = rng.uniform(0.0, total)
    run = 0.0
    for w, c in weighted:
        run += w
        if run >= pick:
            return c
    return weighted[-1][1]


def construct_layout(
    inner: InnerRect,
    slabs: Sequence[SlabType],
    joint_mm: int,
    policy: PlacementPolicy,
    anti: AntiPatternConfig,
    rng: random.Random,
    max_steps: Optional[int] = None,
) -> Tuple[Layout, FillStats]:
    occ = OccupancyGrid(inner.width_mm, inner.depth_mm)
    layout = Layout()
    pattern_state = PatternState(inner.width_mm, inner.depth_mm, anti)
    stats = FillStats()
    if max_steps is None:
        max_steps = occ.rows * occ.cols + 1
    steps = 0

    while steps < max_steps:
        steps += 1
        frontier = occ.first_empty()
        if frontier is None:
            stats.end_reason = "no_frontier_empty_cells"
            break
        lx, ly = frontier
        x = inner.x0_mm + lx
        y = inner.y0_mm + ly
        cands = build_candidate_placements(x, y, inner, joint_mm, slabs, layout, occ)
        if not cands:
            occ.set_rect(lx, ly, GRID_MM, GRID_MM, True)
            stats.frontier_failures += 1
            stats.blocked_cells += 1
            continue

        stats.attempted_placements += len(cands)
        pick = choose_candidate(cands, layout, slabs, occ, inner, policy, anti, pattern_state, rng)

        layout.placements.append(pick)
        layout.used_counts[pick.slab] += 1
        occ.set_rect(pick.x_mm - inner.x0_mm, pick.y_mm - inner.y0_mm, pick.w_mm, pick.h_mm, True)
        pattern_state.commit(pick, inner)

    if steps >= max_steps and stats.end_reason == "unknown":
        stats.end_reason = "max_steps_reached"

    return layout, stats


def improve_layout(
    base: Layout,
    inner: InnerRect,
    slabs: Sequence[SlabType],
    joint_mm: int,
    policy: PlacementPolicy,
    anti: AntiPatternConfig,
    rng: random.Random,
    iterations: int = 60,
) -> Layout:
    current = Layout(list(base.placements), defaultdict(int, base.used_counts))
    patio = Patio(inner.width_mm, inner.depth_mm)
    current_score = score_layout(current, patio, slabs, anti).score_total

    for i in range(iterations):
        if not current.placements:
            break
        k = rng.randint(1, max(1, len(current.placements) // 8))
        remove_idx = set(rng.sample(range(len(current.placements)), k=k))
        kept = [p for idx, p in enumerate(current.placements) if idx not in remove_idx]
        candidate = Layout(placements=kept, used_counts=defaultdict(int))
        for p in kept:
            candidate.used_counts[p.slab] += 1

        occ = OccupancyGrid(inner.width_mm, inner.depth_mm)
        pattern_state = PatternState(inner.width_mm, inner.depth_mm, anti)
        for p in kept:
            occ.set_rect(p.x_mm - inner.x0_mm, p.y_mm - inner.y0_mm, p.w_mm, p.h_mm, True)
            pattern_state.commit(p, inner)

        refill_steps = 0
        while refill_steps < 2500:
            refill_steps += 1
            frontier = occ.first_empty()
            if frontier is None:
                break
            lx, ly = frontier
            x = inner.x0_mm + lx
            y = inner.y0_mm + ly
            cands = build_candidate_placements(x, y, inner, joint_mm, slabs, candidate, occ)
            if not cands:
                occ.set_rect(lx, ly, GRID_MM, GRID_MM, True)
                continue
            pick = choose_candidate(cands, candidate, slabs, occ, inner, policy, anti, pattern_state, rng)
            candidate.placements.append(pick)
            candidate.used_counts[pick.slab] += 1
            occ.set_rect(pick.x_mm - inner.x0_mm, pick.y_mm - inner.y0_mm, pick.w_mm, pick.h_mm, True)
            pattern_state.commit(pick, inner)

        cand_score = score_layout(candidate, patio, slabs, anti).score_total
        temp = max(0.01, 1.0 - i / iterations)
        if cand_score > current_score or rng.random() < math.exp((cand_score - current_score) / (200.0 * temp)):
            current = candidate
            current_score = cand_score

    return current


def layout_to_json(layout: Layout, score: ScoreBreakdown, slabs: Sequence[SlabType], policy: PlacementPolicy, anti: AntiPatternConfig) -> Dict[str, object]:
    counts: Dict[str, int] = defaultdict(int)
    area_by_slab: Dict[str, int] = defaultdict(int)
    placements = []
    for p in layout.placements:
        counts[p.slab] += 1
        area_by_slab[p.slab] += p.area
        placements.append(
            {
                "slab": p.slab,
                "x_mm": int(p.x_mm),
                "y_mm": int(p.y_mm),
                "rotated": bool(p.rotated),
            }
        )
    counts_remaining = {s.name: max(0, s.count_available - counts.get(s.name, 0)) for s in slabs}
    total_area = sum(p.area for p in layout.placements)
    area_percent_by_slab = {
        s.name: round((area_by_slab.get(s.name, 0) / total_area) * 100.0, 3) if total_area > 0 else 0.0 for s in slabs
    }

    return {
        "placements": placements,
        "summary": {
            "coverage_percent": round(score.coverage_ratio * 100.0, 3),
            "counts_used": dict(sorted(counts.items())),
            "counts_remaining": dict(sorted(counts_remaining.items())),
            "area_percent_by_slab": dict(sorted(area_percent_by_slab.items())),
            "policy_used": {
                "strategy": policy.strategy,
                "slab_weights": {s.name: policy.slab_weights.get(s.name, 1.0) for s in slabs},
                "min_remaining_ratio_for_small": policy.min_remaining_ratio_for_small,
                "late_stage_small_bonus": policy.late_stage_small_bonus,
                "small_slab_names": sorted(policy.small_slab_names),
                "small_area_threshold": policy.small_area_threshold,
            },
            "anti_pattern_used": {
                "long_line_threshold_mm": anti.long_line_threshold_mm,
                "max_allowed_long_line_mm": anti.max_allowed_long_line_mm,
                "cross_reject": anti.cross_reject,
                "cross_reject_free_area_below": anti.cross_reject_free_area_below,
                "long_line_reject": anti.long_line_reject,
                "long_line_reject_free_area_below": anti.long_line_reject_free_area_below,
                "cross_penalty": anti.cross_penalty,
                "t_junction_penalty": anti.t_junction_penalty,
                "long_line_penalty_factor": anti.long_line_penalty_factor,
            },
            "score_total": round(score.score_total, 3),
            "score_breakdown": {
                "coverage_score": round(score.coverage_score, 3),
                "uncovered_penalty": round(score.uncovered_penalty, 3),
                "long_line_penalty": round(score.long_line_penalty, 3),
                "cross_penalty": round(score.cross_penalty, 3),
                "t_junction_penalty": round(score.t_junction_penalty, 3),
                "variety_penalty": round(score.variety_penalty, 3),
                "max_long_line_mm": int(score.max_long_line_mm),
                "cross_count": int(score.cross_count),
                "t_junction_count": int(score.t_junction_count),
                "seams_over_threshold": int(score.seams_over_threshold),
            },
        },
    }


def write_svg(path: Path, patio: Patio, placements: Sequence[Placement], wall_gap_mm: int, joint_mm: int) -> None:
    margin = 30
    scale = 0.12
    svg_w = int(patio.width_mm * scale + 2 * margin)
    svg_h = int(patio.depth_mm * scale + 2 * margin)

    palette = [
        "#d9c39a",
        "#c9b38d",
        "#e2ceb2",
        "#bca886",
        "#d8c7ae",
        "#c7b495",
        "#b89c78",
    ]
    color_by_slab: Dict[str, str] = {}

    def color(name: str) -> str:
        if name not in color_by_slab:
            color_by_slab[name] = palette[len(color_by_slab) % len(palette)]
        return color_by_slab[name]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">',
        '<rect width="100%" height="100%" fill="#f6f6f6"/>',
    ]

    px = margin
    py = margin
    pw = patio.width_mm * scale
    ph = patio.depth_mm * scale

    lines.append(f'<rect x="{px:.2f}" y="{py:.2f}" width="{pw:.2f}" height="{ph:.2f}" fill="#f0f0f0" stroke="#222" stroke-width="2"/>')

    inner_x = px + wall_gap_mm * scale
    inner_y = py + wall_gap_mm * scale
    inner_w = max(0, (patio.width_mm - 2 * wall_gap_mm) * scale)
    inner_h = max(0, (patio.depth_mm - 2 * wall_gap_mm) * scale)
    lines.append(
        f'<rect x="{inner_x:.2f}" y="{inner_y:.2f}" width="{inner_w:.2f}" height="{inner_h:.2f}" fill="#e8e8e8" stroke="#aaa" stroke-dasharray="4,4"/>'
    )

    for p in placements:
        x = px + p.x_mm * scale
        y = py + p.y_mm * scale
        w = p.w_mm * scale
        h = p.h_mm * scale
        lines.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{color(p.slab)}" stroke="#333" stroke-width="0.8"/>'
        )
        if w > 45 and h > 20:
            lines.append(
                f'<text x="{x + w / 2:.2f}" y="{y + h / 2:.2f}" font-size="9" fill="#222" text-anchor="middle" dominant-baseline="middle">{p.slab}</text>'
            )

    if joint_mm > 0:
        lines.append(f'<text x="{margin}" y="{svg_h - 8}" font-size="10" fill="#555">Joint gap represented by slab spacing ({joint_mm}mm).</text>')

    lines.append('</svg>')
    path.write_text("\n".join(lines), encoding="utf-8")


def resolve_policy(data: Dict[str, object], slabs: Sequence[SlabType]) -> PlacementPolicy:
    placement_policy = data.get("placement_policy", {})
    if not isinstance(placement_policy, dict):
        placement_policy = {}

    strategy = str(placement_policy.get("strategy", "largest_first")).strip().lower()
    if strategy not in {"largest_first", "weighted", "balanced"}:
        strategy = "largest_first"

    default_weights = {}
    area_sorted = sorted(slabs, key=lambda s: s.w_mm * s.h_mm, reverse=True)
    max_rank = max(1, len(area_sorted) - 1)
    for idx, s in enumerate(area_sorted):
        default_weights[s.name] = 1.0 + 4.0 * (max_rank - idx) / max_rank

    slab_weights_in = placement_policy.get("slab_weights", {})
    slab_weights = dict(default_weights)
    if isinstance(slab_weights_in, dict):
        for name, value in slab_weights_in.items():
            try:
                slab_weights[str(name)] = max(0.01, float(value))
            except (TypeError, ValueError):
                continue

    min_ratio = float(placement_policy.get("min_remaining_ratio_for_small", 0.20))
    min_ratio = min(0.95, max(0.0, min_ratio))
    late_bonus = max(0.01, float(placement_policy.get("late_stage_small_bonus", 2.0)))

    by_area = sorted({s.w_mm * s.h_mm for s in slabs})
    small_area_threshold = by_area[1] if len(by_area) > 1 else by_area[0]
    small_names = {s.name for s in slabs if (s.w_mm * s.h_mm) <= small_area_threshold}
    for s in slabs:
        if s.name in {"290x290", "600x290"}:
            small_names.add(s.name)

    return PlacementPolicy(
        strategy=strategy,
        slab_weights=slab_weights,
        min_remaining_ratio_for_small=min_ratio,
        late_stage_small_bonus=late_bonus,
        small_slab_names=small_names,
        small_area_threshold=small_area_threshold,
    )


def parse_input(path: Path) -> Tuple[Patio, int, int, int, List[SlabType], PlacementPolicy, AntiPatternConfig, int, float, Optional[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    patio_raw = data["patio"]
    patio = Patio(
        width_mm=validate_dim("patio.width_mm", int(patio_raw["width_mm"])),
        depth_mm=validate_dim("patio.depth_mm", int(patio_raw["depth_mm"])),
    )
    joint_mm = validate_dim("joint_mm", int(data.get("joint_mm", 10)))
    wall_gap_mm = max(0, clamp_to_grid(int(data.get("wall_gap_mm", 10))))
    perimeter_gap_mm = max(0, clamp_to_grid(int(data.get("perimeter_gap_mm", 0))))

    slabs: List[SlabType] = []
    for s in data["slabs"]:
        w = validate_dim(f"{s.get('name', 'slab')}.w_mm", int(s["w_mm"]))
        h = validate_dim(f"{s.get('name', 'slab')}.h_mm", int(s["h_mm"]))
        cnt = int(s.get("count", 0))
        if cnt <= 0:
            continue
        slabs.append(SlabType(name=str(s["name"]), w_mm=w, h_mm=h, count_available=cnt))

    if not slabs:
        raise ValueError("No slab types with positive count were provided")

    policy = resolve_policy(data, slabs)

    anti_raw = data.get("anti_pattern", {})
    if not isinstance(anti_raw, dict):
        anti_raw = {}
    anti = AntiPatternConfig(
        long_line_threshold_mm=max(100, int(anti_raw.get("long_line_threshold_mm", LONG_LINE_THRESHOLD_MM))),
        max_allowed_long_line_mm=max(100, int(anti_raw.get("max_allowed_long_line_mm", MAX_ALLOWED_LONG_LINE_MM))),
        cross_reject=bool(anti_raw.get("cross_reject", True)),
        cross_reject_free_area_below=min(0.99, max(0.0, float(anti_raw.get("cross_reject_free_area_below", 0.08)))),
        long_line_reject=bool(anti_raw.get("long_line_reject", True)),
        long_line_reject_free_area_below=min(0.99, max(0.0, float(anti_raw.get("long_line_reject_free_area_below", 0.10)))),
        cross_penalty=max(0.0, float(anti_raw.get("cross_penalty", 5000))),
        t_junction_penalty=max(0.0, float(anti_raw.get("t_junction_penalty", 150))),
        long_line_penalty_factor=max(0.01, float(anti_raw.get("long_line_penalty_factor", 1.0))),
    )

    num_solutions = max(1, int(data.get("num_solutions", 10)))
    coverage_target = float(data.get("coverage_target", 0.95))
    seed = data.get("seed")
    return patio, joint_mm, wall_gap_mm, perimeter_gap_mm, slabs, policy, anti, num_solutions, coverage_target, (int(seed) if seed is not None else None)


def compute_inner_rect(patio: Patio, wall_gap_mm: int, perimeter_gap_mm: int) -> InnerRect:
    x0 = perimeter_gap_mm
    y0 = wall_gap_mm + perimeter_gap_mm
    width = patio.width_mm - 2 * perimeter_gap_mm
    depth = patio.depth_mm - wall_gap_mm - 2 * perimeter_gap_mm
    if width <= 0 or depth <= 0:
        raise ValueError("wall/perimeter gaps leave no patio area")
    return InnerRect(x0_mm=x0, y0_mm=y0, width_mm=width, depth_mm=depth)


def shift_layout(layout: Layout, dx: int, dy: int) -> Layout:
    shifted = Layout(placements=[], used_counts=defaultdict(int, layout.used_counts))
    for p in layout.placements:
        shifted.placements.append(
            Placement(slab=p.slab, x_mm=p.x_mm + dx, y_mm=p.y_mm + dy, w_mm=p.w_mm, h_mm=p.h_mm, rotated=p.rotated)
        )
    return shifted


def run_search(
    patio_full: Patio,
    inner: InnerRect,
    slabs: Sequence[SlabType],
    policy: PlacementPolicy,
    anti: AntiPatternConfig,
    joint_mm: int,
    num_solutions: int,
    coverage_target: float,
    beam_width: int,
    time_limit_seconds: float,
    rng: random.Random,
) -> List[Tuple[Layout, ScoreBreakdown]]:
    start = time.time()
    pool: List[Tuple[Layout, ScoreBreakdown]] = []
    seen_signatures: Set[Tuple[Tuple[str, int, int, bool], ...]] = set()
    attempt = 0

    best_coverage = 0.0
    while (time.time() - start) < time_limit_seconds:
        attempt += 1
        candidate, stats = construct_layout(inner, slabs, joint_mm, policy, anti, rng)
        candidate = improve_layout(candidate, inner, slabs, joint_mm, policy, anti, rng, iterations=40)
        scored = score_layout(candidate, Patio(inner.width_mm, inner.depth_mm), slabs, anti)
        sig = tuple(sorted((p.slab, p.x_mm, p.y_mm, p.rotated) for p in candidate.placements))
        elapsed = time.time() - start
        if stats.end_reason == "unknown":
            stats.end_reason = "search_iteration_complete"
        print(
            f"[debug] attempt={attempt:4d} end_reason={stats.end_reason} tried={stats.attempted_placements} "
            f"frontier_failures={stats.frontier_failures} blocked_cells={stats.blocked_cells} elapsed={elapsed:.2f}s",
            flush=True,
        )
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            pool.append((candidate, scored))
            pool.sort(key=lambda t: t[1].score_total, reverse=True)
            if len(pool) > beam_width:
                pool = pool[:beam_width]
            best = pool[0][1]
            best_coverage = max(best_coverage, best.coverage_ratio)
            print(
                f"[progress] attempt={attempt:4d} pool={len(pool):3d} best_score={best.score_total:9.2f} "
                f"coverage={best.coverage_ratio*100:6.2f}% long={best.max_long_line_mm:4d} cross={best.cross_count:3d}",
                flush=True,
            )
        if best_coverage >= coverage_target and len(pool) >= num_solutions:
            break

    pool.sort(key=lambda t: t[1].score_total, reverse=True)
    best_n = pool[:num_solutions]
    shifted_out: List[Tuple[Layout, ScoreBreakdown]] = []
    for layout, score in best_n:
        shifted_layout = shift_layout(layout, 0, 0)
        shifted_score = score_layout(shifted_layout, patio_full, slabs, anti)
        shifted_out.append((shifted_layout, shifted_score))
    return shifted_out


def write_outputs(
    out_dir: Path,
    patio: Patio,
    wall_gap_mm: int,
    joint_mm: int,
    results: Sequence[Tuple[Layout, ScoreBreakdown]],
    slabs: Sequence[SlabType],
    policy: PlacementPolicy,
    anti: AntiPatternConfig,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = []

    for idx, (layout, score) in enumerate(results, start=1):
        json_obj = layout_to_json(layout, score, slabs, policy, anti)
        json_path = out_dir / f"layout_{idx}.json"
        svg_path = out_dir / f"layout_{idx}.svg"
        json_path.write_text(json.dumps(json_obj, indent=2), encoding="utf-8")
        write_svg(svg_path, patio, layout.placements, wall_gap_mm, joint_mm)

        summary_lines.append(
            f"Layout {idx}: score={score.score_total:.2f} coverage={score.coverage_ratio*100:.2f}% "
            f"long_line_pen={score.long_line_penalty:.2f} max_line={score.max_long_line_mm}mm seams>{anti.long_line_threshold_mm}={score.seams_over_threshold} "
            f"cross={score.cross_count} t_junctions={score.t_junction_count} policy={policy.strategy}\n"
        )

    (out_dir / "summary.txt").write_text("".join(summary_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate candidate patio paving layouts")
    parser.add_argument("input_json", help="Path to input JSON file")
    parser.add_argument("--time-limit-seconds", type=float, default=60.0, help="Time limit for search")
    parser.add_argument("--beam-width", type=int, default=50, help="Max number of candidate layouts retained")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json)
    patio_full, joint_mm, wall_gap_mm, perimeter_gap_mm, slabs, policy, anti, num_solutions, coverage_target, seed = parse_input(input_path)

    rng = random.Random(seed)
    inner = compute_inner_rect(patio_full, wall_gap_mm, perimeter_gap_mm)

    print(
        f"Starting search: patio={patio_full.width_mm}x{patio_full.depth_mm}mm inner={inner.width_mm}x{inner.depth_mm}mm "
        f"slabs={len(slabs)} target_solutions={num_solutions} seed={seed} strategy={policy.strategy}",
        flush=True,
    )

    results = run_search(
        patio_full=patio_full,
        inner=inner,
        slabs=slabs,
        policy=policy,
        anti=anti,
        joint_mm=joint_mm,
        num_solutions=num_solutions,
        coverage_target=coverage_target,
        beam_width=max(2, args.beam_width),
        time_limit_seconds=max(1.0, args.time_limit_seconds),
        rng=rng,
    )

    if not results:
        raise RuntimeError("No layouts generated within time limit")

    write_outputs(Path("out"), patio_full, wall_gap_mm, joint_mm, results, slabs, policy, anti)
    print(f"Wrote {len(results)} layouts to out/", flush=True)


if __name__ == "__main__":
    main()


README = """
Patio Layout Generator (single-file Python 3.11)
=================================================

Usage
-----
python patio_layout.py input.json [--time-limit-seconds 60] [--beam-width 50]

Input JSON example
------------------
{
  "patio": { "width_mm": 4800, "depth_mm": 2100 },
  "joint_mm": 10,
  "wall_gap_mm": 10,
  "slabs": [
    { "name": "900x600", "w_mm": 900, "h_mm": 600, "count": 20 },
    { "name": "600x600", "w_mm": 600, "h_mm": 600, "count": 20 },
    { "name": "600x290", "w_mm": 600, "h_mm": 290, "count": 40 },
    { "name": "290x290", "w_mm": 290, "h_mm": 290, "count": 40 }
  ],
  "num_solutions": 10,
  "seed": 123,
  "placement_policy": {
    "strategy": "largest_first",
    "slab_weights": {
      "900x600": 5.0,
      "600x600": 4.0,
      "600x290": 2.0,
      "290x290": 1.0
    },
    "min_remaining_ratio_for_small": 0.20,
    "late_stage_small_bonus": 2.0
  },
  "anti_pattern": {
    "long_line_threshold_mm": 1200,
    "max_allowed_long_line_mm": 1800,
    "cross_reject": true,
    "cross_reject_free_area_below": 0.08,
    "long_line_reject": true,
    "long_line_reject_free_area_below": 0.10,
    "cross_penalty": 5000,
    "t_junction_penalty": 150,
    "long_line_penalty_factor": 1.0
  }
}

Outputs
-------
- out/layout_<N>.json : placements + score summary.
- out/layout_<N>.svg  : visual layout drawing.
- out/summary.txt     : compact per-layout score overview.

Notes
-----
- Coordinates are top-left origin with y increasing away from the wall.
- Slab rectangles are not cut in this version; uncovered space is penalized.
- The search combines randomized constructive fill + shake/refill improvement.
"""
