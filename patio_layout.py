#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def long_line_penalty(placements: Sequence[Placement], patio_w: int, patio_h: int, threshold: int) -> Tuple[float, int]:
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

    for x, spans in vertical_by_x.items():
        if x <= 0 or x >= patio_w:
            continue
        for a, b in merge(spans):
            length = b - a
            max_len = max(max_len, length)
            if length > threshold:
                over = length - threshold
                penalty += over * 0.025

    for y, spans in horizontal_by_y.items():
        if y <= 0 or y >= patio_h:
            continue
        for a, b in merge(spans):
            length = b - a
            max_len = max(max_len, length)
            if length > threshold:
                over = length - threshold
                penalty += over * 0.025

    return penalty, max_len


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


def score_layout(layout: Layout, patio: Patio, slab_types: Sequence[SlabType]) -> ScoreBreakdown:
    patio_area = patio.width_mm * patio.depth_mm
    covered = sum(p.area for p in layout.placements)
    coverage_ratio = covered / patio_area if patio_area else 0.0

    long_pen, max_line = long_line_penalty(layout.placements, patio.width_mm, patio.depth_mm, LONG_LINE_THRESHOLD_MM)
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

    # Coverage is the primary objective; pattern penalties are secondary tie-breakers.
    coverage_score = coverage_ratio * 30000.0
    uncovered_penalty = (1.0 - coverage_ratio) * 10000.0
    cross_penalty = cross_count * 220.0
    t_penalty = t_count * 20.0

    total = coverage_score - uncovered_penalty - 0.75 * long_pen - cross_penalty - t_penalty - 0.35 * variety_pen

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
    rng: random.Random,
) -> List[Placement]:
    out: List[Placement] = []
    x_limit = inner.x0_mm + inner.width_mm
    y_limit = inner.y0_mm + inner.depth_mm

    slab_order = list(slabs)
    rng.shuffle(slab_order)

    for slab in slab_order:
        if layout.used_counts.get(slab.name, 0) >= slab.count_available:
            continue
        orientations = slab.orientations()
        rng.shuffle(orientations)
        for w, h, rotated in orientations:
            if x + w > x_limit or y + h > y_limit:
                continue
            cand = Placement(slab=slab.name, x_mm=x, y_mm=y, w_mm=w, h_mm=h, rotated=rotated)
            if occ.is_free(x, y, w, h) and has_joint_clearance(cand, layout.placements, joint_mm):
                out.append(cand)
    return out


def placement_delta_score(layout: Layout, placement: Placement, patio: Patio, slab_types: Sequence[SlabType]) -> float:
    trial = Layout(placements=layout.placements + [placement], used_counts=defaultdict(int, layout.used_counts))
    trial.used_counts[placement.slab] += 1
    scored = score_layout(trial, patio, slab_types)
    return scored.score_total


def construct_layout(
    inner: InnerRect,
    slabs: Sequence[SlabType],
    joint_mm: int,
    rng: random.Random,
    max_steps: Optional[int] = None,
) -> Tuple[Layout, FillStats]:
    occ = OccupancyGrid(inner.width_mm, inner.depth_mm)
    layout = Layout()
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
        cands = build_candidate_placements(x, y, inner, joint_mm, slabs, layout, occ, rng)
        if not cands:
            occ.set_rect(lx, ly, GRID_MM, GRID_MM, True)
            stats.frontier_failures += 1
            stats.blocked_cells += 1
            continue

        weighted: List[Tuple[float, Placement]] = []
        for c in cands:
            base = c.area
            jitter = rng.uniform(0.9, 1.15)
            score = placement_delta_score(layout, c, Patio(inner.width_mm, inner.depth_mm), slabs)
            weighted.append((base * jitter + 0.001 * score, c))
            stats.attempted_placements += 1

        weighted.sort(key=lambda t: t[0], reverse=True)
        top = weighted[: min(4, len(weighted))]
        pick = rng.choice(top)[1]

        layout.placements.append(pick)
        layout.used_counts[pick.slab] += 1
        occ.set_rect(pick.x_mm - inner.x0_mm, pick.y_mm - inner.y0_mm, pick.w_mm, pick.h_mm, True)

    if steps >= max_steps and stats.end_reason == "unknown":
        stats.end_reason = "max_steps_reached"

    return layout, stats


def improve_layout(
    base: Layout,
    inner: InnerRect,
    slabs: Sequence[SlabType],
    joint_mm: int,
    rng: random.Random,
    iterations: int = 60,
) -> Layout:
    current = Layout(list(base.placements), defaultdict(int, base.used_counts))
    patio = Patio(inner.width_mm, inner.depth_mm)
    current_score = score_layout(current, patio, slabs).score_total

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
        for p in kept:
            occ.set_rect(p.x_mm - inner.x0_mm, p.y_mm - inner.y0_mm, p.w_mm, p.h_mm, True)

        refill_steps = 0
        while refill_steps < 2500:
            refill_steps += 1
            frontier = occ.first_empty()
            if frontier is None:
                break
            lx, ly = frontier
            x = inner.x0_mm + lx
            y = inner.y0_mm + ly
            cands = build_candidate_placements(x, y, inner, joint_mm, slabs, candidate, occ, rng)
            if not cands:
                occ.set_rect(lx, ly, GRID_MM, GRID_MM, True)
                continue
            pick = max(cands, key=lambda c: c.area + rng.uniform(0, 2000))
            candidate.placements.append(pick)
            candidate.used_counts[pick.slab] += 1
            occ.set_rect(pick.x_mm - inner.x0_mm, pick.y_mm - inner.y0_mm, pick.w_mm, pick.h_mm, True)

        cand_score = score_layout(candidate, patio, slabs).score_total
        temp = max(0.01, 1.0 - i / iterations)
        if cand_score > current_score or rng.random() < math.exp((cand_score - current_score) / (200.0 * temp)):
            current = candidate
            current_score = cand_score

    return current


def layout_to_json(layout: Layout, score: ScoreBreakdown) -> Dict[str, object]:
    counts: Dict[str, int] = defaultdict(int)
    placements = []
    for p in layout.placements:
        counts[p.slab] += 1
        placements.append(
            {
                "slab": p.slab,
                "x_mm": int(p.x_mm),
                "y_mm": int(p.y_mm),
                "rotated": bool(p.rotated),
            }
        )
    return {
        "placements": placements,
        "summary": {
            "coverage_percent": round(score.coverage_ratio * 100.0, 3),
            "counts_used": dict(sorted(counts.items())),
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


def parse_input(path: Path) -> Tuple[Patio, int, int, int, List[SlabType], int, float, Optional[int]]:
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

    num_solutions = max(1, int(data.get("num_solutions", 10)))
    coverage_target = float(data.get("coverage_target", 0.95))
    seed = data.get("seed")
    return patio, joint_mm, wall_gap_mm, perimeter_gap_mm, slabs, num_solutions, coverage_target, (int(seed) if seed is not None else None)


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
        candidate, stats = construct_layout(inner, slabs, joint_mm, rng)
        candidate = improve_layout(candidate, inner, slabs, joint_mm, rng, iterations=40)
        scored = score_layout(candidate, Patio(inner.width_mm, inner.depth_mm), slabs)
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
        shifted_score = score_layout(shifted_layout, patio_full, slabs)
        shifted_out.append((shifted_layout, shifted_score))
    return shifted_out


def write_outputs(
    out_dir: Path,
    patio: Patio,
    wall_gap_mm: int,
    joint_mm: int,
    results: Sequence[Tuple[Layout, ScoreBreakdown]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = []

    for idx, (layout, score) in enumerate(results, start=1):
        json_obj = layout_to_json(layout, score)
        json_path = out_dir / f"layout_{idx}.json"
        svg_path = out_dir / f"layout_{idx}.svg"
        json_path.write_text(json.dumps(json_obj, indent=2), encoding="utf-8")
        write_svg(svg_path, patio, layout.placements, wall_gap_mm, joint_mm)

        summary_lines.append(
            (
                f"Layout {idx}: score={score.score_total:.2f} coverage={score.coverage_ratio*100:.2f}% "
                f"long_line_pen={score.long_line_penalty:.2f} max_line={score.max_long_line_mm}mm "
                f"cross={score.cross_count} t_junctions={score.t_junction_count}\n"
            )
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
    patio_full, joint_mm, wall_gap_mm, perimeter_gap_mm, slabs, num_solutions, coverage_target, seed = parse_input(input_path)

    rng = random.Random(seed)
    inner = compute_inner_rect(patio_full, wall_gap_mm, perimeter_gap_mm)

    print(
        f"Starting search: patio={patio_full.width_mm}x{patio_full.depth_mm}mm inner={inner.width_mm}x{inner.depth_mm}mm "
        f"slabs={len(slabs)} target_solutions={num_solutions} seed={seed}",
        flush=True,
    )

    results = run_search(
        patio_full=patio_full,
        inner=inner,
        slabs=slabs,
        joint_mm=joint_mm,
        num_solutions=num_solutions,
        coverage_target=coverage_target,
        beam_width=max(2, args.beam_width),
        time_limit_seconds=max(1.0, args.time_limit_seconds),
        rng=rng,
    )

    if not results:
        raise RuntimeError("No layouts generated within time limit")

    write_outputs(Path("out"), patio_full, wall_gap_mm, joint_mm, results)
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
  "seed": 123
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
