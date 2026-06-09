"""Maze Crawler Agent — Combined Strategy: A* + FSM + Risk-Aware Navigation.

A high-performance agent for the Kaggle Maze Crawler competition that combines:
- A* pathfinding with dynamic risk weights (safety + enemy avoidance)
- Role-based finite state machine (EXPLORER/HARVESTER/SAPPER/GUARD)
- Dual-mode BFS (pessimistic known-only + optimistic exploration)
- Wall memory caching with mirror-symmetry inference
- Gap-based emergency survival logic
- Crush-hierarchy collision avoidance
- Economic energy budgeting & dynamic build scaling
- Enemy factory tracking & predictive evasion
- Mine ROI calculation & energy transfer optimization
"""

from __future__ import annotations

import heapq
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

TYPE_FACTORY = 0
TYPE_SCOUT = 1
TYPE_WORKER = 2
TYPE_MINER = 3

WALL_N, WALL_E, WALL_S, WALL_W = 1, 2, 4, 8
DIR_TO_BIT = {"NORTH": WALL_N, "EAST": WALL_E, "SOUTH": WALL_S, "WEST": WALL_W}
DIR_DELTA = {"NORTH": (0, 1), "EAST": (1, 0), "SOUTH": (0, -1), "WEST": (-1, 0)}
DIRS = ("NORTH", "EAST", "SOUTH", "WEST")
OPPOSITE = {"NORTH": "SOUTH", "SOUTH": "NORTH", "EAST": "WEST", "WEST": "EAST"}

CRUSH_RANK = {TYPE_FACTORY: 4, TYPE_MINER: 3, TYPE_WORKER: 2, TYPE_SCOUT: 1}

DIR_PRIORITY = {"NORTH": 0, "EAST": 1, "WEST": 2, "SOUTH": 3, "IDLE": 4}

# ═══════════════════════════════════════════════════════════════
# Tunables
# ═══════════════════════════════════════════════════════════════

SAFETY_MARGIN = 4
DANGER_HORIZON = 3
LOW_ENERGY_RATIO = 0.30
LATE_GAME_STOP_BUILD = 30
ROLE_REASSIGN_PERIOD = 8
BFS_MAX_DIST = 80
A_STAR_NODE_LIMIT = 600
WALL_DETOUR_THRESHOLD = 6
ENEMY_EVADE_RADIUS = 3
JUMP_PREFERRED_DIST = 4
CRYSTAL_DETOUR_BUDGET = 3
MINER_NODE_SEARCH_LIMIT = 25
SCOUT_MAX_COUNT = 4
WORKER_MAX_COUNT = 3
MINER_MAX_COUNT = 3

# ═══════════════════════════════════════════════════════════════
# Global State (module-level, persists across turns)
# ═══════════════════════════════════════════════════════════════

_STATE: dict = {
    "walls": {},           # (col, row) → wall_bitfield
    "mines": {},           # (col, row) → (energy, max_energy, owner)
    "mining_nodes": set(), # {(col, row), ...}
    "roles": {},           # uid → role_str
    "targets": {},         # uid → (col, row)
    "turn": 0,
    "enemy_factory_pos": None,
    "last_factory_pos": None,
    "factory_stuck": 0,
}


def _update_state(obs, config):
    """Merge new observation into persistent state."""
    st = _STATE
    width = config.width
    south = obs.southBound

    # Prune scrolled-off data
    st["walls"] = {k: v for k, v in st["walls"].items() if k[1] >= south}
    st["mines"] = {k: v for k, v in st["mines"].items() if k[1] >= south}
    st["mining_nodes"] = {c for c in st["mining_nodes"] if c[1] >= south}

    # Merge walls
    for idx, val in enumerate(obs.walls):
        if val == -1:
            continue
        col, row = idx % width, idx // width + south
        st["walls"][(col, row)] = val
        # Mirror on symmetry axis
        mirror_col = width - 1 - col
        if mirror_col != col:
            # Mirror wall bits (swap E/W)
            mirrored = 0
            if val & WALL_N: mirrored |= WALL_N
            if val & WALL_S: mirrored |= WALL_S
            if val & WALL_E: mirrored |= WALL_W
            if val & WALL_W: mirrored |= WALL_E
            st["walls"][(mirror_col, row)] = mirrored

    # Merge mines
    for key, data in obs.mines.items():
        c, r = (int(x) for x in key.split(","))
        if r >= south:
            st["mines"][(c, r)] = tuple(data)

    # Merge mining nodes
    for key in obs.miningNodes:
        c, r = (int(x) for x in key.split(","))
        if r >= south:
            st["mining_nodes"].add((c, r))

    # Remove mining nodes that are now mines
    for key in obs.mines:
        c, r = (int(x) for x in key.split(","))
        st["mining_nodes"].discard((c, r))

    # Track enemy factory
    for _uid, data in obs.robots.items():
        if data[0] == TYPE_FACTORY and data[4] != obs.player:
            st["enemy_factory_pos"] = (data[1], data[2])

    # Clean roles/targets for dead units
    live = set(obs.robots.keys())
    st["roles"] = {u: r for u, r in st["roles"].items() if u in live}
    st["targets"] = {u: t for u, t in st["targets"].items() if u in live}

    st["turn"] += 1


# ═══════════════════════════════════════════════════════════════
# Unit Data Structure
# ═══════════════════════════════════════════════════════════════

@dataclass(slots=True)
class Unit:
    uid: str
    type: int
    col: int
    row: int
    energy: int
    owner: int
    move_cd: int
    jump_cd: int
    build_cd: int

    @property
    def pos(self) -> Tuple[int, int]:
        return (self.col, self.row)

    @property
    def rank(self) -> int:
        return CRUSH_RANK.get(self.type, 0)


def _make_unit(uid: str, data: list) -> Unit:
    return Unit(
        uid=uid,
        type=data[0], col=data[1], row=data[2], energy=data[3], owner=data[4],
        move_cd=data[5] if len(data) > 5 else 0,
        jump_cd=data[6] if len(data) > 6 else 0,
        build_cd=data[7] if len(data) > 7 else 0,
    )


# ═══════════════════════════════════════════════════════════════
# Game Context (frozen per-turn snapshot)
# ═══════════════════════════════════════════════════════════════

@dataclass(slots=True)
class GameCtx:
    obs: Any
    config: Any
    st: dict
    turn: int
    south: int
    north: int
    width: int
    me: int
    walls: dict
    crystals: Dict[Tuple[int, int], int]
    mines: dict
    nodes: Set[Tuple[int, int]]
    my_factory: Optional[Unit]
    my_units: List[Unit]
    enemy_units: List[Unit]
    enemy_factory: Optional[Unit]
    danger_rows: FrozenSet[int]

    @property
    def gap(self) -> int:
        if self.my_factory is None:
            return 999
        return self.my_factory.row - self.south


def _build_ctx(obs, config) -> GameCtx:
    st = _STATE
    me = obs.player
    all_units = [_make_unit(uid, d) for uid, d in obs.robots.items()]
    my = [u for u in all_units if u.owner == me]
    en = [u for u in all_units if u.owner != me]
    my_f = next((u for u in my if u.type == TYPE_FACTORY), None)
    en_f = next((u for u in en if u.type == TYPE_FACTORY), None)

    danger = frozenset(range(obs.southBound, obs.southBound + DANGER_HORIZON))

    return GameCtx(
        obs=obs, config=config, st=st, turn=st["turn"],
        south=obs.southBound, north=obs.northBound, width=config.width, me=me,
        walls=dict(st["walls"]),
        crystals={tuple(int(x) for x in k.split(",")): v for k, v in obs.crystals.items()},
        mines=dict(st["mines"]),
        nodes=set(st["mining_nodes"]),
        my_factory=my_f, my_units=my, enemy_units=en, enemy_factory=en_f,
        danger_rows=danger,
    )


# ═══════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════

def _in_bounds(ctx: GameCtx, pos: Tuple[int, int]) -> bool:
    c, r = pos
    return 0 <= c < ctx.width and ctx.south <= r <= ctx.north


def _step(pos: Tuple[int, int], d: str) -> Tuple[int, int]:
    dc, dr = DIR_DELTA[d]
    return (pos[0] + dc, pos[1] + dr)


def _wall_at(ctx: GameCtx, pos: Tuple[int, int]) -> int:
    return ctx.walls.get(pos, -1)


def _has_wall(ctx: GameCtx, pos: Tuple[int, int], d: str) -> bool:
    """Check if there's a wall in direction d from pos."""
    val = ctx.walls.get(pos)
    if val is None:
        return False  # unknown = no wall (optimistic)
    return bool(val & DIR_TO_BIT[d])


def _is_fixed_wall(ctx: GameCtx, pos: Tuple[int, int], d: str) -> bool:
    """Check if wall is fixed (perimeter or mirror axis)."""
    col, _ = pos
    w = ctx.width
    half = w // 2
    if d == "WEST" and col == 0:
        return True
    if d == "EAST" and col == w - 1:
        return True
    if d == "EAST" and col == half - 1:
        return True
    if d == "WEST" and col == half:
        return True
    return False


def _legal_moves(ctx: GameCtx, pos: Tuple[int, int], *,
                  strict: bool = True, known_only: bool = False) -> List[str]:
    """Return legal movement directions from pos."""
    moves = []
    for d in DIRS:
        if _has_wall(ctx, pos, d):
            continue
        nxt = _step(pos, d)
        if not _in_bounds(ctx, nxt):
            continue
        if known_only and nxt not in ctx.walls:
            continue
        if strict and pos[1] not in ctx.danger_rows and nxt[1] in ctx.danger_rows:
            continue
        moves.append(d)
    return moves


def _predict_cell(unit: Unit, action: str) -> Tuple[int, int]:
    """Return the cell the unit will occupy after this action."""
    if action in DIRS:
        return _step(unit.pos, action)
    if action.startswith("JUMP_"):
        d = action[5:]
        dc, dr = DIR_DELTA[d]
        return (unit.col + 2 * dc, unit.row + 2 * dr)
    if action in ("BUILD_SCOUT", "BUILD_WORKER", "BUILD_MINER"):
        return (unit.col, unit.row + 1)
    return unit.pos


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ═══════════════════════════════════════════════════════════════
# A* Pathfinding with Risk Weights
# ═══════════════════════════════════════════════════════════════

def _a_star_path(ctx: GameCtx, start: Tuple[int, int], goal: Tuple[int, int], *,
                  optimistic: bool = False, node_limit: int = A_STAR_NODE_LIMIT,
                  risk_weights: Optional[Dict[Tuple[int, int], float]] = None) -> Optional[List[str]]:
    """A* pathfinding returning path as list of direction strings."""
    if start == goal:
        return []

    frontier = [(0, start)]
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    action_from: Dict[Tuple[int, int], str] = {}
    cost_so_far: Dict[Tuple[int, int], float] = {start: 0.0}
    evaluated = 0

    best_node = start
    best_dist = _manhattan(start, goal)

    while frontier and evaluated < node_limit:
        _, curr = heapq.heappop(frontier)
        evaluated += 1

        d = _manhattan(curr, goal)
        if d < best_dist:
            best_dist = d
            best_node = curr

        if curr == goal:
            break

        for d in _legal_moves(ctx, curr, strict=False):
            nxt = _step(curr, d)
            if not optimistic and nxt not in ctx.walls:
                continue

            # Base cost + risk weight
            base_cost = 1.0
            if d == "SOUTH":
                base_cost = 50.0  # heavily penalize SOUTH
            elif d in ("EAST", "WEST"):
                base_cost = 2.0

            risk = 0.0
            if risk_weights and nxt in risk_weights:
                risk = risk_weights[nxt]
            # Southern boundary danger
            death_dist = nxt[1] - ctx.south
            if death_dist < SAFETY_MARGIN:
                risk += (SAFETY_MARGIN - death_dist) ** 2 * 10

            new_cost = cost_so_far[curr] + base_cost + risk
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + _manhattan(goal, nxt)
                heapq.heappush(frontier, (priority, nxt))
                came_from[nxt] = curr
                action_from[nxt] = d

    # Reconstruct path
    target = goal if goal in came_from else best_node
    if target == start:
        return None

    path = []
    node = target
    while node != start and node in action_from:
        path.append(action_from[node])
        node = came_from[node]
    path.reverse()
    return path if path else None


# ═══════════════════════════════════════════════════════════════
# BFS Pathfinding (for cheaper queries)
# ═══════════════════════════════════════════════════════════════

def _bfs_first_step(ctx: GameCtx, start: Tuple[int, int],
                     goal_pred: Callable[[Tuple[int, int]], bool], *,
                     max_dist: int = BFS_MAX_DIST,
                     occupied: FrozenSet[Tuple[int, int]] = frozenset(),
                     known_only: bool = False) -> Optional[str]:
    """BFS returning the first step direction toward a goal satisfying predicate."""
    if goal_pred(start):
        return "IDLE"

    parent: Dict[Tuple[int, int], Tuple[Optional[Tuple[int, int]], str]] = {start: (None, "")}
    queue = deque([(start, 0)])

    while queue:
        cell, dist = queue.popleft()
        if dist >= max_dist:
            continue
        for d in DIRS:
            if _has_wall(ctx, cell, d):
                continue
            nxt = _step(cell, d)
            if not _in_bounds(ctx, nxt):
                continue
            if nxt in parent:
                continue
            if nxt in occupied:
                continue
            if known_only and nxt not in ctx.walls:
                continue
            parent[nxt] = (cell, d)
            if goal_pred(nxt):
                # Walk back to first step
                step_dir = d
                cursor = cell
                while parent[cursor][0] is not None:
                    step_dir = parent[cursor][1]
                    cursor = parent[cursor][0]
                return step_dir
            queue.append((nxt, dist + 1))

    return None


# ═══════════════════════════════════════════════════════════════
# Danger Zone & Collision Detection
# ═══════════════════════════════════════════════════════════════

def _compute_danger(ctx: GameCtx) -> Dict[int, Set[Tuple[int, int]]]:
    """Pre-compute dangerous cells by crush power level."""
    danger: Dict[int, Set[Tuple[int, int]]] = {1: set(), 2: set(), 3: set(), 4: set()}
    for en in ctx.enemy_units:
        for p in range(1, en.rank + 1):
            danger[p].add(en.pos)
            if en.move_cd == 0:
                for d in DIRS:
                    nxt = _step(en.pos, d)
                    if _in_bounds(ctx, nxt) and not _has_wall(ctx, en.pos, d):
                        danger[p].add(nxt)
            if en.type == TYPE_FACTORY and en.jump_cd == 0:
                for d in DIRS:
                    dc, dr = DIR_DELTA[d]
                    jp = (en.col + 2 * dc, en.row + 2 * dr)
                    if _in_bounds(ctx, jp):
                        danger[p].add(jp)
    return danger


def _death_filter(ctx: GameCtx, unit: Unit, candidates: List[str],
                   reservations: Dict[Tuple[int, int], int],
                   danger: Dict[int, Set[Tuple[int, int]]]) -> List[str]:
    """Remove actions that would cause certain death."""
    survivors = []
    my_rank = unit.rank
    for act in candidates:
        if act == "IDLE":
            survivors.append(act)
            continue
        nxt = _predict_cell(unit, act)
        # Off-board check
        if act in DIRS and not _in_bounds(ctx, nxt):
            continue
        # Jump off-board check
        if act.startswith("JUMP_"):
            if not (0 <= nxt[0] < ctx.width and ctx.south <= nxt[1] <= ctx.north):
                continue
        # Danger zone check
        if nxt in danger.get(my_rank, set()):
            continue
        # Friendly reservation check
        res_rank = reservations.get(nxt)
        if res_rank is not None and res_rank >= my_rank:
            continue
        survivors.append(act)
    return survivors


# ═══════════════════════════════════════════════════════════════
# Role Assignment (FSM)
# ═══════════════════════════════════════════════════════════════

def _assign_roles(ctx: GameCtx):
    """Assign roles to units, with stickiness period."""
    roles = dict(ctx.st["roles"])
    reassess = (ctx.turn % ROLE_REASSIGN_PERIOD) == 0

    bottleneck = _has_bottleneck(ctx)
    have_nodes = bool(ctx.nodes)

    for u in ctx.my_units:
        if u.type == TYPE_FACTORY:
            roles[u.uid] = "FACTORY"
            continue
        if u.uid in roles and not reassess:
            continue
        if u.type == TYPE_SCOUT:
            roles[u.uid] = "EXPLORER"
        elif u.type == TYPE_WORKER:
            roles[u.uid] = "SAPPER" if bottleneck else "GUARD"
        elif u.type == TYPE_MINER:
            roles[u.uid] = "HARVESTER" if have_nodes else "GUARD"

    ctx.st["roles"] = roles


def _has_bottleneck(ctx: GameCtx) -> bool:
    """Check if there's a removable wall blocking the factory's path ahead."""
    f = ctx.my_factory
    if f is None:
        return False
    for r in range(f.row, min(f.row + WALL_DETOUR_THRESHOLD, ctx.north) + 1):
        val = ctx.walls.get((f.col, r))
        if val is None:
            continue
        if val & WALL_N and not _is_fixed_wall(ctx, (f.col, r), "NORTH"):
            return True
    return False


# ═══════════════════════════════════════════════════════════════
# Factory Logic
# ═══════════════════════════════════════════════════════════════

def _factory_action(ctx: GameCtx, f: Unit,
                     reservations: Dict[Tuple[int, int], int],
                     danger: Dict[int, Set[Tuple[int, int]]]) -> str:
    gap = ctx.gap
    death_buffer = SAFETY_MARGIN + (ctx.turn // 40)

    # --- P0: Emergency Survival ---
    if gap < death_buffer:
        # Jump north first if available
        if f.jump_cd == 0:
            for j_act in ("JUMP_NORTH", "JUMP_EAST", "JUMP_WEST"):
                nxt = _predict_cell(f, j_act)
                if _in_bounds(ctx, nxt) and ctx.my_factory is not None:
                    res_rank = reservations.get(nxt)
                    if res_rank is None or res_rank < CRUSH_RANK[TYPE_FACTORY]:
                        return j_act

        # Best greedy NORTH-moving safe action
        best = "IDLE"
        max_row = -1
        for d in _legal_moves(ctx, f.pos, strict=False):
            if d == "SOUTH":
                continue
            nxt = _step(f.pos, d)
            if nxt[1] > max_row and reservations.get(nxt, -1) < CRUSH_RANK[TYPE_FACTORY]:
                max_row = nxt[1]
                best = d
        if best != "IDLE":
            return best

        # Desperate: build worker to clear path
        if f.energy >= ctx.config.workerCost and f.build_cd == 0:
            spawn = (f.col, f.row + 1)
            if not _has_wall(ctx, f.pos, "NORTH") and spawn not in reservations:
                return "BUILD_WORKER"

    # --- P1: Offensive Crush via JUMP ---
    if f.jump_cd == 0:
        for j_act in ("JUMP_NORTH", "JUMP_EAST", "JUMP_WEST"):
            jp = _predict_cell(f, j_act)
            if not _in_bounds(ctx, jp):
                continue
            enemy_there = any(e.pos == jp and e.rank < CRUSH_RANK[TYPE_FACTORY]
                              for e in ctx.enemy_units)
            if enemy_there and reservations.get(jp, -1) < CRUSH_RANK[TYPE_FACTORY]:
                return j_act

    # --- P2: Evade enemy factory ---
    if ctx.enemy_factory:
        dist = _manhattan(f.pos, ctx.enemy_factory.pos)
        if dist <= ENEMY_EVADE_RADIUS:
            best_d = "IDLE"
            best_dist = dist
            for d in _legal_moves(ctx, f.pos, strict=False):
                nxt = _step(f.pos, d)
                nd = _manhattan(nxt, ctx.enemy_factory.pos)
                if nd > best_dist:
                    best_dist = nd
                    best_d = d
            if best_d != "IDLE":
                return best_d

    # --- P3: Economic Build ---
    if f.build_cd == 0:
        build = _decide_build(ctx, f, reservations)
        if build:
            return build

    # --- P4: Pathfind toward target ---
    target_row = min(f.row + 8, ctx.north)
    path = _a_star_path(ctx, f.pos, (f.col, target_row), optimistic=False)
    if path:
        nxt = _step(f.pos, path[0])
        if reservations.get(nxt, -1) < CRUSH_RANK[TYPE_FACTORY]:
            return path[0]

    # --- P5: Best safe NORTH-ish move ---
    best = "IDLE"
    best_score = -1
    for d in _legal_moves(ctx, f.pos, strict=False):
        if d == "SOUTH":
            continue
        nxt = _step(f.pos, d)
        score = nxt[1]  # prefer higher row
        if d == "NORTH":
            score += 10
        if reservations.get(nxt, -1) < CRUSH_RANK[TYPE_FACTORY] and score > best_score:
            best_score = score
            best = d
    return best


def _decide_build(ctx: GameCtx, f: Unit,
                   reservations: Dict[Tuple[int, int], int]) -> Optional[str]:
    """Decide what to build based on game state."""
    spawn = (f.col, f.row + 1)
    if not _in_bounds(ctx, spawn):
        return None
    if _has_wall(ctx, f.pos, "NORTH"):
        return None
    if spawn in reservations:
        return None
    if ctx.config.episodeSteps - ctx.turn < LATE_GAME_STOP_BUILD:
        return None

    cfg = ctx.config
    my = ctx.my_units
    n_scouts = sum(1 for u in my if u.type == TYPE_SCOUT)
    n_workers = sum(1 for u in my if u.type == TYPE_WORKER)
    n_miners = sum(1 for u in my if u.type == TYPE_MINER)
    have_nodes = bool(ctx.nodes)
    min_energy = 400

    if n_scouts == 0 and f.energy >= cfg.scoutCost + min_energy:
        return "BUILD_SCOUT"
    if have_nodes and n_miners < MINER_MAX_COUNT and f.energy >= cfg.minerCost + min_energy:
        return "BUILD_MINER"
    if n_workers < WORKER_MAX_COUNT and f.energy >= cfg.workerCost + min_energy:
        if _has_bottleneck(ctx) or n_scouts >= 1:
            return "BUILD_WORKER"
    if n_scouts < SCOUT_MAX_COUNT and f.energy >= cfg.scoutCost + min_energy:
        return "BUILD_SCOUT"
    if n_miners < MINER_MAX_COUNT and have_nodes and f.energy >= cfg.minerCost + min_energy:
        return "BUILD_MINER"
    if n_workers < WORKER_MAX_COUNT and f.energy >= cfg.workerCost + min_energy + 200:
        return "BUILD_WORKER"

    return None


# ═══════════════════════════════════════════════════════════════
# Scout Logic (EXPLORER)
# ═══════════════════════════════════════════════════════════════

def _scout_action(ctx: GameCtx, u: Unit,
                   reservations: Dict[Tuple[int, int], int],
                   danger: Dict[int, Set[Tuple[int, int]]]) -> str:
    f = ctx.my_factory

    # Energy transfer if near factory
    if f and u.energy > 80 and _manhattan(u.pos, f.pos) == 1:
        d = _direction_toward(u.pos, f.pos)
        if d and not _has_wall(ctx, u.pos, d):
            return f"TRANSFER_{d}"

    # Return to factory if starving
    if u.energy < 15 and f:
        p = _bfs_first_step(ctx, u.pos, lambda c: c == f.pos, max_dist=40)
        if p:
            return p

    # Collect crystals (score = energy / dist²)
    best_pos, best_score = None, 0.0
    for pos, energy in ctx.crystals.items():
        if pos in reservations:
            continue
        d = _manhattan(u.pos, pos)
        if d == 0:
            return "IDLE"  # already on crystal
        score = energy / (d * d + 1)
        if score > best_score:
            best_score = score
            best_pos = pos

    if best_pos:
        p = _bfs_first_step(ctx, u.pos, lambda c: c == best_pos,
                            max_dist=CRYSTAL_DETOUR_BUDGET + 10)
        if p:
            return p

    # Explore undiscovered cells
    frontier_cells = []
    for dc in range(-5, 6):
        for dr in range(0, 11):  # bias NORTH
            probe = (u.col + dc, u.row + dr)
            if not _in_bounds(ctx, probe):
                continue
            if probe not in ctx.walls and probe not in reservations:
                frontier_cells.append(probe)

    if frontier_cells:
        frontier_cells.sort(key=lambda c: _manhattan(u.pos, c) - c[1] * 2)
        for fc in frontier_cells[:3]:
            p = _bfs_first_step(ctx, u.pos, lambda c: c == fc, max_dist=30)
            if p:
                return p

    # General northward movement
    for d in DIRS:
        if d == "SOUTH":
            continue
        if not _has_wall(ctx, u.pos, d):
            nxt = _step(u.pos, d)
            if _in_bounds(ctx, nxt):
                return d

    return "IDLE"


# ═══════════════════════════════════════════════════════════════
# Worker Logic (SAPPER / GUARD)
# ═══════════════════════════════════════════════════════════════

def _worker_action(ctx: GameCtx, u: Unit,
                    reservations: Dict[Tuple[int, int], int],
                    danger: Dict[int, Set[Tuple[int, int]]]) -> str:
    f = ctx.my_factory

    # Energy transfer if full
    if f and u.energy > 250 and _manhattan(u.pos, f.pos) == 1:
        d = _direction_toward(u.pos, f.pos)
        if d and not _has_wall(ctx, u.pos, d):
            return f"TRANSFER_{d}"

    # Return to factory if low energy
    if u.energy < 50 and f:
        p = _bfs_first_step(ctx, u.pos, lambda c: c == f.pos, max_dist=40)
        if p:
            return p

    # Wall removal: clear path ahead of factory
    if f and u.energy >= ctx.config.wallRemoveCost:
        # Check walls in factory's column ahead
        for r in range(f.row, min(f.row + 4, ctx.north)):
            cell = (f.col, r)
            if cell in ctx.walls:
                val = ctx.walls[cell]
                if val & WALL_N and not _is_fixed_wall(ctx, cell, "NORTH"):
                    # Move to cell and remove north wall
                    if u.pos == cell:
                        return "REMOVE_NORTH"
                    p = _bfs_first_step(ctx, u.pos, lambda c: c == cell, max_dist=10)
                    if p:
                        return p

        # Also check lateral walls blocking factory
        for c_off in (-1, 1):
            nc = f.col + c_off
            if 0 <= nc < ctx.width:
                cell = (nc, f.row)
                if cell in ctx.walls:
                    val = ctx.walls[cell]
                    check_dir = "WEST" if c_off == -1 else "EAST"
                    if val & DIR_TO_BIT["NORTH"] and not _is_fixed_wall(ctx, cell, "NORTH"):
                        if u.pos == cell:
                            return "REMOVE_NORTH"
                        p = _bfs_first_step(ctx, u.pos, lambda c: c == cell, max_dist=10)
                        if p:
                            return p

    # GUARD: escort factory
    if f:
        escort = (f.col, f.row + 1)
        if _in_bounds(ctx, escort) and u.pos != escort:
            p = _bfs_first_step(ctx, u.pos, lambda c: c == escort, max_dist=10)
            if p:
                return p

    # General northward
    for d in DIRS:
        if d == "SOUTH":
            continue
        if not _has_wall(ctx, u.pos, d):
            nxt = _step(u.pos, d)
            if _in_bounds(ctx, nxt):
                return d
    return "IDLE"


# ═══════════════════════════════════════════════════════════════
# Miner Logic (HARVESTER)
# ═══════════════════════════════════════════════════════════════

def _miner_action(ctx: GameCtx, u: Unit,
                   reservations: Dict[Tuple[int, int], int],
                   danger: Dict[int, Set[Tuple[int, int]]]) -> str:
    f = ctx.my_factory

    # Transform if on a mining node
    if u.pos in ctx.nodes and u.pos not in {(c, r) for (c, r), _ in ctx.mines.items()}:
        if u.energy >= ctx.config.transformCost:
            return "TRANSFORM"

    # Energy transfer if full
    if f and u.energy > 400 and _manhattan(u.pos, f.pos) == 1:
        d = _direction_toward(u.pos, f.pos)
        if d and not _has_wall(ctx, u.pos, d):
            return f"TRANSFER_{d}"

    # Seek nearest mining node
    best_node, best_dist = None, MINER_NODE_SEARCH_LIMIT + 1
    for node in ctx.nodes:
        if node in reservations:
            continue
        d = _manhattan(u.pos, node)
        if d < best_dist:
            best_dist = d
            best_node = node

    if best_node:
        p = _bfs_first_step(ctx, u.pos, lambda c: c == best_node, max_dist=MINER_NODE_SEARCH_LIMIT)
        if p:
            return p

    # Return to factory if no nodes reachable
    if f and _manhattan(u.pos, f.pos) > 5:
        p = _bfs_first_step(ctx, u.pos, lambda c: c == f.pos, max_dist=30)
        if p:
            return p

    # General northward
    for d in DIRS:
        if d == "SOUTH":
            continue
        if not _has_wall(ctx, u.pos, d):
            nxt = _step(u.pos, d)
            if _in_bounds(ctx, nxt):
                return d
    return "IDLE"


def _direction_toward(src: Tuple[int, int], dst: Tuple[int, int]) -> Optional[str]:
    """Get first direction from src toward dst."""
    dc = dst[0] - src[0]
    dr = dst[1] - src[1]
    if abs(dc) >= abs(dr):
        return "EAST" if dc > 0 else "WEST" if dc < 0 else "NORTH" if dr > 0 else "SOUTH" if dr < 0 else None
    else:
        return "NORTH" if dr > 0 else "SOUTH" if dr < 0 else "EAST" if dc > 0 else "WEST" if dc < 0 else None


# ═══════════════════════════════════════════════════════════════
# Main Agent Entry Point
# ═══════════════════════════════════════════════════════════════

def agent(obs, config):
    """Main agent function — Kaggle entry point."""
    # Update persistent state
    _update_state(obs, config)

    # Build turn context
    ctx = _build_ctx(obs, config)

    # Assign roles
    _assign_roles(ctx)

    # Pre-compute danger zones
    danger = _compute_danger(ctx)

    # Process units in crush-rank order (highest first)
    actions: Dict[str, str] = {}
    reservations: Dict[Tuple[int, int], int] = {}

    my_units_sorted = sorted(ctx.my_units, key=lambda u: -u.rank)

    for u in my_units_sorted:
        try:
            if u.type == TYPE_FACTORY:
                act = _factory_action(ctx, u, reservations, danger)
            else:
                # Get raw candidate
                if u.type == TYPE_SCOUT:
                    raw = _scout_action(ctx, u, reservations, danger)
                elif u.type == TYPE_WORKER:
                    raw = _worker_action(ctx, u, reservations, danger)
                elif u.type == TYPE_MINER:
                    raw = _miner_action(ctx, u, reservations, danger)
                else:
                    raw = "IDLE"

                # Apply death filter
                candidates = [raw] if raw == "IDLE" else [raw] + [
                    d for d in _legal_moves(ctx, u.pos, strict=True)
                    if d != raw and d != "SOUTH"
                ]
                filtered = _death_filter(ctx, u, candidates, reservations, danger)
                act = filtered[0] if filtered else "IDLE"

            # Record reservation
            nxt = _predict_cell(u, act)
            if nxt not in reservations:
                reservations[nxt] = u.rank

            actions[u.uid] = act

        except Exception:
            actions[u.uid] = "NORTH"
            nxt = _predict_cell(u, "NORTH")
            if nxt not in reservations:
                reservations[nxt] = u.rank

    # Track factory position for stuck detection
    if ctx.my_factory and ctx.my_factory.uid in actions:
        prev = ctx.st.get("last_factory_pos")
        curr = ctx.my_factory.pos
        if prev and prev == curr:
            ctx.st["factory_stuck"] = ctx.st.get("factory_stuck", 0) + 1
        else:
            ctx.st["factory_stuck"] = 0
        ctx.st["last_factory_pos"] = curr

    return actions
