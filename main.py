"""Maze Crawler Agent v2 — Risk-Aware A* + Combat Micro + Wolf-Pack Trapping.

A high-performance agent combining the best strategies:
- Risk matrix (numpy) for boundary + enemy avoidance
- A* pathfinding with asymmetric cost weights (NORTH cheap, SOUTH expensive)
- Combat micro (offensive crushing + tactical kiting)
- Wolf-pack trapping (workers predict & wall-in enemy factory)
- Dynamic economic scaling with aggressive build caps
- Centralized safety checking (walls, bounds, collisions, danger zones)
- Persistent wall/mine/node memory across turns
"""

import random
import heapq
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple

# ═══════════════════════════════════════
# Constants
# ═══════════════════════════════════════

NORTH, EAST, SOUTH, WEST = 1, 2, 4, 8
A_NORTH = "NORTH"; A_EAST = "EAST"; A_SOUTH = "SOUTH"; A_WEST = "WEST"; A_IDLE = "IDLE"
T_FACTORY, T_SCOUT, T_WORKER, T_MINER = 0, 1, 2, 3

CRUSH = {T_FACTORY: 4, T_MINER: 3, T_WORKER: 2, T_SCOUT: 1}

DIR_BIT = {"NORTH": 1, "EAST": 2, "SOUTH": 4, "WEST": 8}
DIR_DELTA = {"NORTH": (0, 1), "EAST": (1, 0), "SOUTH": (0, -1), "WEST": (-1, 0)}
DIRS = ("NORTH", "EAST", "SOUTH", "WEST")

# ═══════════════════════════════════════
# Robot Data (slots for speed)
# ═══════════════════════════════════════

class Robot:
    __slots__ = ('uid', 'type', 'col', 'row', 'energy', 'owner',
                 'move_cd', 'jump_cd', 'build_cd', 'power')
    def __init__(self, uid: str, data: List[int]):
        self.uid = uid
        self.type = data[0]
        self.col, self.row = data[1], data[2]
        self.energy = data[3]
        self.owner = data[4]
        self.move_cd = data[5] if len(data) > 5 else 0
        self.jump_cd = data[6] if len(data) > 6 else 0
        self.build_cd = data[7] if len(data) > 7 else 0
        self.power = CRUSH.get(self.type, 0)

    @property
    def pos(self) -> Tuple[int, int]:
        return (self.col, self.row)


# ═══════════════════════════════════════
# Game State (persistent across turns)
# ═══════════════════════════════════════

class GameState:
    def __init__(self, config: Any):
        self.config = config
        self.width = config.width
        self.height = config.height
        self.walls: Dict[Tuple[int, int], int] = {}
        self.nodes: Set[Tuple[int, int]] = set()
        self.mines: Dict[Tuple[int, int], List[int]] = {}
        self.crystals: Dict[Tuple[int, int], int] = {}
        self.my: Dict[str, Robot] = {}
        self.enemy: Dict[str, Robot] = {}
        self.step = 0
        self.player = 0
        self.south = 0
        self.north = 0
        self.risk = np.zeros((1000, self.width), dtype=np.float32)
        self.enemy_history: Dict[str, Tuple[int, int]] = {}
        self.path_cache: Dict[Tuple, List[str]] = {}

    def update(self, obs: Any):
        self.step = obs.step
        self.player = obs.player
        self.south = obs.southBound
        self.north = obs.northBound
        w = self.width

        self.crystals.clear()
        self.my.clear()
        self.enemy.clear()
        self.risk.fill(0)
        self.path_cache.clear()

        # Parse robots
        cur_enemy = {}
        for uid, data in obs.robots.items():
            r = Robot(uid, data)
            if r.owner == self.player:
                self.my[uid] = r
            else:
                self.enemy[uid] = r
                cur_enemy[uid] = r.pos

        # Risk matrix: boundary danger
        rows = np.arange(self.south, self.north + 1)
        buffer = 12 + (self.step // 40)
        penalties = np.maximum(0, buffer - (rows - self.south)) ** 2
        valid = rows[rows < 1000]
        self.risk[valid, :] = penalties[:len(valid), np.newaxis]

        # Enemy proximity risk
        for en in self.enemy.values():
            if en.type == T_FACTORY:
                r0, r1 = max(0, en.row - 3), min(999, en.row + 4)
                c0, c1 = max(0, en.col - 3), min(w, en.col + 4)
                for rr in range(r0, r1):
                    for cc in range(c0, c1):
                        dist = abs(rr - en.row) + abs(cc - en.col)
                        if dist <= 3:
                            self.risk[rr, cc] += (4 - dist) * 15
            else:
                r0, r1 = max(0, en.row - 1), min(999, en.row + 2)
                c0, c1 = max(0, en.col - 1), min(w, en.col + 2)
                for rr in range(r0, r1):
                    for cc in range(c0, c1):
                        self.risk[rr, cc] += en.power * 5

        self.enemy_history = cur_enemy

        # Parse walls
        for row in range(self.south, self.north + 1):
            base = (row - self.south) * w
            for col in range(w):
                val = obs.walls[base + col]
                if val != -1:
                    self.walls[(col, row)] = val

        # Parse crystals
        for s, e in obs.crystals.items():
            c, r = map(int, s.split(','))
            self.crystals[(c, r)] = e

        # Parse mining nodes
        for s in obs.miningNodes:
            c, r = map(int, s.split(','))
            if r >= self.south:
                self.nodes.add((c, r))

        # Parse mines
        for s, data in obs.mines.items():
            c, r = map(int, s.split(','))
            if r >= self.south:
                self.mines[(c, r)] = data

    def in_bounds(self, pos: Tuple[int, int]) -> bool:
        c, r = pos
        return 0 <= c < self.width and self.south <= r <= self.north

    def step_pos(self, pos: Tuple[int, int], d: str) -> Tuple[int, int]:
        dc, dr = DIR_DELTA[d]
        return (pos[0] + dc, pos[1] + dr)

    def manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def has_wall(self, pos: Tuple[int, int], d: str) -> bool:
        val = self.walls.get(pos)
        return val is not None and bool(val & DIR_BIT[d])

    def get_neighbors(self, pos: Tuple[int, int],
                       optimistic: bool = False) -> List[Tuple[Tuple[int, int], str]]:
        c, r = pos
        nbrs = []
        w = self.walls.get(pos, -1)
        if w == -1:
            if not optimistic:
                return []
            w = 0
        if not (w & NORTH) and r + 1 <= self.north:
            nbrs.append(((c, r + 1), A_NORTH))
        if not (w & EAST) and c + 1 < self.width:
            nbrs.append(((c + 1, r), A_EAST))
        if not (w & SOUTH) and r - 1 >= self.south:
            nbrs.append(((c, r - 1), A_SOUTH))
        if not (w & WEST) and c - 1 >= 0:
            nbrs.append(((c - 1, r), A_WEST))
        return nbrs


# ═══════════════════════════════════════
# A* Pathfinding with Risk Weights
# ═══════════════════════════════════════

def astar_path(gs: GameState, start: Tuple[int, int], goal: Tuple[int, int], *,
               optimistic: bool = False,
               limit: int = 600) -> Optional[List[str]]:
    if start == goal:
        return []
    cache_key = (start, goal)
    if not optimistic and cache_key in gs.path_cache:
        return gs.path_cache[cache_key]

    frontier = [(0, start)]
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    act_from: Dict[Tuple[int, int], str] = {}
    cost_sofar: Dict[Tuple[int, int], float] = {start: 0.0}
    evals = 0

    best_node, best_dist = start, gs.manhattan(start, goal)

    while frontier and evals < limit:
        _, curr = heapq.heappop(frontier)
        evals += 1

        d = gs.manhattan(curr, goal)
        if d < best_dist:
            best_dist, best_node = d, curr

        if curr == goal:
            break

        c, r = curr
        w = gs.walls.get(curr, -1)
        if w == -1:
            if not optimistic:
                continue
            w = 0

        for bit, dc, dr, act, base_cost in [
            (NORTH, 0, 1, A_NORTH, 1),
            (EAST, 1, 0, A_EAST, 3),
            (SOUTH, 0, -1, A_SOUTH, 100),
            (WEST, -1, 0, A_WEST, 3),
        ]:
            if not (w & bit):
                nc, nr = c + dc, r + dr
                if 0 <= nc < gs.width and gs.south <= nr <= gs.north:
                    nxt = (nc, nr)
                    risk_val = float(gs.risk[nr, nc]) if nr < 1000 else 0.0
                    new_cost = cost_sofar[curr] + base_cost + risk_val
                    if nxt not in cost_sofar or new_cost < cost_sofar[nxt]:
                        cost_sofar[nxt] = new_cost
                        priority = new_cost + gs.manhattan(goal, nxt)
                        heapq.heappush(frontier, (priority, nxt))
                        came_from[nxt] = curr
                        act_from[nxt] = act

    target = goal if goal in came_from else best_node
    if target == start:
        return None

    path = []
    node = target
    while node != start:
        path.append(act_from[node])
        node = came_from[node]
    path.reverse()

    if not optimistic and target == goal:
        gs.path_cache[cache_key] = path
    return path


# ═══════════════════════════════════════
# Dispatcher — orchestrates all unit actions
# ═══════════════════════════════════════

class Dispatcher:
    def __init__(self, gs: GameState):
        self.gs = gs
        self.danger: Dict[int, Set[Tuple[int, int]]] = {}
        self._compute_danger()

    def _compute_danger(self):
        """Pre-compute cells dangerous to each power level."""
        self.danger = {1: set(), 2: set(), 3: set(), 4: set()}
        for en in self.gs.enemy.values():
            for p in range(1, en.power + 1):
                self.danger[p].add(en.pos)
                if en.move_cd == 0:
                    for d in DIRS:
                        nxt = self.gs.step_pos(en.pos, d)
                        if self.gs.in_bounds(nxt) and not self.gs.has_wall(en.pos, d):
                            self.danger[p].add(nxt)
                if en.type == T_FACTORY and en.jump_cd == 0:
                    for d in DIRS:
                        dc, dr = DIR_DELTA[d]
                        jp = (en.col + 2 * dc, en.row + 2 * dr)
                        if self.gs.in_bounds(jp):
                            self.danger[p].add(jp)

    def predict(self, r: Robot, action: str) -> Tuple[int, int]:
        if action in DIRS:
            return self.gs.step_pos(r.pos, action)
        if action.startswith("JUMP_"):
            d = action[5:]
            dc, dr = DIR_DELTA[d]
            return (r.col + 2 * dc, r.row + 2 * dr)
        if action in ("BUILD_SCOUT", "BUILD_WORKER", "BUILD_MINER"):
            return (r.col, r.row + 1)
        return r.pos

    def is_safe(self, r: Robot, action: str,
                reserved: Set[Tuple[int, int]]) -> Tuple[bool, Tuple[int, int]]:
        nxt = self.predict(r, action)

        # Out of bounds
        if not self.gs.in_bounds(nxt):
            return False, r.pos

        # Wall check for simple moves
        if action in DIRS:
            w = self.gs.walls.get(r.pos, 0)
            if action == A_NORTH and (w & 1):
                return False, r.pos
            if action == A_EAST and (w & 2):
                return False, r.pos
            if action == A_SOUTH and (w & 4):
                return False, r.pos
            if action == A_WEST and (w & 8):
                return False, r.pos

        # Already reserved by another unit
        if nxt in reserved:
            return False, r.pos

        # Danger zone
        if nxt in self.danger.get(r.power, set()):
            return False, r.pos

        return True, nxt

    def disp(self) -> Dict[str, str]:
        actions: Dict[str, str] = {}
        reserved: Set[Tuple[int, int]] = set()
        intent: Set[Tuple[int, int]] = set()

        robots = sorted(self.gs.my.values(), key=lambda x: -x.power)
        factory = next((r for r in robots if r.type == T_FACTORY), None)

        for r in robots:
            try:
                if r.type == T_FACTORY:
                    act = self._factory(r)
                elif r.type == T_SCOUT:
                    act = self._scout(r, factory, intent)
                elif r.type == T_WORKER:
                    act = self._worker(r, factory, intent)
                else:
                    act = self._miner(r, intent)
            except Exception:
                act = A_NORTH

            safe, nxt = self.is_safe(r, act, reserved)
            if not safe:
                # Fallback: find best safe alternative
                death_dist = r.row - self.gs.south
                order = {A_NORTH: 0, A_EAST: 1, A_WEST: 1, A_IDLE: 2, A_SOUTH: 3}
                cands = self.gs.get_neighbors(r.pos)
                if death_dist < 5:
                    cands = [c for c in cands if c[1] != A_SOUTH]
                found = False
                for _, cand in sorted(cands, key=lambda x: order.get(x[1], 4)):
                    s, n = self.is_safe(r, cand, reserved)
                    if s:
                        act, nxt, found = cand, n, True
                        break
                if not found:
                    act, nxt = A_IDLE, r.pos

            actions[r.uid] = act
            reserved.add(nxt)

        return actions

    # ─── Combat Micro ───────────────────────────────────────

    def _combat_micro(self, r: Robot) -> Optional[str]:
        gs = self.gs
        # Offensive: crush weaker adjacent enemies
        for nxt, act in gs.get_neighbors(r.pos):
            enemy = next((e for e in gs.enemy.values() if e.pos == nxt), None)
            if enemy and r.power > enemy.power:
                s, _ = self.is_safe(r, act, set())
                if s:
                    return act

        # Defensive: kite away from stronger enemies within 2 tiles
        hostiles = [e for e in gs.enemy.values()
                     if e.power > r.power and gs.manhattan(r.pos, e.pos) <= 2]
        if hostiles:
            best = A_IDLE
            best_min = min(gs.manhattan(r.pos, h.pos) for h in hostiles)
            for _, act in gs.get_neighbors(r.pos):
                s, nxt = self.is_safe(r, act, set())
                if s:
                    nd = min(gs.manhattan(nxt, h.pos) for h in hostiles)
                    if nd > best_min:
                        best_min, best = nd, act
            if best != A_IDLE:
                return best
        return None

    # ─── Energy Transfer ────────────────────────────────────

    def _transfer(self, r: Robot, target: Optional[Robot]) -> Optional[str]:
        if not target:
            return None
        if self.gs.manhattan(r.pos, target.pos) != 1:
            return None
        w = self.gs.walls.get(r.pos, 0)
        if target.row > r.row and not (w & 1):
            return "TRANSFER_NORTH"
        if target.row < r.row and not (w & 4):
            return "TRANSFER_SOUTH"
        if target.col > r.col and not (w & 2):
            return "TRANSFER_EAST"
        if target.col < r.col and not (w & 8):
            return "TRANSFER_WEST"
        return None

    # ─── Collection (crystals + mines + exploration) ────────

    def _collect(self, r: Robot, intent: Set[Tuple[int, int]],
                  max_ratio: float = 0.9) -> Optional[str]:
        gs = self.gs
        max_e = 100 if r.type == 1 else 300 if r.type == 2 else 500
        if r.energy > max_e * max_ratio:
            return None

        # Crystals
        avail = {pos: e for pos, e in gs.crystals.items() if pos not in intent}
        # Friendly mines with energy
        m_avail = {pos: d[0] for pos, d in gs.mines.items()
                    if d[2] == gs.player and d[0] > 0 and pos not in intent}

        targets = []
        for pos, e in avail.items():
            d = gs.manhattan(r.pos, pos)
            targets.append((pos, e / (d * d + 1)))
        for pos, e in m_avail.items():
            d = gs.manhattan(r.pos, pos)
            targets.append((pos, min(e, 100) / (d * d + 1)))

        if targets:
            targets.sort(key=lambda x: -x[1])
            best = targets[0][0]
            intent.add(best)
            if r.pos == best:
                return A_IDLE
            path = astar_path(gs, r.pos, best, limit=300)
            if path:
                return path[0]

        # Explore undiscovered cells (north-biased)
        undiscovered = []
        for rr in range(max(gs.south + 1, r.row - 2),
                         min(gs.north + 1, r.row + 10)):
            for cc in range(gs.width):
                if (cc, rr) not in gs.walls and (cc, rr) not in intent:
                    undiscovered.append((cc, rr))
        if undiscovered:
            undiscovered.sort(key=lambda c: gs.manhattan(r.pos, c))
            idx = hash(r.uid) % min(len(undiscovered), 5)
            closest = undiscovered[min(len(undiscovered) - 1, idx)]
            intent.add(closest)
            path = astar_path(gs, r.pos, closest, limit=150)
            if path:
                return path[0]
        return None

    # ─── Factory Logic ──────────────────────────────────────

    def _factory(self, r: Robot) -> str:
        gs = self.gs
        death = r.row - gs.south
        buffer = 12 + (gs.step // 40)

        # ── P0: Panic Escape ──
        if death < buffer:
            if r.jump_cd == 0:
                for j in ("JUMP_NORTH", "JUMP_EAST", "JUMP_WEST"):
                    s, _ = self.is_safe(r, j, set())
                    if s:
                        return j
                s, _ = self.is_safe(r, A_NORTH, set())
                if s:
                    return A_NORTH

            # Escape pathfinding
            for tr in (r.row + 10, r.row + 5):
                for tc in (r.col, gs.width // 2, 0, gs.width - 1):
                    tgt = (max(0, min(gs.width - 1, tc)), min(gs.north, tr))
                    if tgt[1] > r.row:
                        p = astar_path(gs, r.pos, tgt, limit=1000)
                        if p and p[0] != A_SOUTH:
                            return p[0]

            # Emergency: build worker or scout to break walls / explore
            if r.build_cd == 0:
                s_north, _ = self.is_safe(r, A_NORTH, set())
                if not s_north:
                    if r.energy >= 200:
                        return "BUILD_WORKER"
                    elif r.energy >= 50:
                        return "BUILD_SCOUT"

            # Greedy safe NORTH push
            best, max_r = A_NORTH, -1
            for _, act in gs.get_neighbors(r.pos):
                if act == A_SOUTH:
                    continue
                s, nxt = self.is_safe(r, act, set())
                if s and nxt[1] > max_r:
                    max_r, best = nxt[1], act
            return best

        # ── P1: Offensive Jump Crush ──
        if r.jump_cd == 0:
            for j in ("JUMP_NORTH", "JUMP_EAST", "JUMP_WEST"):
                jp = self.predict(r, j)
                if gs.in_bounds(jp):
                    enemy_there = any(e.pos == jp and e.power < r.power
                                      for e in gs.enemy.values())
                    if enemy_there:
                        s, _ = self.is_safe(r, j, set())
                        if s:
                            return j

        # ── P2: Evade Enemy Factory ──
        ef = next((e for e in gs.enemy.values() if e.type == T_FACTORY), None)
        if ef and gs.manhattan(r.pos, ef.pos) <= 3:
            best_d, best_dist = A_IDLE, gs.manhattan(r.pos, ef.pos)
            if r.jump_cd == 0:
                for j in ("JUMP_NORTH", "JUMP_EAST", "JUMP_WEST"):
                    jp = self.predict(r, j)
                    if gs.in_bounds(jp):
                        nd = gs.manhattan(jp, ef.pos)
                        if nd > best_dist:
                            best_dist, best_d = nd, j
                return best_d
            for _, act in gs.get_neighbors(r.pos):
                s, nxt = self.is_safe(r, act, set())
                if s:
                    nd = gs.manhattan(nxt, ef.pos)
                    if nd > best_dist:
                        best_dist, best_d = nd, act
            if best_d != A_IDLE:
                return best_d

        # ── P3: Economic Build ──
        if r.build_cd == 0:
            b = self._build(r)
            if b:
                return b

        # ── P4: Centering ──
        if death >= buffer:
            tc = gs.width // 2
            if abs(r.col - tc) > 1:
                p = astar_path(gs, r.pos, (tc, r.row))
                if p:
                    return p[0]

        # ── P5: Proactive North Progression ──
        tgt = (r.col, min(r.row + 3, gs.north))
        p = astar_path(gs, r.pos, tgt, optimistic=True)
        if p:
            s, _ = self.is_safe(r, p[0], set())
            if s:
                return p[0]

        # Last resort: if we can move north, do it
        s_n, _ = self.is_safe(r, A_NORTH, set())
        if s_n:
            return A_NORTH
        return A_IDLE

    def _build(self, f: Robot) -> Optional[str]:
        gs = self.gs
        cfg = gs.config
        spawn = (f.col, f.row + 1)
        if not gs.in_bounds(spawn) or gs.has_wall(f.pos, "NORTH"):
            return None

        counts = {t: sum(1 for r in gs.my.values() if r.type == t)
                   for t in (1, 2, 3)}

        # Phase 1: Core trifecta
        if counts[1] == 0 and f.energy >= cfg.scoutCost + 200:
            return "BUILD_SCOUT"
        if counts[2] == 0 and f.energy >= cfg.workerCost + 250:
            return "BUILD_WORKER"
        if counts[3] == 0 and f.energy >= cfg.minerCost + 300:
            return "BUILD_MINER"

        # Phase 2: Dynamic scaling
        min_e = min(800, 400 + len(gs.my) * 50)
        if f.energy > min_e:
            max_miners = 4 if f.energy > 600 else 2
            max_workers = 5 if f.energy > 700 else 3
            max_scouts = 6 if f.energy > 500 else 4

            if counts[3] < max_miners and f.energy > 400:
                return "BUILD_MINER"
            if counts[2] < max_workers and f.energy > 400:
                return "BUILD_WORKER"
            if counts[1] < max_scouts and f.energy > 200:
                return "BUILD_SCOUT"
        return None

    # ─── Scout Logic ────────────────────────────────────────

    def _scout(self, r: Robot, f: Optional[Robot],
                intent: Set[Tuple[int, int]]) -> str:
        gs = self.gs

        t = self._transfer(r, f)
        if t:
            return t

        c = self._combat_micro(r)
        if c:
            return c

        # Return to base if full or starving
        if (r.energy > 80 or r.energy < 30) and f:
            if gs.manhattan(r.pos, f.pos) > 1:
                p = astar_path(gs, r.pos, f.pos)
                if p:
                    return p[0]

        # Collect
        act = self._collect(r, intent)
        if act:
            return act

        # Northward exploration
        tgt = (r.col, min(r.row + 8, gs.north))
        p = astar_path(gs, r.pos, tgt, optimistic=True)
        return p[0] if p else A_NORTH

    # ─── Worker Logic ───────────────────────────────────────

    def _worker(self, r: Robot, f: Optional[Robot],
                 intent: Set[Tuple[int, int]]) -> str:
        gs = self.gs

        t = self._transfer(r, f)
        if t:
            return t

        c = self._combat_micro(r)
        if c:
            return c

        # ── Wolf-pack: Predict & trap enemy factory ──
        for uid, en in gs.enemy.items():
            if en.type == T_FACTORY:
                prev = gs.enemy_history.get(uid, en.pos)
                dc = en.col - prev[0]
                dr = en.row - prev[1]
                pred = (max(0, min(gs.width - 1, en.col + dc)),
                        max(gs.south + 1, min(gs.north, en.row + dr)))

                traps = [(pred[0], pred[1] + 1), (pred[0], pred[1] - 1),
                         (pred[0] + 1, pred[1]), (pred[0] - 1, pred[1])]

                for spot in traps:
                    if not gs.in_bounds(spot) or spot in intent:
                        continue
                    dist = gs.manhattan(r.pos, spot)
                    if dist <= 1:
                        intent.add(spot)
                        if r.pos == spot:
                            if r.row == pred[1] + 1:
                                return "BUILD_SOUTH"
                            if r.row == pred[1] - 1:
                                return "BUILD_NORTH"
                            if r.col == pred[0] + 1:
                                return "BUILD_WEST"
                            if r.col == pred[0] - 1:
                                return "BUILD_EAST"
                            return "BUILD_NORTH"
                    elif dist <= 5:
                        intent.add(spot)
                        p = astar_path(gs, r.pos, spot)
                        if p:
                            return p[0]

        # Return to base
        if (r.energy > 250 or r.energy < 60) and f:
            if gs.manhattan(r.pos, f.pos) > 1:
                p = astar_path(gs, r.pos, f.pos)
                if p:
                    return p[0]

        # Wall removal for factory path
        if r.energy > 150 and f:
            for dr, dc, dir_name in [
                (1, 0, "REMOVE_NORTH"), (-1, 0, "REMOVE_SOUTH"),
                (0, 1, "REMOVE_EAST"), (0, -1, "REMOVE_WEST"),
            ]:
                tp = (r.col + dc, r.row + dr)
                is_fp = (f.col == tp[0] and tp[1] > f.row and tp[1] <= f.row + 4)
                near_f = gs.manhattan(tp, f.pos) < gs.manhattan(r.pos, f.pos)
                if is_fp or near_f:
                    w = gs.walls.get(r.pos, 0)
                    bit = {"NORTH": 1, "EAST": 2, "SOUTH": 4, "WEST": 8}[
                        dir_name.split("_")[1]
                    ]
                    if w & bit:
                        return dir_name

        # Collect
        act = self._collect(r, intent, max_ratio=0.85)
        if act:
            return act

        tgt = (r.col, min(r.row + 3, gs.north))
        p = astar_path(gs, r.pos, tgt, optimistic=True)
        return p[0] if p else A_NORTH

    # ─── Miner Logic ────────────────────────────────────────

    def _miner(self, r: Robot, intent: Set[Tuple[int, int]]) -> str:
        gs = self.gs

        c = self._combat_micro(r)
        if c:
            return c

        # Transform on node
        is_our_mine = r.pos in gs.mines and gs.mines[r.pos][2] == gs.player
        if r.pos in gs.nodes and not is_our_mine and r.energy >= 100:
            return "TRANSFORM"

        # Seek nearest mining node
        targets = [n for n in gs.nodes if n not in gs.mines and n not in intent]
        if targets:
            closest = min(targets, key=lambda n: gs.manhattan(r.pos, n))
            intent.add(closest)
            p = astar_path(gs, r.pos, closest)
            if p:
                return p[0]

        # Return energy to factory
        f = next((r for r in gs.my.values() if r.type == T_FACTORY), None)
        if r.energy > 400 and f:
            if gs.manhattan(r.pos, f.pos) > 1:
                p = astar_path(gs, r.pos, f.pos)
                if p:
                    return p[0]
            t = self._transfer(r, f)
            if t:
                return t

        # Collect
        act = self._collect(r, intent, max_ratio=0.85)
        if act:
            return act

        # Stay near factory
        if f and gs.manhattan(r.pos, f.pos) > 3:
            p = astar_path(gs, r.pos, f.pos)
            if p:
                return p[0]

        return A_IDLE


# ═══════════════════════════════════════
# Kaggle Entry Point
# ═══════════════════════════════════════

_gs: Optional[GameState] = None

def agent(obs: Any, config: Any) -> Dict[str, str]:
    global _gs
    try:
        if _gs is None:
            _gs = GameState(config)
            if hasattr(config, 'randomSeed') and config.randomSeed:
                random.seed(config.randomSeed)
        _gs.update(obs)
        return Dispatcher(_gs).disp()
    except Exception:
        fb: Dict[str, str] = {}
        if obs and hasattr(obs, 'robots'):
            for uid, data in obs.robots.items():
                if data[4] == obs.player:
                    fb[uid] = A_NORTH if data[0] == T_FACTORY else A_IDLE
        return fb
