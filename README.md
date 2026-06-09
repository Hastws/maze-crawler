# Maze Crawler — Kaggle Competition Agent

A high-performance agent for the [Kaggle Maze Crawler](https://www.kaggle.com/competitions/maze-crawler) competition.

## Strategy Overview

This agent combines multiple proven strategies from open-source solutions:

- **A\* Pathfinding** with dynamic risk weights for efficient navigation
- **Role-based FSM** (EXPLORER / HARVESTER / SAPPER / GUARD) with role stickiness
- **Dual-mode BFS** (pessimistic known-only + optimistic exploration)
- **Wall Memory Caching** with mirror-symmetry inference
- **Gap-based Emergency Logic** for late-game survival
- **Crush-Hierarchy Collision Avoidance** to prevent friendly fire
- **Economic Energy Budgeting** with dynamic build scaling
- **Enemy Factory Tracking** with predictive evasion
- **Mine ROI Calculation** and energy transfer optimization

## Project Structure

```
├── src/
│   └── main.py          # Agent implementation
├── main.py              # Symlink/copy of src/main.py (Kaggle submission entry)
├── test_local.py        # Local test harness
└── README.md
```

## Local Testing

```bash
pip install kaggle-environments
python test_local.py
```

## Submission

```bash
# Copy to root for submission
cp src/main.py main.py

# Submit to Kaggle
kaggle competitions submit maze-crawler -f main.py -m "Combined strategy v1"
```

## Key References

- [fs0cietyx/maze-crawler](https://github.com/fs0cietyx/maze-crawler) — A\* + Risk Matrix
- [ChibaRie/maze_crawler_inKaggle](https://github.com/ChibaRie/maze_crawler_inKaggle) — FSM + BFS
- [franklu0819-lang/kaggle](https://github.com/franklu0819-lang/kaggle) — Fog-Aware Strategy
- [tuannm3812/kaggle-maze-crawler](https://github.com/tuannm3812/kaggle-maze-crawler) — Jump-Preferred BFS

## Competition Info

- **Deadline**: June 16, 2026 (final submission)
- **Format**: 1v1 maze crawling strategy game
- **Goal**: Last factory standing wins
