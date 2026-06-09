"""Local test harness for Maze Crawler agent."""

import sys
sys.path.insert(0, "src")

from kaggle_environments import make


def run_test(opponent="random", seeds=10, verbose=True):
    """Run the agent against an opponent for N seeds and report results."""
    wins, losses, draws, errors = 0, 0, 0, 0

    for seed in range(seeds):
        env = make("crawl", configuration={"randomSeed": seed}, debug=True)
        try:
            env.run(["src/main.py", opponent])
            final = env.steps[-1]
            r0 = final[0].reward if final[0].reward is not None else 0
            r1 = final[1].reward if final[1].reward is not None else 0

            if r0 > r1:
                wins += 1
                status = "WIN"
            elif r1 > r0:
                losses += 1
                status = "LOSS"
            else:
                draws += 1
                status = "DRAW"

            if verbose:
                print(f"  Seed {seed:3d}: {status:4s}  (reward: {r0:.1f} vs {r1:.1f})")
        except Exception as e:
            errors += 1
            if verbose:
                print(f"  Seed {seed:3d}: ERROR  {e}")

    print(f"\nResults: {wins}W / {losses}L / {draws}D / {errors}E")
    print(f"Win rate: {wins}/{seeds - errors} = {wins/(seeds-errors)*100:.1f}%" if seeds > errors else "All errors")
    return wins, losses, draws, errors


if __name__ == "__main__":
    opponents = ["random"]
    for opp in opponents:
        print(f"\n{'='*50}")
        print(f"Testing vs {opp}")
        print(f"{'='*50}")
        run_test(opponent=opp, seeds=5, verbose=True)
