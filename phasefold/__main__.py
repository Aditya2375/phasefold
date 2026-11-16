import argparse, json
from .core import analyze, generate

p = argparse.ArgumentParser(
    description="Synthetic stock-regime two-means lab, no trades or forecast claims"
)
p.add_argument("--seed", type=int, default=29)
p.add_argument(
    "--scenario",
    choices=["alternating", "shift", "null", "flat"],
    default="alternating",
)
p.add_argument("--window", type=int, default=20)
p.add_argument("--train-end", type=int, default=240)


def main():
    a = p.parse_args()
    try:
        result = analyze(generate(a.seed, a.scenario)["returns"], a.window, a.train_end)
    except ValueError as e:
        p.error(str(e))
    print(json.dumps(result, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
