import json, sys
from phasefold import analyze, generate

programs = json.load(sys.stdin)
print(
    json.dumps(
        [
            analyze(
                generate(p["seed"], p["scenario"])["returns"],
                p["window"],
                p["train_end"],
            )
            for p in programs
        ],
        allow_nan=False,
    )
)
