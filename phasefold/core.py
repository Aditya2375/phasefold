"""Original two-means stock-regime lab. Synthetic returns; no forecasts or trades."""

import math


def integer(v, lo, hi, label):
    if type(v) is not int or not lo <= v <= hi:
        raise ValueError(f"{label} must be an integer in {lo}..{hi}")
    return v


def features(returns, window):
    """At boundary t, use only returns[t-window:t], excluding return t."""
    integer(window, 5, 60, "window")
    if type(returns) is not list or not 100 <= len(returns) <= 1000:
        raise ValueError("Need 100..1000 log returns")
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 0.25
        for v in returns
    ):
        raise ValueError("Log returns must be finite numbers within -0.25..0.25")
    out = []
    for t in range(window, len(returns)):
        data = returns[t - window : t]
        mean = sum(data) / window
        sd = math.sqrt(sum((v - mean) ** 2 for v in data) / (window - 1))
        out.append({"t": t, "mean": mean, "sd": sd})
    return out


def distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def classify(x, centers, tie_label=0):
    left, right = distance(x, centers[0]), distance(x, centers[1])
    return tie_label if left == right else (0 if left < right else 1)


def fit(points):
    """Train-only population scaler and two-means; four deterministic starts.

    Empty clusters retain their old center. All-constant data collapse to one
    occupied cluster; no fabricated second regime. Ties go to center0.
    """
    if (
        type(points) is not list
        or not 20 <= len(points) <= 1000
        or any(
            type(p) is not list
            or len(p) != 2
            or any(
                type(v) not in (float, int) or not math.isfinite(v) or abs(v) > 1e6
                for v in p
            )
            for p in points
        )
    ):
        raise ValueError("Need 20..1000 finite two-dimensional bounded points")
    means = [sum(p[j] for p in points) / len(points) for j in range(2)]
    scales = [
        math.sqrt(sum((p[j] - means[j]) ** 2 for p in points) / len(points))
        for j in range(2)
    ]
    scales = [v if v > 1e-12 else 1.0 for v in scales]
    scaled = [[(p[j] - means[j]) / scales[j] for j in range(2)] for p in points]
    best = None
    for start in [0, len(points) // 3, 2 * len(points) // 3, len(points) - 1]:
        first = scaled[start][:]
        farthest = max(
            range(len(points)), key=lambda i: (distance(scaled[i], first), -i)
        )
        centers = [first, scaled[farthest][:]]
        trace = []
        previous = None
        converged = False
        for iteration in range(100):
            labels = [classify(p, centers) for p in scaled]
            updated = []
            for k in range(2):
                members = [p for p, label in zip(scaled, labels) if label == k]
                updated.append(
                    [sum(p[j] for p in members) / len(members) for j in range(2)]
                    if members
                    else centers[k][:]
                )
            centers = updated
            # Reassignment is required before evaluating the updated objective.
            labels = [classify(p, centers) for p in scaled]
            objective = sum(
                distance(p, centers[label]) for p, label in zip(scaled, labels)
            )
            trace.append(objective)
            if labels == previous:
                converged = True
                break
            previous = labels
        counts = [labels.count(k) for k in range(2)]
        candidate = {
            "centers": centers,
            "labels": labels,
            "counts": counts,
            "inertia": objective,
            "trace": trace,
            "iterations": len(trace),
            "converged": converged,
            "start": start,
        }
        if best is None or objective < best["inertia"]:
            best = candidate
    # Canonicalize by raw volatility coordinate, then raw mean, preserving original center0 ties.
    order = sorted(
        range(2), key=lambda k: (best["centers"][k][1], best["centers"][k][0], k)
    )
    best["centers"] = [best["centers"][k] for k in order]
    best["counts"] = [best["counts"][k] for k in order]
    mapping = {old: new for new, old in enumerate(order)}
    best["labels"] = [mapping[v] for v in best["labels"]]
    return {"means": means, "scales": scales, "tie_label": mapping[0], **best}


def predict(model, point):
    return classify(
        [(point[j] - model["means"][j]) / model["scales"][j] for j in range(2)],
        model["centers"],
        model["tie_label"],
    )


def analyze(returns, window=20, train_end=240):
    rows = features(returns, window)
    integer(train_end, window + 30, len(returns) - 30, "training boundary")
    training = [r for r in rows if r["t"] < train_end]
    model = fit([[r["mean"], r["sd"]] for r in training])
    # Training assignments use a model fitted after their historical timestamps;
    # only t>=train_end labels are prospective with respect to model fitting.
    for row in rows:
        row["label"] = predict(model, [row["mean"], row["sd"]])
        row["split"] = "fit" if row["t"] < train_end else "holdout"
        scaled = [
            (row["mean"] - model["means"][0]) / model["scales"][0],
            (row["sd"] - model["means"][1]) / model["scales"][1],
        ]
        row["distance"] = min(distance(scaled, c) for c in model["centers"])
    hold = [r for r in rows if r["split"] == "holdout"]
    return {
        "window": window,
        "train_end": train_end,
        "model": model,
        "rows": rows,
        "holdout_counts": [sum(r["label"] == k for r in hold) for k in range(2)],
        "holdout_switches": sum(
            a["label"] != b["label"] for a, b in zip(hold, hold[1:])
        ),
    }


class Random:
    def __init__(self, seed):
        self.state = integer(seed, 1, 4294967295, "seed")

    def uniform(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state + 0.5) / 4294967296

    def normal(self):
        return math.sqrt(-2 * math.log(self.uniform())) * math.cos(
            2 * math.pi * self.uniform()
        )


def generate(seed=29, scenario="alternating"):
    if scenario not in ("alternating", "shift", "null", "flat"):
        raise ValueError("Unknown scenario")
    rng = Random(seed)
    returns = []
    truth = []
    for t in range(420):
        state = (
            (t // 70) % 2
            if scenario == "alternating"
            else int(t >= 270) if scenario == "shift" else 0
        )
        mean = 0.0006 if state == 0 else -0.0005
        sd = 0.005 if state == 0 else 0.018
        value = (
            0.0
            if scenario == "flat"
            else max(-0.25, min(0.25, mean + sd * rng.normal()))
        )
        returns.append(value)
        truth.append(state)
    return {"returns": returns, "truth": truth}
