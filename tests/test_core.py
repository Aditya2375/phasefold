import copy, math, random, statistics, unittest
from phasefold import analyze, features, fit, generate, predict


class Core(unittest.TestCase):
    def test_discrete_canonical_ties(self):
        rng = random.Random(19)
        for _ in range(2000):
            points = [
                [float(rng.randrange(3)), float(rng.randrange(3))] for _ in range(20)
            ]
            model = fit(points)
            self.assertEqual(model["labels"], [predict(model, p) for p in points])
            self.assertEqual(
                model["counts"], [model["labels"].count(k) for k in range(2)]
            )

    def test_analytic_features(self):
        r = [0.001 * i for i in range(100)]
        f = features(r, 5)
        self.assertAlmostEqual(f[0]["mean"], 0.002)
        self.assertAlmostEqual(f[0]["sd"], statistics.stdev(r[:5]))
        self.assertEqual(f[0]["t"], 5)

    def test_two_clouds(self):
        p = [[-2.0, -3.0]] * 10 + [[4.0, 5.0]] * 10
        m = fit(p)
        self.assertEqual(m["labels"], [0] * 10 + [1] * 10)
        self.assertEqual(m["inertia"], 0)
        self.assertEqual(m["counts"], [10, 10])
        self.assertTrue(m["converged"])

    def test_constant(self):
        m = fit([[0.0, 0.0]] * 30)
        self.assertEqual(m["counts"], [30, 0])
        self.assertEqual(m["scales"], [1.0, 1.0])
        self.assertEqual(m["inertia"], 0)
        self.assertEqual(predict(m, [0, 0]), 0)

    def test_future_and_current_invariance(self):
        r = generate()["returns"]
        a = analyze(r)
        other = r[:]
        other[300:] = [0.1] * 120
        b = analyze(other)
        self.assertEqual(a["model"], b["model"])
        self.assertEqual(a["rows"][:281], b["rows"][:281])
        other = r[:]
        other[240] = 0.2
        c = analyze(other)
        self.assertEqual(a["model"], c["model"])
        self.assertEqual(a["rows"][220], c["rows"][220])

    def test_no_mutation(self):
        r = generate()["returns"]
        before = r[:]
        analyze(r)
        self.assertEqual(r, before)
        p = [[1.0, 2.0]] * 10 + [[3.0, 4.0]] * 10
        b = copy.deepcopy(p)
        fit(p)
        self.assertEqual(p, b)

    def test_validation(self):
        for v in [
            None,
            [],
            [0] * 99,
            [0] * 1001,
            [True] * 100,
            [float("nan")] * 100,
            [0.3] * 100,
        ]:
            with self.assertRaises(ValueError):
                features(v, 20)
        for w in [True, 4, 61, 20.5]:
            with self.assertRaises(ValueError):
                features([0] * 100, w)
        for p in [[], [[0, 0]] * 19, [[0, 0, 0]] * 20, [[float("inf"), 0]] * 20]:
            with self.assertRaises(ValueError):
                fit(p)

    def test_scaler_and_sse_oracle(self):
        rng = random.Random(18)
        for _ in range(120):
            points = [[rng.uniform(-3, 3), rng.uniform(-2, 2)] for i in range(40)]
            m = fit(points)
            for j in range(2):
                self.assertAlmostEqual(
                    m["means"][j], statistics.mean(p[j] for p in points)
                )
                self.assertAlmostEqual(
                    m["scales"][j], statistics.pstdev(p[j] for p in points)
                )
            sse = 0
            for p, label in zip(points, m["labels"]):
                z = [(p[j] - m["means"][j]) / m["scales"][j] for j in range(2)]
                d = [math.dist(z, c) ** 2 for c in m["centers"]]
                self.assertLessEqual(d[label], min(d) + 1e-12)
                sse += d[label]
            self.assertAlmostEqual(sse, m["inertia"])
            self.assertTrue(
                all(b <= a + 1e-9 for a, b in zip(m["trace"], m["trace"][1:]))
            )

    def test_train_boundary_and_constant_dimension(self):
        r = generate()["returns"]
        for w, end in [(5, 35), (60, 90), (5, 390), (60, 390)]:
            a = analyze(r, w, end)
            self.assertEqual(len(a["model"]["labels"]), end - w)
            self.assertEqual(a["rows"][end - w]["split"], "holdout")
            self.assertEqual(sum(a["holdout_counts"]), 420 - end)
        m = fit([[2.0, float(i % 2)] for i in range(20)])
        self.assertEqual(m["scales"][0], 1)
        self.assertEqual(m["counts"], [10, 10])

    def test_tie_and_translation_scale_equivariance(self):
        m = fit([[-2.0, -3.0]] * 10 + [[4.0, 5.0]] * 10)
        self.assertEqual(predict(m, [1.0, 1.0]), 0)
        points = [[float(i), float((i * i) % 13)] for i in range(30)]
        a = fit(points)
        b = fit([[p[0] * 3 + 4, p[1] * 2 - 1] for p in points])
        self.assertEqual(a["labels"], b["labels"])
        for x, y in zip(a["centers"], b["centers"]):
            for u, v in zip(x, y):
                self.assertAlmostEqual(u, v)

    def test_full_bound_input_and_cli(self):
        import subprocess, sys, json

        a = analyze([0.0] * 1000, 5, 970)
        self.assertEqual(sum(a["holdout_counts"]), 30)
        for args in [
            ["--seed", "0"],
            ["--window", "4"],
            ["--train-end", "419"],
            ["--scenario", "bad"],
        ]:
            p = subprocess.run(
                [sys.executable, "-m", "phasefold", *args], capture_output=True
            )
            self.assertEqual(p.returncode, 2)
            self.assertEqual(p.stdout, b"")
        p = subprocess.run(
            [sys.executable, "-m", "phasefold", "--scenario", "flat"],
            capture_output=True,
        )
        self.assertEqual(p.returncode, 0)
        self.assertEqual(json.loads(p.stdout)["model"]["counts"], [220, 0])

    def test_all_scenarios_bounds(self):
        for scenario in ["alternating", "shift", "null", "flat"]:
            for seed in [1, 29, 4294967295]:
                a = analyze(generate(seed, scenario)["returns"])
                self.assertEqual(sum(a["holdout_counts"]), 180)
                self.assertEqual(len(a["rows"]), 400)


if __name__ == "__main__":
    unittest.main()
