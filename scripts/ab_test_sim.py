import argparse, json, math
import numpy as np
from scipy.stats import norm
import pathlib

def two_proportion_ztest(x1, n1, x2, n2):
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
    z = ((x2/n2) - (x1/n1)) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))
    p1, p2 = x1/n1, x2/n2
    se_np = math.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    diff = p2 - p1
    ci = (diff - 1.96*se_np, diff + 1.96*se_np)
    return z, p_value, diff, ci

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--cr", type=float, default=0.10)
    parser.add_argument("--lift", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    x1 = rng.binomial(args.n, args.cr)
    x2 = rng.binomial(args.n, args.cr * (1 + args.lift))

    z, p, diff, ci = two_proportion_ztest(x1, args.n, x2, args.n)
    out = {
        "control": {"n": args.n, "conv": int(x1), "cr": round(x1/args.n, 6)},
        "treatment": {"n": args.n, "conv": int(x2), "cr": round(x2/args.n, 6)},
        "diff_abs": round(diff, 6),
        "ci95": [round(ci[0], 6), round(ci[1], 6)],
        "z": round(z, 4),
        "p_value": round(p, 6),
        "significant_α_0.05": bool(p < 0.05),
    }
    print(json.dumps(out, indent=2))

    pathlib.Path("artifacts").mkdir(exist_ok=True, parents=True)
    with open("artifacts/ab_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
