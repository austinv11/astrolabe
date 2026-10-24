"""Seeded calibration of standard donor-summary OLS/HC3 contrasts.

Prespecified checks: at nominal .05, upper 95% binomial bound on null rejection
<= .08; lower 95% bound on power >= .60 for the specified .15 effect. Coverage
and untestable fractions are reported separately. This is a limited simulation
suite, not a guarantee for arbitrary compositional or confounded data.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from astrolabe.tl import diff_niche_abundance


def scenario(rng, name):
    paired = name == "paired"
    rare = name == "rare"
    effect = .15 if name == "effect" else 0.
    n_reference, n_tested = (8, 16) if name == "imbalanced" else (16, 16)
    shared = rng.beta(12, 28, n_reference)
    rows = []
    for condition, n in (("control", n_reference), ("case", n_tested)):
        probabilities = shared if paired else rng.beta(.5, 49.5, n) if rare else rng.beta(12, 28, n)
        for donor, p in enumerate(probabilities):
            p += effect if condition == "case" else 0.
            p = min(p, 1.)
            # Unequal section/cell yields; technical sections share donor labels.
            for section in range(1 + donor % 3):
                cells = int(rng.integers(40, 401))
                count = rng.binomial(cells, p)
                rows.append({"donor": str(donor) if paired else f"{condition}{donor}", "condition": condition,
                             "niche": "A", "count": count, "exposure": cells})
    table = pd.DataFrame(rows)
    table.attrs.update(donor_key="donor", condition_key="condition", exposure="all_cells")
    result = diff_niche_abundance(table, contrast=("case", "control"), paired=paired).iloc[0]
    return float(result.pvalue), bool(result.ci_low <= effect <= result.ci_high)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=400)
    parser.add_argument("--output", type=Path, default=Path("docs/niche-calibration-results.json"))
    args = parser.parse_args()
    results = []
    for seed, name in enumerate(("balanced", "imbalanced", "paired", "rare", "effect"), 2026):
        rng = np.random.default_rng(seed)
        outcomes = [scenario(rng, name) for _ in range(args.repetitions)]
        p, covered = np.asarray(outcomes).T
        rejected = int((p < .05).sum())
        interval = binomtest(rejected, args.repetitions).proportion_ci()
        record = {"scenario": name, "seed": seed, "repetitions": args.repetitions,
                  "rejection_rate": rejected/args.repetitions,
                  "rejection_ci": [interval.low, interval.high], "interval_coverage": float(covered.mean()),
                  "n_untestable": int((~np.isfinite(p)).sum()),
                  "gate_passed": bool(interval.low >= .60 if name == "effect" else interval.high <= .08)}
        results.append(record)
        args.output.write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(record), flush=True)
