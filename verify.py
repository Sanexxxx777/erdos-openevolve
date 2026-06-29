"""Standalone verification of the evolved solution.

Runs best_program.py, then independently re-checks the two constraints and the
reported C5 upper bound using only numpy (no JAX, no trust in the program's own
numbers). Prints a PASS/FAIL summary.

    python verify.py
"""

import importlib.util
import numpy as np

BENCHMARK = 0.38092303510845016  # AlphaEvolve's reported numerical upper bound


def load_run(path="best_program.py"):
    spec = importlib.util.spec_from_file_location("best", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run()


def main():
    h, c5_reported, n = load_run()
    h = np.asarray(h, dtype=float)
    dx = 2.0 / n

    in_unit = bool(h.min() >= 0.0 and h.max() <= 1.0)
    integral = float(np.sum(h) * dx)
    integral_ok = abs(integral - 1.0) < 1e-3

    # Independent C5: max of the cross-correlation of h with j = 1 - h, scaled by dx.
    j = 1.0 - h
    c5_independent = float(np.max(np.correlate(h, j, mode="full") * dx))
    c5_consistent = abs(c5_independent - c5_reported) < 1e-4

    print(f"n intervals        : {n}")
    print(f"h in [0, 1]        : {in_unit}  (range [{h.min():.4f}, {h.max():.4f}])")
    print(f"integral of h      : {integral:.8f}  (target 1.0, |dev| = {abs(integral - 1.0):.2e})")
    print(f"C5 (reported)      : {c5_reported:.8f}")
    print(f"C5 (recomputed)    : {c5_independent:.8f}")
    print(f"AlphaEvolve bench  : {BENCHMARK:.8f}")
    print()
    ok = in_unit and integral_ok and c5_consistent
    print("constraints PASS   :", ok)
    if c5_independent < BENCHMARK:
        print(f"=> BEATS benchmark by {BENCHMARK - c5_independent:.2e}")
    else:
        print(f"=> matches benchmark to {100 * BENCHMARK / c5_independent:.2f}% "
              f"(gap {c5_independent - BENCHMARK:.2e})")


if __name__ == "__main__":
    main()
