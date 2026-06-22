# Calculate relative-sigma tolerances for edge-vs-reference net_beta tests.
#
# The edge implementation samples condom use per act via Bernoulli draws, while the
# reference (StructuredSexual in stisim) uses a blended per-act transmission probability.
# So net_beta is a random variable whose mean (across edges) fluctuates around the
# reference value. Here we can compute how much fluctuation to expect, so we can
# set rtol tight enough to catch real bugs but loose enough that a bad seed won't
# flip the test.
#
# The math, briefly:
#
#   Each edge has n_acts acts (held constant in tests for sanity). For each act,
#   a Bernoulli(condom_data) coin flip decides whether a condom is used.
#   The number of protected acts on one edge is k ~ Binomial(n_acts, condom_data).
#   Given k, the edge net_beta is:
#       net_beta(k) = 1 - (1-db)^(n_acts - k) * (1 - db*(1-eff))^k
#                 unprotected_transmission──┘   └──protected_act_reduction
#
#   where db is the per-act disease transmission probability, and eff is condom
#   efficacy. The reference instead uses the expected (blended) per-act rate:
#       net_beta_ref = 1 - (1 - db*(1 - condom_data*eff))^n_acts
#         unprotected_transmission──┘   └──protected_act_reduction
#
#   Because k is random, net_beta(k) has variance. We compute this exactly by
#   summing over all possible k values weighted by the binomial PMF.
#
#   The test asserts on the mean across n_edges independent edges, so the
#   standard error of that mean is sqrt(Var / n_edges). One sigma of relative
#   error is std_of_mean / net_beta_ref; multiply by N to get an N-sigma bound.
#
# What drives the tolerance:
#
#   - n_edges: more edges = more averaging = tighter. Approximate steady-state
#     edge count for a default StructuredSexual network:
#         n_edges ≈ 0.33 * n_agents - 8
#     e.g. n_agents=200 → ~58 edges, n_agents=100 → ~25 edges.
#   - n_acts: more acts = binomial concentrates = less per-edge variance.
#   - eff * condom_data: when the gap between protected and unprotected
#     transmission is large, variation in k matters more.
#   - db: when db is small, net_beta is roughly linear in k, so binomial noise
#     maps directly through. When db is large, net_beta saturates near 1.0
#     regardless of k, compressing variance.
#
# Usage:
#     python tests/calc_tolerances.py                        # prints examples + full sweep
#     python tests/calc_tolerances.py 0.038 0.9 0.5 100 200  # db eff cond acts n_agents

import sys
import numpy as np
from scipy.stats import binom


def calc_tolerance(db, eff, condom_data, n_acts, n_agents=None, n_edges=None):
    if n_edges is None:
        n_edges = max(1, round(0.33 * n_agents - 8))  # approx steady-state edges for default StructuredSexual

    # reference (blended-rate) net_beta -- this is what the analytical formula gives
    db_blended = db * (1 - condom_data * eff)
    ref = 1 - (1 - db_blended) ** n_acts

    # per-edge: net_beta(k) = 1 - a^(n-k) * b^k, where k ~ Bin(n_acts, condom_data)
    # a = survival prob for an unprotected act, b = survival prob for a protected act
    a = 1 - db
    b = 1 - db * (1 - eff)

    ks = np.arange(n_acts + 1)
    pmf = binom.pmf(ks, n_acts, condom_data)

    # net_beta for each possible k
    survival = a ** (n_acts - ks) * b**ks
    nb_vals = 1 - survival

    # exact moments over the binomial distribution of k
    E_edge = np.sum(pmf * nb_vals)
    E_edge2 = np.sum(pmf * nb_vals**2)
    Var_edge = max(0.0, E_edge2 - E_edge**2)  # clamp floating-point noise

    # the test asserts on the mean across n_edges independent edges
    std_of_mean = np.sqrt(Var_edge / n_edges)
    sigma = std_of_mean / ref if ref > 0 else float("inf")

    return dict(
        db=db,
        eff=eff,
        condom_data=condom_data,
        n_acts=n_acts,
        n_agents=n_agents,
        n_edges=n_edges,
        ref=ref,
        E_edge=E_edge,
        Var_edge=Var_edge,
        std_of_mean=std_of_mean,
        sigma=sigma,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        assert sys.argv
        db, eff, condom_data, n_acts, n_agents = sys.argv[1:]
        r = calc_tolerance(float(db), float(eff), float(condom_data), int(n_acts), round(0.33 * int(n_agents) - 8))

    else:
        print("test_starsim_only::test_condom_net_beta_analytical (n_agents=200)")
        sig = calc_tolerance(0.0083, 0.8, 0.5, 100, 200)["sigma"]
        print(f"(0.0083, 0.8, 0.5, 100, 200, {round(sig * 3, 2)})", end="\n\n")

        print("test_core::test_net_beta_range")
        scenarios = [
            (0.005,  1.0,  0.8, 10,  100, "Worst case: all drivers aligned"),       # 0.13
            (0.019,  0.9,  0.5, 10,  100, "Typical hard case, no extreme"),         # 0.05
            (0.0083, 0.8,  0.8, 100, 100, "n_acts=100 concentrates the binomial"),  # 0.02
            (0.05,   1.0,  0.5, 10,  200, "Population size averaging"),             # 0.04
            (0.1,    0.9,  0.8, 10,  200, "High db saturation compresses"),         # 0.05
            (0.038,  0.46, 0.2, 10,  100, "Low eff * cond → near-deterministic"),   # 0.01
        ]

        lines = []
        for db, eff, cond, n_acts, n_agents, comment in scenarios:
            sig = calc_tolerance(db, eff, cond, n_acts, n_agents)["sigma"]
            tup = f"({db}, {eff}, {cond}, {n_acts}, {n_agents}, {round(sig * 3, 2)})"
            lines.append((tup, comment))

        max_tup = max(len(t) for t, _ in lines)
        for tup, comment in lines:
            print(f"{tup:<{max_tup}}  # {comment}")
