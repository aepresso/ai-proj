import math, random, statistics, time
from datetime import datetime

from astar_scheduler import ROUTE, DAY, Params, successors, run, metrics, print_schedule

# A schedule is one decision per intermediate stop (route[1] .. route[-2]):
PASS, BREAK, REST = 0, 1, 2
NAMES = {PASS: "pass", BREAK: "break", REST: "rest"}


# ---------------------------------------------------------------- representation
def step(s, kind, route, p, j=None):
    """Apply one action using the A* successor function; None if it is illegal there."""
    for act, ns, c in successors(s, route, p):
        if act[0] == kind and (j is None or act[2] == j):
            return act, ns, c
    return None


def decode(plan, route, depart_clk, p):
    """Turn a decision vector into (actions, cost), or None if any step breaks an HOS rule.
    Consecutive PASS stops are merged into one DRIVE, exactly as A* drives i -> j, so
    legality and cost are identical to the A* model and the two costs are comparable."""
    s = (0, depart_clk % DAY, 0, 0, 0, 0)
    acts, cost = [], 0.0
    stops = [(i + 1, dec) for i, dec in enumerate(plan) if dec != PASS] + [(len(route) - 1, PASS)]
    for j, dec in stops:
        r = step(s, "DRIVE", route, p, j)
        if r is None:
            return None
        acts.append(r[0]); s = r[1]; cost += r[2]
        if dec != PASS:
            r = step(s, "BREAK" if dec == BREAK else "REST", route, p)
            if r is None:
                return None
            acts.append(r[0]); s = r[1]; cost += r[2]
    return acts, cost


def initial_plan(route, depart_clk, p):
    """Greedy feasible start: pass every stop you legally can, else break, else rest."""
    plan = []
    for i in range(1, len(route) - 1):
        for dec in (PASS, BREAK, REST):
            if decode(plan + [dec], route[:i + 2], depart_clk, p) is not None:
                plan.append(dec)
                break
        else:
            return None                                     # a single leg is longer than 8 h
    return plan


def neighbor(plan, rng):
    """Shift a break/rest to an adjacent stop, or change the decision at one stop."""
    q = plan[:]
    i = rng.randrange(len(q))
    if len(q) > 1 and rng.random() < 0.5:
        j = i + 1 if i + 1 < len(q) else i - 1
        q[i], q[j] = q[j], q[i]
    else:
        q[i] = rng.choice([d for d in (PASS, BREAK, REST) if d != q[i]])
    return q


# ---------------------------------------------------------------- search
def anneal(route, depart_clk, p, iters=4000, T0=5.0, T_end=0.01, seed=0):
    """Simulated annealing over decision vectors. Illegal neighbours are rejected, so every
    schedule held (and returned) is legal by construction, as in A*."""
    rng = random.Random(seed)
    cur = initial_plan(route, depart_clk, p)
    if cur is None:
        return None
    cur_cost = decode(cur, route, depart_clk, p)[1]
    best, best_cost = cur, cur_cost
    alpha = (T_end / T0) ** (1.0 / iters)                   # geometric cooling T0 -> T_end
    T = T0
    accepted = illegal = 0
    for _ in range(iters if cur else 0):
        cand = neighbor(cur, rng)
        dec = decode(cand, route, depart_clk, p)
        if dec is None:
            illegal += 1
        else:
            delta = dec[1] - cur_cost
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                cur, cur_cost = cand, dec[1]
                accepted += 1
                if cur_cost < best_cost - 1e-12:
                    best, best_cost = cur, cur_cost
        T *= alpha
    acts, cost = decode(best, route, depart_clk, p)
    return dict(actions=acts, cost=cost, plan=best, accepted=accepted, illegal=illegal)


# ---------------------------------------------------------------- evaluation
def compare(title, route, depart_dt, p, seeds=range(10)):
    """A* (optimal reference) vs. SA over several seeds on one trip."""
    depart_clk = depart_dt.hour * 4 + depart_dt.minute // 15
    t0 = time.perf_counter(); opt = run(route, depart_clk, p, "astar"); t_a = time.perf_counter() - t0
    m_a = metrics(opt, route, depart_clk, p)

    sa = []
    for sd in seeds:
        t0 = time.perf_counter(); r = anneal(route, depart_clk, p, seed=sd); t = time.perf_counter() - t0
        sa.append((r, metrics(r, route, depart_clk, p), t))
    costs = [r["cost"] for r, _, _ in sa]
    best_r, best_m, _ = min(sa, key=lambda x: x[0]["cost"])
    hits = sum(abs(c - opt["cost"]) < 1e-9 for c in costs)

    print(f"\n=== {title} | depart {depart_dt:%a %H:%M} | {route[-1][1]} mi | lambda={p.lam} ===")
    print(f"{'Algorithm':<16}{'Legal %':>8}{'Time (ms)':>10}{'Cost':>8}{'Gap %':>7}"
          f"{'Trip h':>8}{'S_fat':>8}{'Rests':>6}")
    print(f"{'A* (optimal)':<16}{m_a['compliance']:>8.0f}{t_a*1000:>10.2f}{opt['cost']:>8.2f}{0:>7.1f}"
          f"{m_a['T_total']:>8.2f}{m_a['S_fatigue']:>8.2f}{m_a['rests']:>6}")
    print(f"{'SA best':<16}{best_m['compliance']:>8.0f}{statistics.median(t for *_, t in sa)*1000:>10.2f}"
          f"{best_r['cost']:>8.2f}{100*(best_r['cost']/opt['cost']-1):>7.1f}"
          f"{best_m['T_total']:>8.2f}{best_m['S_fatigue']:>8.2f}{best_m['rests']:>6}")
    mean = statistics.mean(costs)
    print(f"{'SA mean':<16}{statistics.mean(m['compliance'] for _, m, _ in sa):>8.0f}{'':>10}"
          f"{mean:>8.2f}{100*(mean/opt['cost']-1):>7.1f}")
    print(f"   SA found the A* optimum in {hits}/{len(costs)} seeds "
          f"(SA time = median over seeds, 4000 iterations each)")
    return opt, best_r


if __name__ == "__main__":
    p = Params()
    d0 = datetime(2026, 9, 27, 6, 0)

    # 1) Single shift: Atlanta -> Nashville
    compare("SINGLE SHIFT: Atlanta -> Nashville", ROUTE[:3], d0, p)

    # 2) Two shifts: Atlanta -> Chicago, with the SA schedule printed
    _, r = compare("TWO SHIFTS: Atlanta -> Chicago", ROUTE, d0, p)
    print("   SA best schedule:")
    print_schedule(metrics(r, ROUTE, 24, p), d0)

    # 3) Departure-time sweep: does SA still reach the optimum when the fatigue term matters?
    for hr in (12, 18, 22):
        compare("TWO SHIFTS: Atlanta -> Chicago", ROUTE, d0.replace(hour=hr), p)
