import heapq, itertools, math, time, statistics
from dataclasses import dataclass
from datetime import datetime, timedelta

# ---------------------------------------------------------------- constants
STEP_H = 0.25                      # 15-minute time step
LIM_BREAK, LIM_DRIVE, LIM_WINDOW, LIM_WEEK = 32, 44, 56, 280
BREAK_STEPS, REST_STEPS, DAY = 2, 40, 96
REST_H = REST_STEPS * STEP_H       # 10 h
SPEED_MPH = 60.0                   # average speed (same as the walkthrough example)

# Candidate stops along I-75 / I-24 / I-65 / I-65-I-65N / I-65 -> Chicago.
# Mile markers are APPROXIMATE (illustrative); replace with OSRM / OpenStreetMap data.
ROUTE = [
    ("Atlanta, GA", 0), ("Chattanooga, TN", 118), ("Nashville, TN", 253),
    ("Bowling Green, KY", 318), ("Elizabethtown, KY", 383), ("Louisville, KY", 428),
    ("Columbus, IN", 480), ("Indianapolis, IN", 535), ("Lafayette, IN", 600),
    ("Rensselaer, IN", 650), ("Merrillville, IN", 690), ("Chicago, IL", 715),
]


@dataclass
class Params:
    lam: float = 1.0       # weight of fatigue-related cost vs. elapsed time
    w_circ: float = 1.0    # cost rate (per hour) while driving 12:00 AM - 6:00 AM
    w_dur: float = 0.5     # cost rate growth (per hour^2) after 4 h continuous driving


# ---------------------------------------------------------------- cost model
def drive_fatigue(clk, b, k, p):
    """Fatigue-related cost of k driving steps starting at clock clk with b steps
    already driven since the last break (rule-based, not a validated safety measure)."""
    f = 0.0
    for s in range(k):
        hod = ((clk + s + 0.5) * STEP_H) % 24
        circ = 1.0 if hod < 6 else 0.0                        # circadian low window
        excess = max(0.0, (b + s + 0.5) * STEP_H - 4.0)       # hours beyond hour 4
        f += STEP_H * (p.w_circ * circ + p.w_dur * excess)
    return f


def successors(s, route, p):
    i, clk, b, d, w, W = s
    N = len(route)
    out = []
    # --- DRIVE i -> j : drive time is monotone in j, so stop at the first illegal j
    for j in range(i + 1, N):
        dist = route[j][1] - route[i][1]
        k = math.ceil(dist / SPEED_MPH / STEP_H - 1e-9)
        if b + k > LIM_BREAK or d + k > LIM_DRIVE or w + k > LIM_WINDOW or W + k > LIM_WEEK:
            break
        cost = k * STEP_H + p.lam * drive_fatigue(clk, b, k, p)
        out.append((("DRIVE", i, j, k), (j, (clk + k) % DAY, b + k, d + k, w + k, W + k), cost))
    # --- BREAK 30 min (only if there is driving to reset)
    if b > 0 and w + BREAK_STEPS <= LIM_WINDOW:
        out.append((("BREAK", i), (i, (clk + BREAK_STEPS) % DAY, 0, d, w + BREAK_STEPS, W),
                    BREAK_STEPS * STEP_H))
    # --- REST 10 h (only mid-trip and only if the window is open)
    if i < N - 1 and w > 0:
        out.append((("REST", i), (i, (clk + REST_STEPS) % DAY, 0, 0, 0, W), REST_H))
    return out


def make_heuristic(route):
    D = route[-1][1]

    def h(s):
        i, clk, b, d, w, W = s
        R = (D - route[i][1]) / SPEED_MPH                   # remaining driving hours at v
        extra = R - (LIM_DRIVE - d) * STEP_H                # beyond the current shift
        rests = math.ceil(extra / 11.0 - 1e-9) if extra > 1e-9 else 0
        return R + REST_H * rests                           # fatigue term = 0
    return h


# ---------------------------------------------------------------- search
def run(route, depart_clk, p, mode="astar", check_consistency=False):
    """mode: 'astar' (g+h), 'ucs' (g only), 'greedy' (h only). Returns plan + true path cost."""
    h = make_heuristic(route)
    start = (0, depart_clk % DAY, 0, 0, 0, 0)
    goal = len(route) - 1
    tie = itertools.count()
    prio = lambda g, s: {"astar": g + h(s), "ucs": g, "greedy": h(s)}[mode]
    best_g, parent, closed = {start: 0.0}, {start: None}, set()
    pq = [(prio(0.0, start), next(tie), start)]
    expanded = generated = violations = 0
    while pq:
        _, _, s = heapq.heappop(pq)
        if s in closed:
            continue
        closed.add(s)
        expanded += 1
        if s[0] == goal:
            acts, cur = [], s
            while parent[cur] is not None:
                prev, act = parent[cur]
                acts.append(act)
                cur = prev
            return dict(actions=acts[::-1], cost=best_g[s], expanded=expanded,
                        generated=generated, violations=violations)
        for act, ns, c in successors(s, route, p):
            generated += 1
            if check_consistency and h(s) > c + h(ns) + 1e-9:
                violations += 1
            ng = best_g[s] + c
            if ng < best_g.get(ns, float("inf")):
                best_g[ns] = ng
                parent[ns] = (s, act)
                heapq.heappush(pq, (prio(ng, ns), next(tie), ns))
    return None


# ---------------------------------------------------------------- evaluation
def timeline(actions, route, depart_clk):
    """Turn an action list into (kind, start_step, end_step, label) events."""
    t, ev = 0, []
    for a in actions:
        if a[0] == "DRIVE":
            _, i, j, k = a
            ev.append(("DRIVE", t, t + k, f"{route[i][0]} -> {route[j][0]}"))
            t += k
        elif a[0] == "BREAK":
            ev.append(("BREAK", t, t + BREAK_STEPS, f"30-min break at {route[a[1]][0]}"))
            t += BREAK_STEPS
        else:
            ev.append(("REST", t, t + REST_STEPS, f"10-h rest at {route[a[1]][0]}"))
            t += REST_STEPS
    return ev


def validate(events):
    """INDEPENDENT HOS checker (does not reuse the search code). Returns (passed, total)."""
    passed = total = 0
    cont = shift_drive = weekly = 0
    shift_start = 0
    off_run = 0
    last_end = 0
    for kind, s, e, _ in events:
        if kind == "DRIVE":
            cont += e - s; shift_drive += e - s; weekly += e - s
            for ok in (cont <= LIM_BREAK, shift_drive <= LIM_DRIVE,
                       e - shift_start <= LIM_WINDOW, weekly <= LIM_WEEK):
                total += 1; passed += ok
            off_run = 0
        else:
            off_run = e - s
            if off_run >= BREAK_STEPS:
                cont = 0
            if off_run >= REST_STEPS:
                shift_drive = 0
                shift_start = e
        last_end = e
    return passed, total


def metrics(res, route, depart_clk, p):
    ev = timeline(res["actions"], route, depart_clk)
    fat = circ_h = 0.0
    clk, b = depart_clk, 0
    n_rest = n_break = 0
    for kind, s, e, _ in ev:
        k = e - s
        if kind == "DRIVE":
            fat += drive_fatigue((depart_clk + s) % DAY, b, k, p)
            circ_h += sum(1 for x in range(k) if (((depart_clk + s + x + 0.5) * STEP_H) % 24) < 6) * STEP_H
            b += k
        else:
            n_break += kind == "BREAK"; n_rest += kind == "REST"
            b = 0
    passed, total = validate(ev)
    return dict(T_total=ev[-1][2] * STEP_H, S_fatigue=fat, circ_hours=circ_h,
                breaks=n_break, rests=n_rest, compliance=100.0 * passed / total,
                events=ev)


def print_schedule(m, depart_dt):
    for kind, s, e, label in m["events"]:
        a = depart_dt + timedelta(minutes=15 * s)
        b = depart_dt + timedelta(minutes=15 * e)
        print(f"   {a:%a %H:%M} - {b:%a %H:%M}  {kind:<5} {label}")


def timed(route, depart_clk, p, mode, repeats=5):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = run(route, depart_clk, p, mode)
        ts.append(time.perf_counter() - t0)
    return res, statistics.median(ts)


def compare(title, route, depart_dt, p):
    depart_clk = depart_dt.hour * 4 + depart_dt.minute // 15
    print(f"\n=== {title} | depart {depart_dt:%a %Y-%m-%d %H:%M} | "
          f"{route[-1][1]} mi | lambda={p.lam} ===")
    print(f"{'Algorithm':<14}{'Legal %':>8}{"Time (ms)":>10}{'Cost':>8}{'Trip h':>8}"
          f"{'S_fat':>8}{'Rests':>6}{'Expanded':>10}{'h-consist. viol.':>18}")
    rows = {}
    for mode, name in (("astar", "A*"), ("ucs", "UCS (h=0)"), ("greedy", "Greedy BFS")):
        res, sec = timed(route, depart_clk, p, mode)
        viol = run(route, depart_clk, p, mode, check_consistency=True)["violations"] if mode == "astar" else "-"
        m = metrics(res, route, depart_clk, p)
        rows[mode] = (res, m)
        print(f"{name:<14}{m['compliance']:>8.0f}{sec*1000:>10.2f}{res['cost']:>8.2f}{m['T_total']:>8.2f}"
              f"{m['S_fatigue']:>8.2f}{m['rests']:>6}{res['expanded']:>10}{str(viol):>18}")
    assert abs(rows["astar"][0]["cost"] - rows["ucs"][0]["cost"]) < 1e-9, "A* not optimal!"
    print("   A* cost == UCS cost  ->  optimality confirmed on this instance")
    print("   A* schedule:")
    print_schedule(rows["astar"][1], depart_dt)
    return rows


if __name__ == "__main__":
    p = Params()
    d0 = datetime(2026, 9, 27, 6, 0)

    # 1) Single-shift baseline: Atlanta -> Nashville (253 mi, ~4.2 h)
    compare("SINGLE SHIFT: Atlanta -> Nashville", ROUTE[:3], d0, p)

    # 2) Atlanta -> Chicago (11.9 h of driving > 11 h limit => at least one 10 h rest)
    compare("TWO SHIFTS: Atlanta -> Chicago", ROUTE, d0, p)

    # 3) Departure-time sweep (A* only): the fatigue term moves the rest placement
    print("\n=== A* departure-time sweep, Atlanta -> Chicago ===")
    print(f"{'Depart':<8}{'Cost':>8}{'Trip h':>8}{'S_fat':>8}{'Circ-low h':>11}{'Rests':>6}{'Legal %':>9}")
    for hr in (6, 12, 18, 22):
        dep = d0.replace(hour=hr)
        clk = hr * 4
        r = run(ROUTE, clk, p, "astar"); m = metrics(r, ROUTE, clk, p)
        print(f"{hr:02d}:00   {r['cost']:>8.2f}{m['T_total']:>8.2f}{m['S_fatigue']:>8.2f}"
              f"{m['circ_hours']:>11.2f}{m['rests']:>6}{m['compliance']:>9.0f}")

    # 4) Sensitivity to the fatigue weight lambda (06:00 departure)
    print("\n=== A* sensitivity to lambda, Atlanta -> Chicago, depart 06:00 ===")
    print(f"{'lambda':<8}{'Cost':>8}{'Trip h':>8}{'S_fat':>8}{'Circ-low h':>11}")
    for lam in (0.0, 0.5, 1.0, 3.0):
        q = Params(lam=lam)
        r = run(ROUTE, 24, q, "astar"); m = metrics(r, ROUTE, 24, q)
        print(f"{lam:<8}{r['cost']:>8.2f}{m['T_total']:>8.2f}{m['S_fatigue']:>8.2f}{m['circ_hours']:>11.2f}")
