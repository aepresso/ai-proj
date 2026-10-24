# HOS-Compliant Truck Driver Scheduling Agent

CS 3642 (Artificial Intelligence) semester project, Fall 2026, Kennesaw State University.

An agent that plans legal, low-fatigue schedules for long-haul truck trips under U.S. FMCSA hours-of-service rules
(11 h driving / 14 h window / 30-min break after 8 h / 10 h rest / weekly cap).

## Methods

| Component | File | Status |
|---|---|---|
| A* search over duty-status states | `astar_scheduler.py` | working prototype |
| Rule-based fatigue risk model | — | currently inline in `astar_scheduler.py` (`drive_fatigue`) |
| Simulated annealing for multi-day trips | — | not started |

## Running

Requires Python 3.12+ (no third-party packages).

```sh
python3 astar_scheduler.py
```

This compares A*, UCS and greedy best-first search on Atlanta -> Nashville and Atlanta -> Chicago,
then runs a departure-time sweep and a sensitivity analysis on the fatigue weight `lambda`.

Route mile markers are approximate and for illustration only.
