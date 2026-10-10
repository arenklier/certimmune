"""Expose the optimization seed that delta_optimize_fast.py hard-codes.

At the wall-clock-matched budget one identity out of twenty (identity_00) fails
to reach the floor at 80 steps although it reached it at 50, and there is no way
to tell an over-optimization effect from one unlucky draw while the seed is
fixed: every step count is a different theta sequence, so "more steps" and
"different draw" are confounded. --seed defaults to 0 and reproduces the runs
already on disk.
"""
import re
import sys

P = "/mnt/data/certimmune/src/delta_optimize_fast.py"
s = open(P, encoding="utf-8").read()
NL = chr(10)

EDITS = [
    ('    ap.add_argument("--tag", default="_fasteot")',
     '    ap.add_argument("--seed", type=int, default=0,' + NL +
     '                    help="optimization seed. 0 reproduces the runs already on disk; other values "' + NL +
     '                         "give an independent draw of the same budget.")' + NL +
     '    ap.add_argument("--tag", default="_fasteot")'),
    ('                                              args.k_theta, text_emb, args.epsilon, args.steps, args.step_size)',
     '                                              args.k_theta, text_emb, args.epsilon, args.steps,' + NL +
     '                                              args.step_size, seed=args.seed)'),
]
for old, new in EDITS:
    if new.strip() in s:
        print("  done (already patched):", old.strip()[:50]); continue
    if s.count(old) != 1:
        print("  FAIL %d hits: %s" % (s.count(old), old.strip()[:60])); sys.exit(1)
    s = s.replace(old, new)
    print("  ok  ", old.strip()[:56])
open(P, "w", encoding="utf-8").write(s)
import ast; ast.parse(s); print("  syntax ok")
