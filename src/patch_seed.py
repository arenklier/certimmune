"""Expose the run seed that lora_calibration.py currently hard-codes.

Both stochastic parts of the chain were fixed constants: train_lora_instance
got seed=0, and the twenty scoring generations came from seed_offset=1000. That
is why every stored run is nominally the same draw -- and why the run-to-run
spread we measured (sd 0.039 at the documented setting) is pure GPU
non-determinism rather than a designed replication.

--seed defaults to 0 and reproduces the existing behaviour byte for byte, so
nothing already on disk is invalidated; seeds 1 and 2 give genuine independent
replicates. The seed is recorded in the output so a file can never be mistaken
for a different replicate than it is.
"""
import sys

P = "/mnt/data/certimmune/src/lora_calibration.py"
s = open(P, encoding="utf-8").read()
orig = s

EDITS = [
    ('    ap.add_argument("--out", default=os.path.join(',
     '    ap.add_argument("--seed", type=int, default=0,\n'
     '                    help="Replicate index. 0 reproduces the runs already on disk exactly; any other "\n'
     '                         "value shifts both the LoRA training seed and the generation seeds, giving an "\n'
     '                         "independent draw of the same cell.")\n'
     '    ap.add_argument("--out", default=os.path.join('),

    ('losses = train_lora_instance(pipe, train_imgs, args.prompt, args.steps, args.lr, seed=0,',
     'losses = train_lora_instance(pipe, train_imgs, args.prompt, args.steps, args.lr, seed=args.seed,'),

    ('                                        args.n_generate, seed_offset=1000)',
     '                                        args.n_generate, seed_offset=1000 + 1000 * args.seed)'),

    ('''            "final_train_loss": losses[-1],
        }''',
     '''            "final_train_loss": losses[-1],
            "seed": args.seed,
        }'''),
]

for old, new in EDITS:
    if s.count(old) != 1:
        print("FAIL %d occurrences of: %s" % (s.count(old), old[:70]))
        sys.exit(1)
    s = s.replace(old, new)

open(P, "w", encoding="utf-8").write(s)
print("patched %d edits (%d -> %d chars)" % (len(EDITS), len(orig), len(s)))
