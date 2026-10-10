#!/bin/bash
# Restart the three jobs the disk failure interrupted, in value order:
#   1. complete PhotoGuard at a uniform n=30 so the matched-budget head-to-head
#      covers all twenty identities against one floor rather than two;
#   2. the budget and seed sweeps, which decide whether identity_00's one miss
#      at 80 steps was over-fitting or a draw;
#   3. the cohort expansion to ~59 identities.
# Nothing from the completed fast-EOT campaign is re-run: fsck left lost+found
# empty and all 417 result files verified intact.
set -u
cd /mnt/data/certimmune/src
bash pg_complete.sh   2>&1 | sed 's/^/[pg] /'
bash budget_curve.sh  2>&1 | sed 's/^/[bc] /'
bash cohort60.sh      2>&1 | sed 's/^/[ch] /'
echo RERUN_ALL_DONE
