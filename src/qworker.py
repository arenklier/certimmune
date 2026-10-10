"""Job-queue worker: claims jobs atomically, runs them, records the outcome.

Several workers can run against the same phase file; os.mkdir is the atomic
claim, so a job is never picked up twice. A worker exits when nothing is left
to claim, which makes "all workers exited" the phase-complete signal.

The job runs in its own session (start_new_session), and its pgid is written
into the claim directory. Without that, shell=True leaves an intermediate
shell between the worker and the real process: killing the worker kills the
shell and orphans the GPU job, which is exactly how 6.3 GB went missing once.
stop_worker.sh reads the pgid and kills the whole group; a SIGTERM to the
worker itself now also tears the job down and releases the claim, so a stopped
job becomes re-runnable instead of permanently claimed.
"""

import os
import signal
import subprocess
import sys
import time

ROOT = "/mnt/data/certimmune"
QDIR = f"{ROOT}/queue"
SRC = f"{ROOT}/src"

_current = {"claim": None, "proc": None}


def _cleanup(signum, frame):
    proc = _current.get("proc")
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    claim = _current.get("claim")
    if claim and os.path.isdir(claim):
        for f in os.listdir(claim):
            try:
                os.remove(os.path.join(claim, f))
            except OSError:
                pass
        try:
            os.rmdir(claim)  # release so the job can be picked up again
        except OSError:
            pass
    sys.exit(143)


def main():
    phase = sys.argv[1]
    wid = sys.argv[2] if len(sys.argv) > 2 else "0"
    jobs_file = os.path.join(QDIR, f"jobs_{phase}.txt")

    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT, _cleanup)

    claim_dir = os.path.join(QDIR, "claimed", phase)
    done_dir = os.path.join(QDIR, "done", phase)
    fail_dir = os.path.join(QDIR, "failed", phase)
    log_dir = os.path.join(QDIR, "logs", phase)
    for d in (claim_dir, done_dir, fail_dir, log_dir):
        os.makedirs(d, exist_ok=True)

    jobs = []
    with open(jobs_file) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            jid, cmd = line.split("\t", 1)
            jobs.append((jid, cmd))

    ran = 0
    for jid, cmd in jobs:
        if os.path.exists(os.path.join(done_dir, jid)):
            continue
        claim = os.path.join(claim_dir, jid)
        try:
            os.mkdir(claim)  # atomic claim
        except FileExistsError:
            continue
        _current["claim"] = claim

        t0 = time.time()
        log_path = os.path.join(log_dir, f"{jid}.log")
        print(f"[w{wid}] START {jid}", flush=True)
        with open(log_path, "w") as lf:
            lf.write(f"# {cmd}\n")
            lf.flush()
            proc = subprocess.Popen(cmd, shell=True, cwd=SRC, stdout=lf,
                                    stderr=subprocess.STDOUT,
                                    start_new_session=True)
            _current["proc"] = proc
            with open(os.path.join(claim, "pgid"), "w") as pf:
                pf.write(str(os.getpgid(proc.pid)))
            rc = proc.wait()
        _current["proc"] = None
        dt = time.time() - t0
        marker = os.path.join(done_dir if rc == 0 else fail_dir, jid)
        with open(marker, "w") as mf:
            mf.write(f"rc={rc} seconds={dt:.1f}\n")
        print(f"[w{wid}] {'DONE' if rc == 0 else 'FAIL'} {jid} rc={rc} {dt:.0f}s",
              flush=True)
        _current["claim"] = None
        ran += 1

    print(f"[w{wid}] exiting, ran {ran} jobs", flush=True)


if __name__ == "__main__":
    main()
