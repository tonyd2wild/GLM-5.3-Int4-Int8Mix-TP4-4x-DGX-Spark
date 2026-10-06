#!/usr/bin/env python3
"""guard.py OWN TRIES -- CMD ...   run CMD on an idle server; retry while other clients interfered.
Waits for running+waiting == 0 for 5 s, samples it every 0.5 s while CMD runs, and calls the run clean
when the server never saw more than OWN requests. Prints one GUARD line per attempt. Stdlib only."""
import subprocess, sys, threading, time, urllib.request

URL = "http://127.0.0.1:8000/metrics"


def busy():
    try:
        txt = urllib.request.urlopen(URL, timeout=10).read().decode()
    except Exception:
        return None
    tot = 0.0
    for line in txt.splitlines():
        if line.startswith("vllm:num_requests_running{") or line.startswith("vllm:num_requests_waiting{"):
            tot += float(line.rsplit(" ", 1)[1])
    return tot


def wait_idle(quiet=5.0, limit=5400):
    t0 = time.time(); since = None
    while time.time() - t0 < limit:
        b = busy()
        if b == 0:
            since = since or time.time()
            if time.time() - since >= quiet:
                return time.time() - t0
        else:
            since = None
        time.sleep(0.5)
    return None


def main():
    own, tries = int(sys.argv[1]), int(sys.argv[2])
    cmd = sys.argv[sys.argv.index("--") + 1:]
    for t in range(tries):
        waited = wait_idle()
        peak = [0.0]; stop = threading.Event()

        def sample():
            while not stop.is_set():
                b = busy()
                if b is not None:
                    peak[0] = max(peak[0], b)
                time.sleep(0.5)
        th = threading.Thread(target=sample, daemon=True); th.start()
        rc = subprocess.call(cmd)
        stop.set(); th.join()
        clean = peak[0] <= own
        print(f"GUARD attempt={t + 1} waited_s={round(waited or -1, 1)} max_busy={int(peak[0])} own={own} "
              f"clean={clean} rc={rc}", file=sys.stderr, flush=True)
        if clean:
            sys.exit(rc)
    sys.exit(3)


main()
