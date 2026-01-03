import argparse
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import requests


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] * (c - k) + xs[c] * (k - f)


def load_tasks(path: str) -> List[Dict]:
    tasks: List[Dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


#  fast polling: reuse a Session per thread
_thread_local = threading.local()


def get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def fetch_status(base_url: str, task_id: str, timeout: float) -> Tuple[str, Dict]:
    url = f"{base_url}/result/{task_id}"
    session = get_session()
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return task_id, r.json()


def extract_submit_ts(meta: Optional[Dict], fallback: float) -> float:
    if not meta:
        return fallback
    for k in ("client_queued_at", "submitted_at", "server_queued_at"):
        v = meta.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll /result/{task_id} until tasks complete.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--in", dest="infile", default="artifacts/tasks.jsonl", help="Input JSONL from load_test.py")
    parser.add_argument("--interval-ms", type=int, default=200, help="Polling interval per loop in ms")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout seconds")
    parser.add_argument("--max-wait-s", type=int, default=600, help="Max total wait time in seconds")
    parser.add_argument("--poll-concurrency", type=int, default=50, help="Parallel status checks per loop")
    parser.add_argument("--poll-limit", type=int, default=1000, help="Max task IDs to poll per loop")
    parser.add_argument("--out", default="artifacts/results.jsonl", help="Output results JSONL file")
    args = parser.parse_args()

    tasks = load_tasks(args.infile)
    if not tasks:
        print(f"No tasks found in {args.infile}")
        return

    pending: Dict[str, Dict] = {t["task_id"]: t for t in tasks}
    done: Dict[str, Dict] = {}
    failed: Dict[str, Dict] = {}

    Path("artifacts").mkdir(parents=True, exist_ok=True)

    print(f"Polling {len(pending)} tasks from {args.infile}")
    t_start = time.time()
    deadline = t_start + args.max_wait_s

    while pending and time.time() < deadline:
        loop_start = time.time()

        ids = list(pending.keys())[: args.poll_limit]

        completed_this_loop = 0
        with ThreadPoolExecutor(max_workers=args.poll_concurrency) as ex:
            futs = [ex.submit(fetch_status, args.base_url, task_id, args.timeout) for task_id in ids]
            for fut in as_completed(futs):
                try:
                    task_id, resp = fut.result()
                    status = resp.get("status", "UNKNOWN")

                    if status in ("SUCCESS", "FAILURE"):
                        meta = pending.pop(task_id, None)
                        submit_ts = extract_submit_ts(meta, t_start)
                        now = time.time()

                        if status == "SUCCESS":
                            result = resp.get("result") or {}
                            timing = result.get("timing") or {}
                            compute_s = timing.get("compute_s")

                            done[task_id] = {
                                "task_id": task_id,
                                "status": status,
                                "result": result,
                                "observed_latency_s": now - submit_ts,
                                "compute_s": float(compute_s) if isinstance(compute_s, (int, float)) else None,
                                "submit_ts": submit_ts,
                                "observed_done_ts": now,
                            }
                        else:
                            failed[task_id] = {
                                "task_id": task_id,
                                "status": status,
                                "error": resp.get("error"),
                                "observed_latency_s": now - submit_ts,
                                "submit_ts": submit_ts,
                                "observed_done_ts": now,
                            }

                        completed_this_loop += 1

                except Exception:
                    pass

        print(
            f"Pending: {len(pending)} | Done: {len(done)} | Failed: {len(failed)} | "
            f"Completed(loop): {completed_this_loop}",
            end="\r",
        )

        elapsed = time.time() - loop_start
        sleep_s = max(0.0, (args.interval_ms / 1000.0) - elapsed)
        if sleep_s:
            time.sleep(sleep_s)

    total_time = time.time() - t_start

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for item in done.values():
            f.write(json.dumps(item) + "\n")
        for item in failed.values():
            f.write(json.dumps(item) + "\n")

    latencies = [x["observed_latency_s"] for x in done.values()]
    compute_times = [x["compute_s"] for x in done.values() if x.get("compute_s") is not None]

    completed = len(done)
    total = len(tasks)
    throughput = completed / total_time if total_time > 0 else 0.0

    print("\n\n=== Poll Summary ===")
    print(f"Total tasks:              {total}")
    print(f"Completed:                {completed}")
    print(f"Failed:                   {len(failed)}")
    print(f"Still pending:            {len(pending)}")
    print(f"Total wait (s):           {total_time:.2f}")
    print(f"Throughput (t/s):         {throughput:.2f}")

    if latencies:
        print(f"Avg observed latency (s): {sum(latencies)/len(latencies):.3f}")
        print(f"p50 observed latency (s): {percentile(latencies, 50):.3f}")
        print(f"p95 observed latency (s): {percentile(latencies, 95):.3f}")
        print(f"p99 observed latency (s): {percentile(latencies, 99):.3f}")
        print(f"max observed latency (s): {max(latencies):.3f}")
    else:
        print("No successful tasks to compute observed latency stats.")

    if compute_times:
        print("\n=== Worker Compute Time (from result.timing.compute_s) ===")
        print(f"Avg compute (s):          {sum(compute_times)/len(compute_times):.6f}")
        print(f"p50 compute (s):          {percentile(compute_times, 50):.6f}")
        print(f"p95 compute (s):          {percentile(compute_times, 95):.6f}")
        print(f"p99 compute (s):          {percentile(compute_times, 99):.6f}")
        print(f"max compute (s):          {max(compute_times):.6f}")
    else:
        print("\nNo compute_s found in worker results. That means the worker isn't returning timing yet.")

    print(f"\nSaved results:            {out_path}")


if __name__ == "__main__":
    main()
