import argparse
import json
import random
import string
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

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


def random_text(min_words: int = 8, max_words: int = 30) -> str:
    words = random.randint(min_words, max_words)
    vocab = [
        "great", "bad", "awesome", "terrible", "fast", "slow", "love", "hate", "product",
        "service", "experience", "support", "quality", "price", "recommend", "never",
        "again", "satisfied", "disappointed", "amazing", "okay", "excellent", "worst"
    ]
    tokens = [random.choice(vocab) for _ in range(words)]
    noise = "".join(random.choices(string.ascii_lowercase, k=10))
    return " ".join(tokens) + f" ({noise})"


def post_inference(session: requests.Session, base_url: str, text: str, timeout: float) -> dict:
    url = f"{base_url}/inference"

    t0 = time.time()
    r = session.post(url, json={"text": text}, timeout=timeout)
    t1 = time.time()

    r.raise_for_status()
    data = r.json()

    return {
        "task_id": data["task_id"],
        "enqueue_s": t1 - t0,
        "client_sent_at": t0, # time when request started
        "client_queued_at": t1, # time hen API responded with task_id
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Load test the /inference endpoint.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--requests", type=int, default=200, help="Total number of requests")
    parser.add_argument("--concurrency", type=int, default=20, help="Concurrent workers")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout seconds")
    parser.add_argument("--out", default="artifacts/tasks.jsonl", help="Output JSONL file path")
    args = parser.parse_args()

    Path("artifacts").mkdir(parents=True, exist_ok=True)

    results = []
    errors = 0

    print(f"Sending {args.requests} requests to {args.base_url}/inference "
          f"(concurrency={args.concurrency})")

    t_start = time.time()

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = [
                ex.submit(post_inference, session, args.base_url, random_text(), args.timeout)
                for _ in range(args.requests)
            ]
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    errors += 1
                    if errors <= 10:
                        print(f"[ERROR] {e}")

    t_total = time.time() - t_start

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    ok = len(results)
    rps = ok / t_total if t_total > 0 else 0.0

    enqueue_times = [x["enqueue_s"] for x in results]
    avg_enqueue = sum(enqueue_times) / ok if ok else 0.0

    print("\n=== Load Test Summary ===")
    print(f"Queued tasks:        {ok}")
    print(f"Errors:              {errors}")
    print(f"Total time (s):      {t_total:.2f}")
    print(f"Queue rate (rps):    {rps:.2f}")
    if enqueue_times:
        print(f"Avg enqueue (s):     {avg_enqueue:.3f}")
        print(f"p50 enqueue (s):     {percentile(enqueue_times, 50):.3f}")
        print(f"p95 enqueue (s):     {percentile(enqueue_times, 95):.3f}")
        print(f"p99 enqueue (s):     {percentile(enqueue_times, 99):.3f}")
    print(f"Saved task_ids:      {out_path}")


if __name__ == "__main__":
    main()
