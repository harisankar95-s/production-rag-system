import time
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_timings = {}

@contextmanager
def timer(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = round(time.perf_counter() - start, 3)
        _timings[label] = elapsed
        logger.info(f"TIMER | {label:<40} | {elapsed}s")

def get_timings() -> dict:
    return dict(_timings)

def reset_timings():
    _timings.clear()

def print_breakdown():
    if not _timings:
        print("No timings recorded.")
        return
    total = sum(_timings.values())
    print("\n" + "=" * 55)
    print("LATENCY BREAKDOWN")
    print("=" * 55)
    for label, elapsed in sorted(_timings.items(), key=lambda x: x[1], reverse=True):
        pct = round((elapsed / total) * 100, 1) if total > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"{label:<35} {elapsed:>6}s  {pct:>5}%  {bar}")
    print("-" * 55)
    print(f"{'TOTAL':<35} {round(total,3):>6}s  100%")
    print("=" * 55)