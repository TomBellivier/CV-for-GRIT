"""
parallel.py
===========

A bounded, as-completed thread-pool map -- the generalisation of the sliding
window in your test_process_hf.py.

Your script kept `buffer` DOWNLOADS in flight and yielded them in order, so the
network stayed saturated while a single consumer processed. Here the SAME idea
is applied to the whole task (download + decode + inference): a pool of
`workers` threads runs `func` on the items, at most `buffer` at a time, and
results are yielded AS THEY COMPLETE.

Why threads (and not processes) for CPU work?
    NumPy / SciPy / PyTorch release the GIL during their heavy C/CUDA sections,
    so several inferences really do run at once. Threads also share the loaded
    models cheaply (one copy per thread via worker.py) and let a thread that is
    blocked on a network download yield the GIL to a thread that is computing.
    On 16 CPUs this overlaps I/O and compute without the memory cost of 16
    separate processes.

Why bounded (the `buffer`)?
    Submitting millions of tasks at once would build millions of futures and,
    worse, could decode many images into RAM ahead of time. Keeping only
    `buffer` in flight caps memory regardless of dataset size.

Ordering: results are yielded in completion order, not input order. That is
fine because every CSV row carries its own image name.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Callable, Iterable, Iterator


def bounded_unordered_map(func: Callable, items: Iterable, workers: int,
                          buffer: int) -> Iterator:
    """Apply `func` to `items` with `workers` threads, `buffer` in flight.

    Yields (item, result, error) tuples:
        - on success: (item, result, None)
        - on failure: (item, None, exception)
    so one failing image never aborts the whole run.
    """
    buffer = max(buffer, workers)          # always enough to feed every worker
    it = iter(items)
    # Map each future back to the item it came from (for error reporting).
    future_to_item = {}

    def _submit_next(ex) -> bool:
        try:
            item = next(it)
        except StopIteration:
            return False
        future_to_item[ex.submit(func, item)] = item
        return True

    with ThreadPoolExecutor(max_workers=workers) as ex:
        # Prime the window.
        for _ in range(buffer):
            if not _submit_next(ex):
                break

        while future_to_item:
            done, _ = wait(future_to_item, return_when=FIRST_COMPLETED)
            for fut in done:
                item = future_to_item.pop(fut)
                # Refill BEFORE handling the result so the pool never idles.
                _submit_next(ex)
                try:
                    yield item, fut.result(), None
                except Exception as exc:  # noqa: BLE001 - reported, not raised
                    yield item, None, exc
