"""Process-pool helpers for per-image batch transforms."""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")
R = TypeVar("R")


def resolve_jobs(jobs: int) -> int:
    cpu_count = os.cpu_count() or 1
    if jobs == 0:
        return cpu_count
    if jobs < 0:
        return max(1, cpu_count + jobs)
    return max(1, jobs)


def parallel_map(
    func: Callable[[T], R],
    items: list[T],
    *,
    jobs: int,
    desc: str,
) -> list[R]:
    if not items:
        return []

    worker_count = resolve_jobs(jobs)
    if worker_count == 1 or len(items) == 1:
        return [
            func(item)
            for item in tqdm(
                items,
                desc=desc,
                unit="image",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
            )
        ]

    results: list[R | None] = [None] * len(items)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(func, item): index
            for index, item in enumerate(items)
        }
        with tqdm(
            total=len(items),
            desc=desc,
            unit="image",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
        ) as progress:
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                results[index] = future.result()
                progress.update(1)

    return results  # type: ignore[return-value]
