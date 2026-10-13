"""Consistent progress reporting for long-running pipeline work."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import sys
from typing import Any, TypeVar

from tqdm import tqdm


T = TypeVar("T")


def _options(overrides: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {
        "file": sys.stdout,
        "dynamic_ncols": False,
        "mininterval": 5.0,
        "maxinterval": 30.0,
        "leave": True,
    }
    options.update(overrides)
    return options


class ManualProgress:
    """Progress bar for work completed outside an iterable."""

    def __init__(self, *, total: int | None, **kwargs: Any) -> None:
        self._completed = int(kwargs.pop("initial", 0))
        self._bar = tqdm(
            total=total,
            initial=self._completed,
            **_options(kwargs),
        )

    @property
    def enabled(self) -> bool:
        return not self._bar.disable

    def update(self, amount: int = 1) -> None:
        self.update_to(self._completed + amount)

    def update_to(self, completed: int) -> None:
        if completed <= self._completed:
            return
        difference = completed - self._completed
        self._completed = completed
        self._bar.update(difference)

    def close(self) -> None:
        self._bar.close()

    def __enter__(self) -> "ManualProgress":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def manual_progress(*, total: int | None, **kwargs: Any) -> ManualProgress:
    return ManualProgress(total=total, **kwargs)


def progress(items: Iterable[T], **kwargs: Any) -> Iterator[T]:
    yield from tqdm(items, **_options(kwargs))
