"""Disk cache for the forecasting benchmark's expensive stages.

Detector fits and forecast-model backtests dominate the benchmark's runtime,
so both are cached on disk keyed by *content*: every key embeds a fingerprint
of the exact training values plus every config knob that influences the stage.
An interrupted run therefore resumes where it stopped, and a finished run can
be re-executed (e.g. after adding a strategy or a forecast model) recomputing
only what changed — editing the data or any relevant option changes the key
and invalidates the entry automatically. Delete the cache directory to clear.

Entries are pickles named by a SHA-256 of the canonical-JSON key; the full key
is stored inside each entry and verified on read, so a hash collision or a
stale format degrades to a cache miss, never to wrong reuse.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import marshal
import os
import pickle
import tempfile
from collections import Counter
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Bump when the cached payloads change shape or meaning (keys embed it, so
#: old entries simply stop matching instead of being misread). v2: backtest
#: MASE switched to the shared raw-history insample. v3: causal train/val split
#: (validation is the most recent block; posterior blocks dropped from train).
#: v4: the entire variable-length evaluation block is excluded from training.
#: v5: rolling validation/test origins use explicit per-regime strides. v6:
#: keys include effective config, artifact content, and transform behavior. v7:
#: forecasting-only local/foundation model definitions and Darts 0.46 support.
#: v8: full-series detection and fixed holdout selected from common masks.
#: v9: each detector is fitted once on all block-local windows of a station.
#: v10: injected rankings and detector eligibility are resolved per block. v11:
#: detector coverage is finite-score-aware and failed backtests are not persisted.
#: v12: test terminology is explicit and the paired synthetic foundation-context
#: experiment has its own content-keyed payloads. v13: detector semantics use
#: TSB-AD Sub_PCA, full-window Hampel, and timestamp-aware Prophet. v14: CARLA
#: training stride is an explicit forecasting configuration. v15: centered
#: Hampel uses an odd effective window for even nominal hourly spans. v16:
#: common holdout support ignores abstentions and raw+frozen is a source arm. v17:
#: inject-vote uses ranked pointwise backfill with its configured quorum.
CACHE_VERSION = 17


def series_fingerprint(series: pd.Series) -> str:
    """Content hash of one series: exact values (NaN included) + index span.

    Two series with the same name but different data (e.g. after re-scraping)
    get different fingerprints, so cache entries never survive a data change.
    """
    values = np.ascontiguousarray(series.to_numpy(dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(values.tobytes())
    index = series.index
    meta = f"{series.name}|{len(series)}|{index[0] if len(index) else ''}|{index[-1] if len(index) else ''}"
    digest.update(meta.encode("utf-8"))
    return digest.hexdigest()[:16]


def artifact_fingerprint(path: Path | str | None) -> str | None:
    """Content hash of one local artifact file or directory."""
    if path is None:
        return None
    root = Path(path).expanduser()
    if not root.exists():
        return None

    files = (
        [root]
        if root.is_file()
        else sorted(item for item in root.rglob("*") if item.is_file())
    )
    digest = hashlib.sha256()
    for item in files:
        name = item.name if root.is_file() else item.relative_to(root).as_posix()
        digest.update(f"{name}\0{item.stat().st_size}\0".encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()[:24]


def effective_config(config: Any) -> Any:
    """Return cache-relevant config with runtime device selection removed."""
    if isinstance(config, Mapping):
        return {
            str(key): effective_config(value)
            for key, value in config.items()
            if str(key).lower() not in {"device", "accelerator"}
        }
    if isinstance(config, (list, tuple)):
        return [effective_config(value) for value in config]
    return config


def _key_hash(key: dict[str, Any]) -> str:
    """Stable hash of a JSON-serializable key dict (order-independent)."""
    canonical = json.dumps(key, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


class BenchmarkCache:
    """Namespace → key-dict → pickled value store under one root directory.

    Construct with ``root=None`` to disable caching (every ``get`` misses and
    ``put`` is a no-op), so the pipeline code stays branch-free.
    """

    def __init__(self, root: Path | None) -> None:
        self.root = root
        self.hits = 0
        self.misses = 0
        self.hits_by_namespace: Counter[str] = Counter()
        self.misses_by_namespace: Counter[str] = Counter()

    @property
    def enabled(self) -> bool:
        return self.root is not None

    def _path(self, namespace: str, key: dict[str, Any]) -> Path:
        return self.root / namespace / f"{_key_hash(key)}.pkl"

    def get(self, namespace: str, key: dict[str, Any]) -> Any | None:
        """Return the cached value for ``key`` or ``None`` on any miss."""
        if not self.enabled:
            return None
        path = self._path(namespace, key)
        try:
            with path.open("rb") as handle:
                entry = pickle.load(handle)
        except FileNotFoundError:
            self.misses += 1
            self.misses_by_namespace[namespace] += 1
            return None
        except Exception as exc:  # corrupt/stale entry -> recompute
            logging.warning("[cache] entrada ilegible %s (%s); se recalcula", path.name, exc)
            self.misses += 1
            self.misses_by_namespace[namespace] += 1
            return None
        if (
            not isinstance(entry, Mapping)
            or entry.get("key") != key
            or "value" not in entry
        ):  # hash collision or format drift
            self.misses += 1
            self.misses_by_namespace[namespace] += 1
            return None
        self.hits += 1
        self.hits_by_namespace[namespace] += 1
        return entry["value"]

    def put(self, namespace: str, key: dict[str, Any], value: Any) -> None:
        """Persist ``value`` under ``key`` (atomic write via temp file + rename)."""
        if not self.enabled:
            return
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                pickle.dump({"key": key, "value": value}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_name, path)
        except Exception:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def stats(self) -> str:
        """One-line usage summary for logging."""
        if not self.enabled:
            return "cache deshabilitada"
        details = "; ".join(
            f"{namespace}={counts['hits']} aciertos/{counts['misses']} fallos"
            for namespace, counts in self.stats_by_namespace().items()
        )
        suffix = f"; {details}" if details else ""
        return (
            f"{self.hits} aciertos / {self.misses} fallos en {self.root}"
            f"{suffix}"
        )

    def stats_by_namespace(self) -> dict[str, dict[str, int]]:
        """Return hit/miss counters grouped by cache namespace."""
        namespaces = sorted(
            self.hits_by_namespace.keys() | self.misses_by_namespace.keys()
        )
        return {
            namespace: {
                "hits": self.hits_by_namespace[namespace],
                "misses": self.misses_by_namespace[namespace],
            }
            for namespace in namespaces
        }


def _stable_state(value: Any, seen: set[int] | None = None) -> Any:
    """Reduce common callable state to deterministic JSON-compatible values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"bytes": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Path):
        return {"path": str(value)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "array": hashlib.sha256(array.tobytes()).hexdigest(),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }

    seen = set() if seen is None else seen
    if id(value) in seen:
        return {"cycle": f"{type(value).__module__}.{type(value).__qualname__}"}
    seen.add(id(value))
    try:
        if isinstance(value, Mapping):
            items = [
                (_stable_state(key, seen), _stable_state(item, seen))
                for key, item in value.items()
            ]
            return {
                "mapping": sorted(
                    items, key=lambda pair: json.dumps(pair[0], sort_keys=True)
                )
            }
        if isinstance(value, (list, tuple)):
            return {type(value).__name__: [_stable_state(item, seen) for item in value]}
        if isinstance(value, (set, frozenset)):
            items = [_stable_state(item, seen) for item in value]
            return {
                type(value).__name__: sorted(
                    items, key=lambda item: json.dumps(item, sort_keys=True)
                )
            }
        if callable(value):
            return {"callable": _transform_fingerprint(value)}
        state = _object_state(value)
        if state:
            return {
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "state": _stable_state(state, seen),
            }
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "repr": repr(value),
        }
    finally:
        seen.remove(id(value))


def _object_state(value: Any) -> dict[str, Any] | None:
    state = dict(vars(value)) if hasattr(value, "__dict__") else {}
    for cls in type(value).__mro__:
        slots = getattr(cls, "__slots__", ())
        slots = (slots,) if isinstance(slots, str) else slots
        state.update(
            (slot, getattr(value, slot))
            for slot in slots
            if slot not in {"__dict__", "__weakref__"} and hasattr(value, slot)
        )
    return state or None


def _transform_fingerprint(transform: Any) -> str:
    if isinstance(transform, partial):
        implementation = transform.func
        state = {"args": transform.args, "keywords": transform.keywords}
    elif inspect.ismethod(transform):
        implementation = transform.__func__
        state = _object_state(transform.__self__)
    elif inspect.isfunction(transform):
        implementation = transform
        state = vars(transform) or None
    else:
        implementation = type(transform).__call__
        state = _object_state(transform)

    name = getattr(transform, "__qualname__", type(transform).__qualname__)
    module = getattr(transform, "__module__", type(transform).__module__)
    code = getattr(implementation, "__code__", None)
    closure = getattr(implementation, "__closure__", None)
    payload = {
        "name": f"{module}.{name}",
        "code": hashlib.sha256(marshal.dumps(code)).hexdigest() if code is not None else None,
        "defaults": _stable_state(getattr(implementation, "__defaults__", None)),
        "kwdefaults": _stable_state(getattr(implementation, "__kwdefaults__", None)),
        "closure": (
            _stable_state(tuple(cell.cell_contents for cell in closure)) if closure else None
        ),
        "state": _stable_state(state),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    return f"{module}.{name}:{digest}"


def transform_fingerprints(transforms) -> list[str]:
    """Stable implementation-and-state identities for mask transforms."""
    if not transforms:
        return []
    return [_transform_fingerprint(transform) for transform in transforms]


__all__ = [
    "BenchmarkCache",
    "CACHE_VERSION",
    "artifact_fingerprint",
    "effective_config",
    "series_fingerprint",
    "transform_fingerprints",
]
