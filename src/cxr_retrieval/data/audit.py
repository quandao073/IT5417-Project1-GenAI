"""Cong chan Phase 1: do ti le join giua CheXpert Plus va CheXpert-v1.0-small.

Xem §2.3 cua ke hoach. Khong tai anh truoc khi cong chan nay dat.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

FULL_CORPUS_MIN_JOIN_RATE = 0.95
SAMPLED_CORPUS_MIN_JOIN_RATE = 0.70


class CorpusMode(Enum):
    """Quyet dinh corpus rut ra tu ti le join."""

    FULL_FRONTAL = "full_frontal"
    SAMPLED_25K = "sampled_25k"
    ABORT = "abort"


def decide_corpus_mode(join_rate: float) -> CorpusMode:
    """Ap bang quyet dinh §2.3 len ti le join da do."""
    if not 0.0 <= join_rate <= 1.0:
        raise ValueError(f"join_rate phai nam trong [0, 1], nhan duoc {join_rate}")
    if join_rate >= FULL_CORPUS_MIN_JOIN_RATE:
        return CorpusMode.FULL_FRONTAL
    if join_rate >= SAMPLED_CORPUS_MIN_JOIN_RATE:
        return CorpusMode.SAMPLED_25K
    return CorpusMode.ABORT


_TRIPLE = re.compile(r"(patient\d+)/(study\d+)/([^/]+?)(?:\.[A-Za-z0-9]+)?$")


def image_key(path: str) -> str | None:
    """Rut khoa `patientN/studyN/viewN_*` bo prefix va duoi file.

    Tra None khi duong dan khong chua bo ba nay, de goi y goi phan biet
    duoc "khong khop" voi "khong phai duong dan anh".
    """
    match = _TRIPLE.search(path.replace("\\", "/"))
    if match is None:
        return None
    return "/".join(match.groups())


def _exact(path: str) -> str | None:
    return path


def _posix_lower(path: str) -> str | None:
    return path.replace("\\", "/").lower()


# Tu chat den long. Chien luoc nao thang thi ghi vao data_audit.json va dung
# no cho buoc join that o Phase 2.
STRATEGIES: dict[str, Callable[[str], str | None]] = {
    "exact": _exact,
    "posix_lower": _posix_lower,
    "image_key": image_key,
}


@dataclass(frozen=True)
class JoinReport:
    """Ket qua do join cua mot chien luoc chuan hoa."""

    strategy: str
    plus_total: int
    small_total: int
    matched: int
    join_rate: float
    unmatched_samples: tuple[str, ...]


def evaluate_join(
    plus_paths: Iterable[str],
    small_paths: Iterable[str],
    sample_size: int = 20,
) -> list[JoinReport]:
    """Do ti le join cho tung chien luoc, tra ve danh sach xep theo ti le giam dan.

    `join_rate` la ti le dong cua CheXpert Plus tim duoc anh tuong ung, vi day
    moi la con so quyet dinh corpus: report khong co anh thi khong dung duoc.
    """
    plus = list(plus_paths)
    small = list(small_paths)
    if not plus:
        raise ValueError("plus_paths rong: khong do duoc ti le join")

    reports = []
    for name, normalize in STRATEGIES.items():
        index = {key for key in map(normalize, small) if key is not None}
        unmatched = [p for p in plus if normalize(p) not in index]
        matched = len(plus) - len(unmatched)
        reports.append(
            JoinReport(
                strategy=name,
                plus_total=len(plus),
                small_total=len(small),
                matched=matched,
                join_rate=matched / len(plus),
                unmatched_samples=tuple(unmatched[:sample_size]),
            )
        )
    return sorted(reports, key=lambda r: r.join_rate, reverse=True)


PLUS_PATH_COLUMN = "path_to_image"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def read_plus_image_paths(csv_path: Path) -> list[str]:
    """Doc cot `path_to_image` cua df_chexpert_plus_*.csv."""
    with open(csv_path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or PLUS_PATH_COLUMN not in reader.fieldnames:
            raise KeyError(
                f"Khong thay cot {PLUS_PATH_COLUMN!r} trong {csv_path}. "
                f"Cac cot doc duoc: {reader.fieldnames}"
            )
        return [row[PLUS_PATH_COLUMN] for row in reader if row.get(PLUS_PATH_COLUMN)]


def list_small_image_paths(root: Path) -> list[str]:
    """Liet ke anh cua CheXpert-v1.0-small, tra ve duong dan POSIX tuong doi voi root."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Khong tim thay thu muc anh: {root}")
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    )


def run_audit(plus_csv: Path, small_root: Path, sample_size: int = 20) -> dict:
    """Chay cong chan Phase 1 va tra ve bao cao ghi thang ra data_audit.json."""
    plus_paths = read_plus_image_paths(plus_csv)
    small_paths = list_small_image_paths(small_root)
    reports = evaluate_join(plus_paths, small_paths, sample_size=sample_size)
    best = reports[0]

    return {
        "plus_csv": str(plus_csv),
        "small_root": str(small_root),
        "plus_total": best.plus_total,
        "small_total": best.small_total,
        "best_strategy": best.strategy,
        "join_rate": best.join_rate,
        "decision": decide_corpus_mode(best.join_rate).value,
        "strategies": [asdict(r) for r in reports],
        "unmatched_samples": list(best.unmatched_samples),
    }
