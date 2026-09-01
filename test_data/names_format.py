"""
names_format.py

Filename builders + path helpers that return full pathlib.Path objects.
Use these from notebooks so you never hand-write f-strings again.

# Groups supported:
- ORIGINAL: subjects 100-102, days 1-5
- LABELED: subjects 104-106, movements w/j/wj/jw, trials 1-5, directions f/b
- CURRENT: subject 110, days 1-3, sessions 1-3, optional session parts
- REPETITIVE: subjects 001+, days 1-3, session parts s/m/e, tasks w/j/c/a
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

CSVKind = Literal["pos", "rot"]

# ============================================================
# Allowed sets (edit as needed)
# ============================================================
# ORIGINAL group
ORIGINAL_SUBJECTS = {"100", "101", "102"}
ORIGINAL_DAYS = {"1", "2", "3", "4", "5"}

# LABELED group
LABELED_SUBJECTS = {"104", "105", "106"}
LABELED_MOVEMENTS = {"w", "j", "wj", "jw"}  # walk, jump, walk-jump, jump-walk
LABELED_TRIALS = {"1", "2", "3", "4", "5"}
LABELED_DIRECTIONS = {"f", "b"}  # forward, backward

# CURRENT group
CURRENT_SUBJECTS = {"110", "111", "112", "113", "114", "115", "116"} # Add more after running more subjects!
CURRENT_DAYS = {"1", "2", "3"}
CURRENT_SESSIONS = {"1", "2", "3", "4"}


# REPETITIVE group
REPETITIVE_DAYS = {"1", "2", "3"} 
REPETITIVE_SESSION_PARTS = {"s", "m", "e"} # start, middle, end
REPETITIVE_TASKS = {"w", "j", "c", "a"} # walk, jump, climb, a?

# ============================================================
# Validation helpers
# ============================================================
def _validate_in(value: str, allowed: set[str], name: str) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}; got {value!r}")


def _validate_numeric(value: str, name: str) -> None:
    if not value.isdigit():
        raise ValueError(f"{name} must be a numeric string; got {value!r}")

def _validate_numeric_len(value: str, name: str, length: int) -> None:
    _validate_numeric(value, name)
    if len(value) != length:
        raise ValueError(f"{name} must be a {length}-digit string; got {value!r}")


def _csv_suffix(kind: CSVKind, aligned: bool) -> str:
    return f"{kind}_aligned.csv" if aligned else f"{kind}.csv"


# ============================================================
# ORIGINAL group
# ============================================================
class Original:
    @staticmethod
    def csv_name(subject: str, day: str, kind: CSVKind, aligned: bool = False, validate: bool = True) -> str:
        if validate:
            _validate_in(subject, ORIGINAL_SUBJECTS, "subject")
            _validate_in(day, ORIGINAL_DAYS, "day")
        return f"Subj_{subject}_Day_{day}_{_csv_suffix(kind, aligned)}"

    @staticmethod
    def bvh_name(subject: str, day: str, validate: bool = True) -> str:
        if validate:
            _validate_in(subject, ORIGINAL_SUBJECTS, "subject")
            _validate_in(day, ORIGINAL_DAYS, "day")
        return f"{subject}-{day}.BVH"

    @staticmethod
    def paths(raw_data_dir: Path, processed_data_dir: Path, bvh_dir: Path, subject: str, day: str, validate: bool = True) -> "AlignPaths":
        return AlignPaths(
            rot_in=raw_data_dir / Original.csv_name(subject, day, "rot", aligned=False, validate=validate),
            pos_in=raw_data_dir / Original.csv_name(subject, day, "pos", aligned=False, validate=validate),
            rot_out=processed_data_dir / Original.csv_name(subject, day, "rot", aligned=True, validate=validate),
            pos_out=processed_data_dir / Original.csv_name(subject, day, "pos", aligned=True, validate=validate),
            bvh=bvh_dir / Original.bvh_name(subject, day, validate=validate),
        )


# ============================================================
# LABELED group
# ============================================================
class Labeled:
    @staticmethod
    def csv_name(
        subj_id: str,
        movement: str,
        trial: str,
        direction: str,
        kind: CSVKind,
        aligned: bool = False,
        validate: bool = True,
    ) -> str:
        if validate:
            _validate_in(subj_id, LABELED_SUBJECTS, "subj_id")
            _validate_in(movement, LABELED_MOVEMENTS, "movement")
            _validate_in(trial, LABELED_TRIALS, "trial")
            _validate_in(direction, LABELED_DIRECTIONS, "direction")

        return f"Subj_{subj_id}_{movement}_{trial}_{direction}_{_csv_suffix(kind, aligned)}"

    @staticmethod
    def bvh_name(subj_id: str, movement: str, trial: str, direction: str, validate: bool = True) -> str:
        if validate:
            _validate_in(subj_id, LABELED_SUBJECTS, "subj_id")
            _validate_in(movement, LABELED_MOVEMENTS, "movement")
            _validate_in(trial, LABELED_TRIALS, "trial")
            _validate_in(direction, LABELED_DIRECTIONS, "direction")
        return f"{subj_id}-{movement}-{trial}-{direction}.BVH"

    @staticmethod
    def paths(
        raw_data_dir: Path,
        processed_data_dir: Path,
        bvh_dir: Path,
        subj_id: str,
        movement: str,
        trial: str,
        direction: str,
        validate: bool = True,
    ) -> "AlignPaths":
        return AlignPaths(
            rot_in=raw_data_dir / Labeled.csv_name(subj_id, movement, trial, direction, "rot", aligned=False, validate=validate),
            pos_in=raw_data_dir / Labeled.csv_name(subj_id, movement, trial, direction, "pos", aligned=False, validate=validate),
            rot_out=processed_data_dir / Labeled.csv_name(subj_id, movement, trial, direction, "rot", aligned=True, validate=validate),
            pos_out=processed_data_dir / Labeled.csv_name(subj_id, movement, trial, direction, "pos", aligned=True, validate=validate),
            bvh=bvh_dir / Labeled.bvh_name(subj_id, movement, trial, direction, validate=validate),
        )


# ============================================================
# CURRENT group
# ============================================================
class Current:
    @staticmethod
    def bvh_name(subj_id: str, day: str, session: str, part: Optional[str] = None, validate: bool = True) -> str:
        if validate:
            _validate_in(subj_id, CURRENT_SUBJECTS, "subj_id")
            _validate_in(day, CURRENT_DAYS, "day")
            _validate_in(session, CURRENT_SESSIONS, "session")
            if part is not None:
                _validate_numeric(part, "part")
        return f"{subj_id}-{day}-{session}.BVH" if part is None else f"{subj_id}-{day}-{session}.{part}.BVH"

    @staticmethod
    def csv_name(
        subj_id: str,
        day: str,
        session: str,
        kind: CSVKind,
        aligned: bool = False,
        part: Optional[str] = None,
        validate: bool = True,
    ) -> str:
        if validate:
            _validate_in(subj_id, CURRENT_SUBJECTS, "subj_id")
            _validate_in(day, CURRENT_DAYS, "day")
            _validate_in(session, CURRENT_SESSIONS, "session")
            if part is not None:
                _validate_numeric(part, "part")

        if part is None:
            return f"Subj_{subj_id}_Day_{day}_Session_{session}_{_csv_suffix(kind, aligned)}"
        return f"Subj_{subj_id}_Day_{day}_Session_{session}_Part_{part}_{_csv_suffix(kind, aligned)}"

    @staticmethod
    def paths(
        raw_data_dir: Path,
        processed_data_dir: Path,
        bvh_dir: Path,
        subj_id: str,
        day: str,
        session: str,
        part: Optional[str] = None,
        validate: bool = True,
    ) -> "AlignPaths":
        return AlignPaths(
            rot_in=raw_data_dir / Current.csv_name(subj_id, day, session, "rot", aligned=False, part=part, validate=validate),
            pos_in=raw_data_dir / Current.csv_name(subj_id, day, session, "pos", aligned=False, part=part, validate=validate),
            rot_out=processed_data_dir / Current.csv_name(subj_id, day, session, "rot", aligned=True, part=part, validate=validate),
            pos_out=processed_data_dir / Current.csv_name(subj_id, day, session, "pos", aligned=True, part=part, validate=validate),
            bvh=bvh_dir / Current.bvh_name(subj_id, day, session, part=part, validate=validate),
        )




# ============================================================
# REPETITIVE group
# ============================================================
class Repetitive:
    @staticmethod
    def bvh_name(
        participant: str,
        day: str,
        session_part: str,
        task: str,
        part: Optional[str] = None,
        validate: bool = True,
    ) -> str:
        """Examples:
        - no part: 001-1-s-w.BVH
        - with part: 001-1-s-w_2.BVH
        """
        if validate:
            _validate_numeric_len(participant, "participant", 3)
            _validate_in(day, REPETITIVE_DAYS, "day")
            _validate_in(session_part, REPETITIVE_SESSION_PARTS, "session_part")
            _validate_in(task, REPETITIVE_TASKS, "task")
            if part is not None:
                _validate_numeric(part, "part")

        base = f"{participant}-{day}-{session_part}-{task}"
        return f"{base}.BVH" if part is None else f"{base}_{part}.BVH"

    @staticmethod
    def csv_name(
        participant: str,
        day: str,
        session_part: str,
        task: str,
        kind: CSVKind,
        aligned: bool = False,
        part: Optional[str] = None,
        validate: bool = True,
    ) -> str:
        """Examples:
        - raw, no part: 001-1-s-w_pos.csv
        - aligned, no part: 001-1-s-w_pos_aligned.csv
        - raw, with part: 001-1-s-w_2_pos.csv
        - aligned, with part: 001-1-s-w_2_pos_aligned.csv
        """
        if validate:
            _validate_numeric_len(participant, "participant", 3)
            _validate_in(day, REPETITIVE_DAYS, "day")
            _validate_in(session_part, REPETITIVE_SESSION_PARTS, "session_part")
            _validate_in(task, REPETITIVE_TASKS, "task")
            if part is not None:
                _validate_numeric(part, "part")

        base = f"{participant}-{day}-{session_part}-{task}"
        if part is not None:
            base = f"{base}_{part}"
        return f"{base}_{_csv_suffix(kind, aligned)}"

    @staticmethod
    def paths(
        raw_data_dir: Path,
        processed_data_dir: Path,
        bvh_dir: Path,
        participant: str,
        day: str,
        session_part: str,
        task: str,
        part: Optional[str] = None,
        validate: bool = True,
    ) -> "AlignPaths":
        return AlignPaths(
            rot_in=raw_data_dir / Repetitive.csv_name(participant, day, session_part, task, "rot", aligned=False, part=part, validate=validate),
            pos_in=raw_data_dir / Repetitive.csv_name(participant, day, session_part, task, "pos", aligned=False, part=part, validate=validate),
            rot_out=processed_data_dir / Repetitive.csv_name(participant, day, session_part, task, "rot", aligned=True, part=part, validate=validate),
            pos_out=processed_data_dir / Repetitive.csv_name(participant, day, session_part, task, "pos", aligned=True, part=part, validate=validate),
            bvh=bvh_dir / Repetitive.bvh_name(participant, day, session_part, task, part=part, validate=validate),
        )

# ============================================================
# Common container for alignment I/O
# ============================================================
@dataclass(frozen=True)
class AlignPaths:
    rot_in: Path
    pos_in: Path
    rot_out: Path
    pos_out: Path
    bvh: Path


# ============================================================
# Convenience runner (optional)
# ============================================================
def align_one(ma_module, paths: AlignPaths, *, reference_frame: int = 1, align_rot: bool = True, align_pos: bool = True, **kwargs):
    """
    Thin wrapper so notebooks can be 3 lines:
      p = Original.paths(...)
      align_one(ma, p, reference_frame=1)
    """
    return ma_module.align_motion_data(
        rot_file_path=paths.rot_in,
        pos_file_path=paths.pos_in,
        output_rot_path=paths.rot_out,
        output_pos_path=paths.pos_out,
        bvh_path=paths.bvh,
        reference_frame=reference_frame,
        align_rot=align_rot,
        align_pos=align_pos,
        **kwargs,
    )
