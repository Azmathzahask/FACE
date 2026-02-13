from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import pandas as pd

from detector import Detection


@dataclass
class DetectionRecord:
    """Structured record for a single detection event."""

    timestamp: float
    frame_index: int
    face_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    label: str

    @property
    def datetime_str(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))


def append_detections_to_records(
    records: List[DetectionRecord],
    detections: List[Detection],
    frame_index: int,
    ts: float | None = None,
) -> None:
    """Append a batch of detections to an in-memory list of DetectionRecord."""
    ts = ts or time.time()
    for i, det in enumerate(detections, start=1):
        records.append(
            DetectionRecord(
                timestamp=ts,
                frame_index=frame_index,
                face_id=i,
                x1=det.x1,
                y1=det.y1,
                x2=det.x2,
                y2=det.y2,
                score=det.score,
                label=det.label,
            )
        )


def records_to_dataframe(records: List[DetectionRecord]) -> pd.DataFrame:
    """Convert a list of DetectionRecord into a pandas DataFrame."""
    if not records:
        return pd.DataFrame(
            columns=[
                "datetime",
                "frame_index",
                "face_id",
                "x1",
                "y1",
                "x2",
                "y2",
                "score",
                "label",
            ]
        )
    data = []
    for r in records:
        row = asdict(r)
        row["datetime"] = r.datetime_str
        data.append(row)
    df = pd.DataFrame(data)
    # Reorder columns for readability
    cols = [
        "datetime",
        "frame_index",
        "face_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "score",
        "label",
    ]
    return df[cols]


def save_records_to_csv(records: List[DetectionRecord], path: str | Path) -> Path:
    """Save detection records to a CSV file."""
    df = records_to_dataframe(records)
    path = Path(path)
    df.to_csv(path, index=False)
    return path

