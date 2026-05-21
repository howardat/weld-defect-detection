import time
import csv
from contextlib import contextmanager
from pathlib import Path

_records: list[dict] = []

@contextmanager
def track(image_name: str, module: str, call_type: str, call_index: int = 0):
    start = time.perf_counter()
    yield
    _records.append({
        "image_name": image_name,
        "module": module,
        "call_type": call_type,
        "call_index": call_index,
        "duration_s": round(time.perf_counter() - start, 4),
    })

def save_csv(path: str | Path) -> None:
    if not _records:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_name", "module", "call_type", "call_index", "duration_s"]
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(_records)

def clear() -> None:
    _records.clear()
