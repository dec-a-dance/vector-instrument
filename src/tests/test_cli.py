import json
import subprocess
import sys

import pandas as pd

def run_cli(args, cwd=None):
    result = subprocess.run(
        [sys.executable, "-m", "app.cli"] + args,
        capture_output=True,
        text=True,
        cwd=cwd
    )
    return result


def test_cli_full_flow(tmp_path):
    csv_path = tmp_path / "data.csv"
    db_path = tmp_path / "storage.db"

    df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [10, 20, 30]
    })
    df.to_csv(csv_path, index=False)

    r1 = run_cli(["--db", str(db_path), "ingest", "--file", str(csv_path)])
    assert r1.returncode == 0
    payload1 = json.loads(r1.stdout)
    assert payload1["ok"] is True

    r2 = run_cli([
        "--db", str(db_path),
        "set-subjective",
        "--vector", json.dumps([0.0, 0.0])
    ])
    assert r2.returncode == 0
    payload2 = json.loads(r2.stdout)
    assert payload2["ok"] is True

    r3 = run_cli([
        "--db", str(db_path),
        "knn",
        "--vector", json.dumps([0.1, 0.1]),
        "--k", "2"
    ])
    assert r3.returncode == 0
    payload3 = json.loads(r3.stdout)
    assert payload3["ok"] is True
    assert len(payload3["result"]) == 2

    r4 = run_cli(["--db", str(db_path), "stats"])
    assert r4.returncode == 0
    payload4 = json.loads(r4.stdout)
    assert payload4["ok"] is True
    assert payload4["result"]["vector_count"] == 3