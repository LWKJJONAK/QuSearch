"""Compare regenerated CSV files with the reference data.

    python python/compare.py [output/data] [data]

Text and integers must agree exactly. Floating-point entries may differ by 1e-9 (absolute or
relative), which covers rounding differences between platforms. Exits with status 1 on mismatch.
"""
import csv
import math
import pathlib
import sys

new = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "output/data")
ref = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "data")
TOL = 1e-9


def same(a, b):
    if a == b:
        return True
    try:
        x, y = float(a), float(b)
    except ValueError:
        return False
    return "." in a + b and math.isclose(x, y, rel_tol=TOL, abs_tol=TOL)


failed = False
for path in sorted(ref.glob("*.csv")):
    with open(path, newline="") as f, open(new / path.name, newline="") as g:
        old, cur = list(csv.reader(f)), list(csv.reader(g))
    ok = len(old) == len(cur) and all(len(r) == len(s) and all(map(same, r, s)) for r, s in zip(old, cur))
    print(f"{'ok      ' if ok else 'MISMATCH'}  {path.name}  ({len(old) - 1} rows)")
    failed |= not ok
sys.exit(int(failed))
