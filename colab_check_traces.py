import os
from pathlib import Path
traces = Path("/content/output/traces")
figs = Path("/content/output/figures")
if traces.exists():
    files = sorted(traces.glob("*.nc"))
    print(f"Traces ({len(files)}):")
    for f in files:
        print(f"  {f.name}  ({f.stat().st_size / 1024 / 1024:.1f} MB)")
else:
    print("No traces directory")
if figs.exists():
    files = sorted(figs.glob("*"))
    print(f"\nFigures ({len(files)}):")
    for f in files:
        print(f"  {f.name}")
else:
    print("No figures directory")
# Check if hier_trace exists in memory
try:
    print(f"\nhier_trace in memory: {type(hier_trace)}")
    print(f"hier_ppc in memory: {type(hier_ppc)}")
except NameError:
    print("\nhier_trace NOT in memory (sampling may still be running or failed)")
