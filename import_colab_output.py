"""Import Colab MCMC output zip back into the project structure."""

import shutil
import zipfile
import sys
from pathlib import Path

PROJECT = Path(__file__).parent
ZIP_NAME = "mcmc_output.zip"

# Check common download locations
candidates = [
    PROJECT / ZIP_NAME,
    Path.home() / "Downloads" / ZIP_NAME,
    Path.home() / "Desktop" / ZIP_NAME,
]

zip_path = None
for c in candidates:
    if c.exists():
        zip_path = c
        break

if zip_path is None:
    if len(sys.argv) > 1:
        zip_path = Path(sys.argv[1])
    else:
        print(f"Could not find {ZIP_NAME} in:")
        for c in candidates:
            print(f"  {c}")
        print(f"\nUsage: python import_colab_output.py <path_to_zip>")
        sys.exit(1)

print(f"Importing from: {zip_path}")

with zipfile.ZipFile(zip_path, "r") as zf:
    for member in zf.namelist():
        if member.startswith("traces/") and member.endswith(".nc"):
            dest = PROJECT / "data" / member
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            print(f"  -> data/{member}")

        elif member.startswith("figures/"):
            dest = PROJECT / member
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            print(f"  -> {member}")

        elif member.startswith("data/") and member.endswith(".tex"):
            dest = PROJECT / member
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            print(f"  -> {member}")

print("\nDone! Traces, figures, and comparison tables imported.")
print("You can now compile the paper: cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main")
