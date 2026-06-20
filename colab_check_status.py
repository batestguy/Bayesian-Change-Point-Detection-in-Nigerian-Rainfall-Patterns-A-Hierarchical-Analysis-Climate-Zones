import os
dirs = os.listdir("/content/")
print("Contents of /content/:", dirs)
if "output" in dirs:
    out = os.listdir("/content/output/")
    print("Contents of /content/output/:", out)
    for f in out:
        fp = os.path.join("/content/output", f)
        sz = os.path.getsize(fp)
        print(f"  {f}: {sz/1024:.1f} KB")
if os.path.exists("/content/mcmc_output.zip"):
    sz = os.path.getsize("/content/mcmc_output.zip")
    print(f"mcmc_output.zip: {sz/1024:.1f} KB")
