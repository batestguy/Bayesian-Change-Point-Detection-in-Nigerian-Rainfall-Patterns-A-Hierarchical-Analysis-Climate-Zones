"""
Orchestrator: runs all 4 MCMC steps on Colab sequentially.
Uses colab exec -f for each script with long timeouts.
Sends keepalive pings between steps to prevent session expiry.
"""
import subprocess
import sys
import time

SESSION = None  # will be auto-detected

def colab_cmd(args, timeout=60):
    cmd = (
        f'export HOME=/home/batesthommie; '
        f'export PATH=/home/batesthommie/.local/bin:/usr/local/bin:/usr/bin:/bin; '
        f'colab {args}'
    )
    result = subprocess.run(
        ["wsl", "--exec", "bash", "-c", cmd],
        capture_output=True, text=True, timeout=timeout
    )
    return result.stdout + result.stderr, result.returncode

def colab_exec(script_file, timeout_sec=2400):
    """Execute a script on Colab with long timeout."""
    out, rc = colab_cmd(f'exec -f {script_file} --timeout {timeout_sec}', timeout=timeout_sec + 60)
    return out, rc

def keepalive():
    """Send a simple computation to keep the kernel alive."""
    # Create a tiny keepalive script
    with open("colab_keepalive.py", "w") as f:
        f.write('import time; print(f"Keepalive at {time.strftime(\\"%H:%M:%S\\")}")\n')
    out, rc = colab_cmd('exec -f colab_keepalive.py --timeout 30', timeout=60)
    print(f"  Keepalive: {out.strip()}")
    return rc == 0

def main():
    # Check session
    print("Checking Colab session...")
    out, rc = colab_cmd('ls')
    print(out)
    if rc != 0:
        print("ERROR: No active Colab session. Create one first:")
        print("  colab create")
        sys.exit(1)

    scripts = [
        ("colab_shared.py", "Shared preamble (imports + data)", 120),
        ("colab_mcmc_01_single_cp.py", "Step 1: Single CP models (3 zones)", 2400),
        ("colab_mcmc_02_hierarchical.py", "Step 2: Hierarchical model", 1800),
        ("colab_mcmc_03_comparison.py", "Step 3: Model comparison (9 models)", 2400),
        ("colab_mcmc_04_figures.py", "Step 4: Figures + zip", 300),
    ]

    for i, (script, desc, timeout) in enumerate(scripts):
        print(f"\n{'='*60}")
        print(f"Running: {desc}")
        print(f"Script: {script} (timeout: {timeout}s)")
        print(f"{'='*60}\n")

        t0 = time.time()
        out, rc = colab_exec(script, timeout_sec=timeout)
        elapsed = time.time() - t0

        # Print last 50 lines of output
        lines = out.strip().split('\n')
        if len(lines) > 50:
            print(f"  ... ({len(lines) - 50} lines omitted) ...")
            for line in lines[-50:]:
                print(line)
        else:
            print(out)

        if rc != 0:
            print(f"\nERROR: {script} failed (rc={rc}) after {elapsed:.0f}s")
            print("Session may have expired. Check with: colab ls")
            sys.exit(1)

        print(f"\nCompleted {script} in {elapsed:.0f}s")

        # Keepalive between steps
        if i < len(scripts) - 1:
            print("  Sending keepalive...")
            keepalive()
            time.sleep(2)

    print("\n" + "="*60)
    print("ALL STEPS COMPLETE!")
    print("="*60)
    print("\nDownload results with:")
    print("  colab download /content/mcmc_output.zip")

if __name__ == "__main__":
    main()
