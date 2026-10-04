"""
Compare our GPU drivers (dumped by md_dump.cu) against the OpenMM oracle.

Expected files in DATA_DIR (written by md_dump.cu):
    dump_state0.csv        i, r, v, F_fp64, F_mixed         (t = 0)
    dump_traj_fp64.csv     step, KE, PE, E                  (our FP64 driver)
    dump_traj_mixed.csv    step, KE, PE, E                  (our mixed driver)
    dump_final_fp64.csv    i, r, v                          (positions at the end)
    dump_final_mixed.csv   i, r, v

Usage:
    python compare_to_oracle.py DATA_DIR [STEPS CHECKPOINT]      (defaults 30000 1000)
    python compare_to_oracle.py --selftest                      (pipeline test only)

The tolerances below were fixed before looking at any result.
"""
import os
import sys
import tempfile
import numpy as np

import oracle_openmm as orc

# tolerances, fixed before the first real comparison
TOL_F_FP64 = 1e-12      # max |dF|, our FP64 kernel vs oracle
TOL_PE_REL = 1e-12      # |PE_ours - PE_oracle| / |PE|
TOL_F_MIXED = 2e-5      # max |dF|, our mixed kernel vs oracle (bench saw 7e-6 at N=65536)
TOL_E_TRAJ = 1e-9       # max over checkpoints of |E_ours - E_oracle| / |E0|, FP64 driver


def load(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def status(ok):
    return "PASS" if ok else "FAIL"


def compare(data_dir, steps, cp):
    s0 = load(os.path.join(data_dir, "dump_state0.csv"))
    r, v = s0["r"], s0["v"]
    n = len(r)
    print(f"N = {n}, steps = {steps}, checkpoint = {cp}\n")

    # ---------- test 1: static forces ----------
    F_or, PE_or = orc.forces_pe(r)
    dF64 = np.abs(s0["F_fp64"] - F_or)
    dFmx = np.abs(s0["F_mixed"] - F_or)
    rms = np.sqrt(np.mean(F_or ** 2))
    print("TEST 1  forces at t = 0 vs OpenMM (double)")
    print(f"  rms |F|                        = {rms:.6e}")
    print(f"  FP64  kernel: max |dF| = {dF64.max():.3e}   rms |dF| = {np.sqrt(np.mean(dF64**2)):.3e}"
          f"   [{status(dF64.max() <= TOL_F_FP64)}, tol {TOL_F_FP64:.0e}]")
    print(f"  MIXED kernel: max |dF| = {dFmx.max():.3e}   rms |dF| = {np.sqrt(np.mean(dFmx**2)):.3e}"
          f"   [{status(dFmx.max() <= TOL_F_MIXED)}, tol {TOL_F_MIXED:.0e}]\n")

    # ---------- test 2: potential and kinetic energy at t = 0 ----------
    t64 = load(os.path.join(data_dir, "dump_traj_fp64.csv"))
    tmx = load(os.path.join(data_dir, "dump_traj_mixed.csv"))
    ke_or = 0.5 * orc.MASS * np.sum(v ** 2)
    print("TEST 2  energies at t = 0")
    for name, t in (("FP64 ", t64), ("MIXED", tmx)):
        dpe = abs(t["PE"][0] - PE_or) / abs(PE_or)
        dke = abs(t["KE"][0] - ke_or) / abs(ke_or)
        print(f"  {name}: PE {t['PE'][0]:.12e}  vs oracle {PE_or:.12e}  rel diff {dpe:.3e}"
              f"   [{status(dpe <= TOL_PE_REL)}]   KE rel diff {dke:.3e}")
    print()

    # ---------- test 3: trajectory energies vs oracle ----------
    print(f"TEST 3  energy trajectory vs OpenMM velocity Verlet ({steps} steps, dt = 1e-4)")
    hist, r_fin_or = orc.run_velocity_verlet(r, v, 1e-4, steps, cp)
    o_step = np.array([h[0] for h in hist])
    o_E = np.array([h[3] for h in hist])
    E0 = o_E[0]
    assert np.array_equal(o_step, t64["step"].astype(int)), "checkpoint grids differ"
    print(f"  {'step':>6} {'oracle |dE|/|E0|':>17} {'FP64 |dE|/|E0|':>15} {'MIXED |dE|/|E0|':>16}"
          f" {'|FP64-oracle|/|E0|':>19} {'|MIXED-oracle|/|E0|':>20}")
    worst64 = worst_mx = 0.0
    for k in range(len(o_step)):
        e_or = abs(o_E[k] - E0) / abs(E0)
        e64 = abs(t64["E"][k] - t64["E"][0]) / abs(t64["E"][0])
        emx = abs(tmx["E"][k] - tmx["E"][0]) / abs(tmx["E"][0])
        d64 = abs(t64["E"][k] - o_E[k]) / abs(E0)
        dmx = abs(tmx["E"][k] - o_E[k]) / abs(E0)
        worst64, worst_mx = max(worst64, d64), max(worst_mx, dmx)
        print(f"  {o_step[k]:6d} {e_or:17.3e} {e64:15.3e} {emx:16.3e} {d64:19.3e} {dmx:20.3e}")
    print(f"  max over run: |E_FP64  - E_oracle|/|E0| = {worst64:.3e}   "
          f"[{status(worst64 <= TOL_E_TRAJ)}, tol {TOL_E_TRAJ:.0e}]")
    print(f"  max over run: |E_MIXED - E_oracle|/|E0| = {worst_mx:.3e}   [informational]\n")

    # ---------- test 4: final positions ----------
    f64 = load(os.path.join(data_dir, "dump_final_fp64.csv"))
    fmx = load(os.path.join(data_dir, "dump_final_mixed.csv"))
    print("TEST 4  final positions vs OpenMM (informational: chaotic system, errors amplify)")
    print(f"  FP64 : max |dx| = {np.abs(f64['r'] - r_fin_or).max():.3e}   "
          f"rms |dx| = {np.sqrt(np.mean((f64['r'] - r_fin_or) ** 2)):.3e}")
    print(f"  MIXED: max |dx| = {np.abs(fmx['r'] - r_fin_or).max():.3e}   "
          f"rms |dx| = {np.sqrt(np.mean((fmx['r'] - r_fin_or) ** 2)):.3e}")


def make_synthetic(data_dir, n, steps, cp):
    """Pipeline self-test ONLY: fabricates dump files from the oracle itself,
    so every difference should be ~0. This validates the script, not our code."""
    os.makedirs(data_dir, exist_ok=True)
    r = orc.lattice_jitter(n)
    v = np.where(np.arange(n) % 2 == 0, 0.01, -0.01)
    F, _ = orc.forces_pe(r)
    hdr = "i,r,v,F_fp64,F_mixed"
    np.savetxt(os.path.join(data_dir, "dump_state0.csv"),
               np.column_stack([np.arange(n), r, v, F, F.astype(np.float32).astype(np.float64)]),
               delimiter=",", header=hdr, comments="", fmt="%.17g")
    hist, rf = orc.run_velocity_verlet(r, v, 1e-4, steps, cp)
    for tag in ("fp64", "mixed"):
        np.savetxt(os.path.join(data_dir, f"dump_traj_{tag}.csv"), np.array(hist),
                   delimiter=",", header="step,KE,PE,E", comments="", fmt="%.17g")
        np.savetxt(os.path.join(data_dir, f"dump_final_{tag}.csv"),
                   np.column_stack([np.arange(n), rf, np.zeros(n)]),
                   delimiter=",", header="i,r,v", comments="", fmt="%.17g")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        with tempfile.TemporaryDirectory() as d:
            make_synthetic(d, 1024, 1000, 500)
            print("PIPELINE SELF-TEST (synthetic data built from the oracle; expect ~0 diffs)\n")
            compare(d, 1000, 500)
    else:
        d = sys.argv[1]
        st = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
        cp = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
        compare(d, st, cp)
