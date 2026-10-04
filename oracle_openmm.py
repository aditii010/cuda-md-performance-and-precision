"""
External oracle for the CUDA MD project: OpenMM (Reference platform, double
precision) evaluating the SAME model our kernels implement:

    LJ, sigma = epsilon = mass = 1, cutoff rc = 2.5, shifted-force:
        U_sf(r) = U(r) - U(rc) + F(rc) * (r - rc),   r < rc,  else 0
        F_sf(r) = F(r) - F(rc)

The 1D chain is run as a collinear 3D system (y = z = 0). By symmetry all
y/z forces are exactly zero, so this is exactly the 1D problem.

Units: OpenMM's (nm, ps, kJ/mol, amu) are mutually consistent
(1 amu = 1 kJ/mol ps^2/nm^2), so with mass = 1 the numbers are the reduced
LJ numbers unchanged.

Usage as a module: forces_pe(), run_velocity_verlet(), numpy_reference().
Run as a script: self-test (OpenMM vs an independent numpy brute force).
"""
import sys
import numpy as np
import openmm as mm
import openmm.unit as u

SIGMA = 1.0
EPS = 1.0
MASS = 1.0
RC = 2.5


def lj_u(r):
    sr6 = (SIGMA / r) ** 6
    return 4.0 * EPS * (sr6 * sr6 - sr6)


def lj_f(r):
    sr6 = (SIGMA / r) ** 6
    return 24.0 * EPS / r * (2.0 * sr6 * sr6 - sr6)


UC = lj_u(RC)
FC = lj_f(RC)


# ---------------------------------------------------------------------------
# Independent reference (numpy, no OpenMM): O(N^2) brute force for small N.
# ---------------------------------------------------------------------------
def numpy_reference(x):
    """Return (F, PE) for a 1D chain x (shape (N,)) with the shifted-force LJ."""
    n = len(x)
    F = np.zeros(n)
    PE = 0.0
    # process in row blocks to keep memory modest
    block = 512
    for s in range(0, n, block):
        e = min(n, s + block)
        dx = x[s:e, None] - x[None, :]          # (b, n), dx = x_i - x_j
        r = np.abs(dx)
        idx = np.arange(s, e)[:, None] == np.arange(n)[None, :]
        mask = (r < RC) & (~idx)
        rs = np.where(mask, r, 1.0)
        f = np.where(mask, lj_f(rs) - FC, 0.0)
        F[s:e] = np.sum(f * np.sign(dx), axis=1)
        u_pair = np.where(mask, lj_u(rs) - UC + FC * (rs - RC), 0.0)
        PE += 0.5 * np.sum(u_pair)               # each pair seen twice
    return F, PE


# ---------------------------------------------------------------------------
# OpenMM oracle
# ---------------------------------------------------------------------------
def build_context(x, v, dt, platform="Reference"):
    n = len(x)
    system = mm.System()
    for _ in range(n):
        system.addParticle(MASS)

    force = mm.CustomNonbondedForce(
        "4*(1/r^12-1/r^6) - uc + fc*(r-rc)"
    )
    force.addGlobalParameter("uc", UC)
    force.addGlobalParameter("fc", FC)
    force.addGlobalParameter("rc", RC)
    for _ in range(n):
        force.addParticle([])
    force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
    force.setCutoffDistance(RC)
    force.setUseLongRangeCorrection(False)
    system.addForce(force)

    # velocity Verlet (kick-drift-kick), NOT OpenMM's leap-frog, so the
    # integrator matches ours step for step.
    integ = mm.CustomIntegrator(dt)
    integ.addComputePerDof("v", "v + 0.5*dt*f/m")
    integ.addComputePerDof("x", "x + dt*v")
    integ.addComputePerDof("v", "v + 0.5*dt*f/m")

    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName(platform))
    pos = np.zeros((n, 3))
    pos[:, 0] = x
    vel = np.zeros((n, 3))
    vel[:, 0] = v
    ctx.setPositions(pos)
    ctx.setVelocities(vel)
    return ctx, integ


def forces_pe(x, platform="Reference"):
    """OpenMM forces along x and potential energy for positions x."""
    ctx, _ = build_context(x, np.zeros(len(x)), 1e-4, platform)
    st = ctx.getState(getForces=True, getEnergy=True)
    f = st.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer)
    pe = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    assert np.max(np.abs(f[:, 1:])) < 1e-9, "y/z force nonzero: not collinear"
    return f[:, 0], pe


def run_velocity_verlet(x, v, dt, steps, checkpoint, platform="Reference"):
    """Run and return a list of (step, KE, PE, E) at each checkpoint (+ step 0)."""
    ctx, integ = build_context(x, v, dt, platform)

    def snap(step):
        st = ctx.getState(getEnergy=True)
        ke = st.getKineticEnergy().value_in_unit(u.kilojoule_per_mole)
        pe = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        return (step, ke, pe, ke + pe)

    out = [snap(0)]
    done = 0
    while done < steps:
        k = min(checkpoint, steps - done)
        integ.step(k)
        done += k
        out.append(snap(done))
    final = ctx.getState(getPositions=True).getPositions(asNumpy=True)
    final = final.value_in_unit(u.nanometer)[:, 0]
    return out, final


def lattice_jitter(n, seed=42, spacing=1.5, amp=0.2):
    rng = np.random.default_rng(seed)
    return np.arange(n) * spacing + rng.uniform(-amp, amp, n)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    x = lattice_jitter(n)
    v = np.where(np.arange(n) % 2 == 0, 0.01, -0.01)

    print(f"OpenMM {mm.__version__}, platform Reference (double), N = {n}")
    print(f"model: shifted-force LJ, rc = {RC}, U(rc) = {UC:.12e}, F(rc) = {FC:.12e}\n")

    # --- self-test 1: static forces and PE, OpenMM vs independent numpy ---
    F_np, PE_np = numpy_reference(x)
    F_mm, PE_mm = forces_pe(x)
    dF = np.abs(F_np - F_mm)
    print("SELF-TEST 1: static forces / PE, OpenMM vs numpy brute force")
    print(f"  rms |F|            = {np.sqrt(np.mean(F_np**2)):.6e}")
    print(f"  max |dF|           = {dF.max():.3e}")
    print(f"  PE numpy           = {PE_np:.12e}")
    print(f"  PE OpenMM          = {PE_mm:.12e}")
    print(f"  |dPE| / |PE|       = {abs(PE_np - PE_mm) / abs(PE_np):.3e}")
    print(f"  sum of forces      = {F_mm.sum():.3e}  (Newton 3rd law, should be ~0)\n")

    # --- self-test 2: short trajectory, energy behaviour ---
    steps, cp = 2000, 500
    hist, _ = run_velocity_verlet(x, v, 1e-4, steps, cp)
    e0 = hist[0][3]
    print(f"SELF-TEST 2: velocity Verlet, dt = 1e-4, {steps} steps")
    print(f"  {'step':>6} {'KE':>16} {'PE':>18} {'E-E0':>14} {'|E-E0|/|E0|':>14}")
    for s, ke, pe, e in hist:
        print(f"  {s:6d} {ke:16.9e} {pe:18.9e} {e - e0:14.6e} {abs(e - e0) / abs(e0):14.3e}")
