"""
Phase-Topology Laboratory
=========================
Integrated extension of the unified operator:

  ψ_{t+1} = P_sat [ L_κ + B_θ + B_z + N_GL + D_ext + N_noise ] ψ_t

with three-tier curvature κ_t = κ_0 + α_fb⟨|ψ|²⟩ + β_outer|Ψ_t|² + Σ A_j cos(ω_j t)

Computes simultaneously:
  R(t)    — Kuramoto order parameter (phase coherence)
  S_ω     — spectral entropy over Fourier modes
  ξ       — correlation length from C(r) = ⟨e^i(φ(x)-φ(x+r))⟩
  Q       — topological charge (total winding number)
  λ_max   — Lyapunov exponent from twin-trajectory divergence

Phase diagram swept over (A_drive, σ_noise). Defect braiding tracked in
the strong-coupling regime where vortices nucleate and migrate.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.fft import fft, fftfreq

# ============================================================
# Lattice parameters
# ============================================================
N_p = 13
N_z = 50
seam_phase_base = np.pi * 0.75

# Standard coupling regime
J_ang_std = 0.22
J_ax_std  = 0.28
alpha_fb  = 0.18
beta_outer = 0.25
kappa_0 = 0.08
decay = 0.05
g_cubic = 0.10
tau_inner = 6
sat_max = 2.0

# Global Bloom sentinel
def global_sentinel(N_p, N_z, sigma_n=4.0, sigma_z=14.0):
    n_arr = np.arange(N_p) - (N_p-1)/2
    m_arr = np.arange(N_z) - (N_z-1)/2
    rho = np.exp(-(n_arr[:,None]**2)/(2*sigma_n**2)) * \
          np.exp(-(m_arr[None,:]**2)/(2*sigma_z**2))
    return rho / rho.sum()
rho_global = global_sentinel(N_p, N_z)

# ============================================================
# Deterministic chaos generator (replaces PRNG for noise too)
# ============================================================
class ChaosSource:
    """Logistic-map chaos as the noise generator — honest within the ontology."""
    def __init__(self, seed=0.4123, r=3.978):
        self.s = seed; self.r = r
    def step(self):
        self.s = self.r * self.s * (1 - self.s)
        return self.s
    def gaussian_like(self, shape):
        out = np.zeros(np.prod(shape))
        for i in range(out.size):
            out[i] = self.step() - 0.5
        return out.reshape(shape) * np.sqrt(12)  # var → 1

# ============================================================
# Core driven evolution
# ============================================================
def evolve(psi, history, t, kappa_prev, omega_d, A_drive, sigma_noise,
           seam_phase, J_ang, J_ax, chaos_R, chaos_I):
    """Single time step of the unified operator."""
    look = min(tau_inner, len(history))
    if look > 1:
        tw = np.exp(-np.arange(look)**2 / (2*(tau_inner/2)**2))
        tw /= tw.sum()
        tp = np.exp(-1j * kappa_prev * np.arange(look)**2 / 2)
        B_inner = sum(history[-(k+1)] * tw[k] * tp[k] for k in range(look))
    else:
        B_inner = psi.copy()

    # Angular Bloom with seam
    pL = np.roll(B_inner,  1, axis=0)
    pR = np.roll(B_inner, -1, axis=0)
    pL[0, :]  *= np.exp(-1j * seam_phase)
    pR[-1, :] *= np.exp(+1j * seam_phase)
    B_ang = 0.5 * (pL + pR)

    # Axial Bloom
    B_ax = 0.5 * (np.roll(B_inner, 1, axis=1) + np.roll(B_inner, -1, axis=1))

    # Outer-outer Bloom
    Psi_t = np.sum(rho_global * psi)

    # Three-tier + drive curvature
    intensity = np.mean(np.abs(psi)**2)
    kappa_t = (kappa_0 + alpha_fb * intensity
               + beta_outer * np.abs(Psi_t)**2
               + A_drive * np.cos(omega_d * t))

    # Update with stochastic injection
    psi_new = (psi * np.exp(-1j * kappa_t) * (1.0 - decay)
               + J_ang * B_ang + J_ax * B_ax
               - g_cubic * np.abs(psi)**2 * psi)
    if sigma_noise > 0:
        eta_R = chaos_R.gaussian_like(psi.shape)
        eta_I = chaos_I.gaussian_like(psi.shape)
        # Cap noise injection to prevent runaway
        noise = sigma_noise * (eta_R + 1j * eta_I)
        noise_mag = np.abs(noise)
        cap = 0.5
        noise_mag_capped = np.where(noise_mag > cap, cap, noise_mag)
        noise = noise * (noise_mag_capped / (noise_mag + 1e-12))
        psi_new += noise

    # Saturation
    a = np.abs(psi_new)
    m = a > sat_max
    psi_new[m] = psi_new[m] * sat_max / a[m]

    return psi_new, kappa_t, Psi_t, intensity

def init_psi(seed=0.4123):
    psi = np.zeros((N_p, N_z), dtype=complex)
    s = seed
    for n in range(N_p):
        for m in range(N_z):
            s = 3.97 * s * (1 - s)
            psi[n, m] = 0.10 * np.exp(2j*np.pi*s)
    return psi

# ============================================================
# Observables
# ============================================================
def kuramoto_R(psi):
    return float(np.abs(np.mean(np.exp(1j * np.angle(psi)))))

def spectral_entropy(psi):
    F = np.abs(np.fft.fft2(psi))**2
    P = F.flatten() / (F.sum() + 1e-12)
    P = P[P > 1e-12]
    H = -np.sum(P * np.log(P))
    return float(H / np.log(P.size))

def correlation_length(psi, max_r=40):
    phs = np.angle(psi)
    C = np.zeros(max_r)
    for r in range(1, max_r):
        shifted = np.roll(phs, r, axis=1)
        diff = phs - shifted
        C[r] = float(np.abs(np.mean(np.exp(1j * diff))))
    C[0] = 1.0
    below = np.where(C < np.exp(-1))[0]
    return float(below[0]) if len(below) else max_r

def topological_charge(psi, seam_phase):
    """
    Returns (Q_bulk, Q_seam, Q_field) where Q_bulk excludes the seam column.
    The seam contributes a fixed offset; bulk vortices are the dynamical part.
    """
    phs = np.angle(psi)
    Q_field = np.zeros((N_p, N_z-1))
    Q_bulk = 0.0; Q_seam = 0.0
    for n in range(N_p):
        for m in range(N_z - 1):
            n2 = (n + 1) % N_p
            is_seam = (n == N_p-1)
            sc = -seam_phase if is_seam else 0.0
            dphi = ((phs[n2,m]-phs[n,m]) + (phs[n2,m+1]-phs[n2,m])
                    + (phs[n,m+1]-phs[n2,m+1]) + (phs[n,m]-phs[n,m+1]) + sc)
            dphi = (dphi + np.pi) % (2*np.pi) - np.pi
            q = dphi / (2*np.pi)
            Q_field[n, m] = q
            if is_seam:
                Q_seam += q
            else:
                Q_bulk += q
    return float(Q_bulk), float(Q_seam), Q_field

# ============================================================
# (1) Phase diagram: (A_drive, σ_noise) sweep
# ============================================================
print("="*60)
print("Phase diagram sweep over (A_drive, σ_noise)")
print("="*60)
A_grid = np.linspace(0.0, 0.45, 6)
sigma_grid = np.linspace(0.0, 0.25, 6)
omega_d = 1.15  # peak from previous sweep

R_map  = np.zeros((len(A_grid), len(sigma_grid)))
S_map  = np.zeros_like(R_map)
xi_map = np.zeros_like(R_map)
Q_map  = np.zeros_like(R_map)

T_run = 400
T_drop = 250

for i, A in enumerate(A_grid):
    for j, sig in enumerate(sigma_grid):
        psi = init_psi()
        history = [psi.copy()]
        kappa_prev = kappa_0
        chR = ChaosSource(seed=0.31 + 0.07*i + 0.013*j)
        chI = ChaosSource(seed=0.59 + 0.07*i + 0.013*j)
        R_acc, S_acc, xi_acc, Q_acc, count = 0,0,0,0,0
        for t in range(T_run):
            psi, kappa_prev, _, _ = evolve(psi, history, t, kappa_prev,
                                           omega_d, A, sig, seam_phase_base,
                                           J_ang_std, J_ax_std, chR, chI)
            history.append(psi.copy())
            if len(history) > tau_inner + 2: history.pop(0)
            if t >= T_drop:
                R_acc  += kuramoto_R(psi)
                S_acc  += spectral_entropy(psi)
                xi_acc += correlation_length(psi)
                qb, qs, _ = topological_charge(psi, seam_phase_base)
                Q_acc  += abs(qb)
                count  += 1
        R_map[i,j]  = R_acc  / count
        S_map[i,j]  = S_acc  / count
        xi_map[i,j] = xi_acc / count
        Q_map[i,j]  = Q_acc  / count
        print(f"  A={A:.2f}  σ={sig:.2f}   R={R_map[i,j]:.3f}  S_ω={S_map[i,j]:.3f}  ξ={xi_map[i,j]:.1f}  |Q|={Q_map[i,j]:.2f}")

# ============================================================
# (2) Vortex regime — push coupling and seam to nucleate defects
# ============================================================
print("\n" + "="*60)
print("Vortex regime — defect nucleation and tracking")
print("="*60)

# Strong drive + sharper seam + reduced GL stiffness → defects nucleate
J_ang_v = 0.42
J_ax_v  = 0.50
seam_v  = np.pi * 0.98     # nearly maximal frustration
sigma_v = 0.18
A_v     = 0.42
g_v     = 0.04             # softer cubic confinement → vortices possible
T_vortex = 500

psi = init_psi(seed=0.6217)
history = [psi.copy()]
kappa_prev = kappa_0
chR = ChaosSource(seed=0.273)
chI = ChaosSource(seed=0.821)

Q_bulk_trace = []
Q_seam_trace = []
defect_positions = []

# Local override of g_cubic for vortex regime — patch evolve via global
g_cubic_saved = g_cubic
g_cubic = g_v

for t in range(T_vortex):
    psi, kappa_prev, _, _ = evolve(psi, history, t, kappa_prev,
                                    omega_d, A_v, sigma_v, seam_v,
                                    J_ang_v, J_ax_v, chR, chI)
    history.append(psi.copy())
    if len(history) > tau_inner + 2: history.pop(0)

    if t > 100 and t % 2 == 0:
        qb, qs, Q_field = topological_charge(psi, seam_v)
        Q_bulk_trace.append(qb)
        Q_seam_trace.append(qs)
        defects_here = []
        for n in range(N_p - 1):  # exclude seam column
            for m in range(N_z - 1):
                if abs(Q_field[n,m]) > 0.4:
                    defects_here.append((n, m, np.sign(Q_field[n,m])))
        defect_positions.append((t, defects_here))

g_cubic = g_cubic_saved

defect_count = [len(d[1]) for d in defect_positions]
print(f"Mean bulk defect count = {np.mean(defect_count):.2f}")
print(f"Max bulk defect count  = {max(defect_count) if defect_count else 0}")
print(f"Mean |Q_bulk| = {np.mean(np.abs(Q_bulk_trace)):.3f}")
print(f"Mean |Q_seam| = {np.mean(np.abs(Q_seam_trace)):.3f}  (seam baseline)")

# Track world lines: extract trajectories of +/- defects
times = np.array([d[0] for d in defect_positions])
plus_traj  = []   # list of (t, m) for positive defects per snapshot
minus_traj = []
for t_i, dlist in defect_positions:
    for (n, m, s) in dlist:
        if s > 0:
            plus_traj.append((t_i, n, m))
        else:
            minus_traj.append((t_i, n, m))
plus_traj  = np.array(plus_traj)  if plus_traj  else np.empty((0,3))
minus_traj = np.array(minus_traj) if minus_traj else np.empty((0,3))

# ============================================================
# (3) Lyapunov estimate — twin trajectories
# ============================================================
print("\n" + "="*60)
print("Lyapunov exponent estimate (twin trajectory divergence)")
print("="*60)

def run_twin_pair(A_drive, sigma_noise, J_ang, J_ax, seam, T=300,
                  perturb=1e-6):
    psi_a = init_psi(seed=0.4123)
    psi_b = psi_a.copy()
    # Small perturbation at one site
    psi_b[5, 25] += perturb * (1.0 + 1j) / np.sqrt(2)

    hist_a = [psi_a.copy()]; hist_b = [psi_b.copy()]
    kappa_a = kappa_b = kappa_0
    chR_a = ChaosSource(seed=0.111); chI_a = ChaosSource(seed=0.222)
    chR_b = ChaosSource(seed=0.111); chI_b = ChaosSource(seed=0.222)

    divergence = []
    for t in range(T):
        psi_a, kappa_a, _, _ = evolve(psi_a, hist_a, t, kappa_a, omega_d,
                                      A_drive, sigma_noise, seam,
                                      J_ang, J_ax, chR_a, chI_a)
        psi_b, kappa_b, _, _ = evolve(psi_b, hist_b, t, kappa_b, omega_d,
                                      A_drive, sigma_noise, seam,
                                      J_ang, J_ax, chR_b, chI_b)
        hist_a.append(psi_a.copy()); hist_b.append(psi_b.copy())
        if len(hist_a) > tau_inner+2: hist_a.pop(0); hist_b.pop(0)
        d = np.linalg.norm(psi_a - psi_b)
        divergence.append(d)
    return np.array(divergence)

# Three regimes for Lyapunov
lyap_regimes = {
    'ordered':    (0.10, 0.02, J_ang_std, J_ax_std, seam_phase_base),
    'critical':   (0.25, 0.10, J_ang_std, J_ax_std, seam_phase_base),
    'turbulent':  (A_v, sigma_v, J_ang_v, J_ax_v, seam_v),
}
lyap_results = {}
for name, (A, sig, Ja, Jz, sm) in lyap_regimes.items():
    div = run_twin_pair(A, sig, Ja, Jz, sm, T=250)
    # Fit log-linear over growth window
    valid = (div > 1e-10) & (div < 1.0)
    if valid.sum() > 20:
        ts = np.where(valid)[0]
        lam = np.polyfit(ts, np.log(div[valid]), 1)[0]
    else:
        lam = float('nan')
    lyap_results[name] = (div, lam)
    print(f"  {name:10s}   λ_max ≈ {lam:+.4f}")

# ============================================================
# Visualization
# ============================================================
fig = plt.figure(figsize=(14, 13))
fig.patch.set_facecolor('#0a0a0f')
gs = fig.add_gridspec(4, 4, hspace=0.5, wspace=0.4)

def style(ax):
    ax.set_facecolor('#0a0a0f')
    for s in ax.spines.values(): s.set_color('#6a6a8a')
    ax.tick_params(colors='#b8b8d0')
    ax.xaxis.label.set_color('#d8d8ee')
    ax.yaxis.label.set_color('#d8d8ee')
    ax.title.set_color('#e8e8ff')
    ax.grid(True, alpha=0.15, color='#5a5a7a')

# Row 1: phase diagram across (A, σ) for R, S_ω, ξ, |Q|
ext = [sigma_grid.min(), sigma_grid.max(), A_grid.min(), A_grid.max()]
maps = [(R_map, 'Kuramoto R', 'magma'),
        (S_map, 'spectral entropy S_ω', 'viridis'),
        (xi_map, 'correlation length ξ', 'plasma'),
        (Q_map, '|Q| (topological)', 'inferno')]
for j, (M, lbl, cm_name) in enumerate(maps):
    ax = fig.add_subplot(gs[0, j])
    im = ax.imshow(M, aspect='auto', origin='lower', extent=ext,
                   cmap=cm_name, interpolation='bilinear')
    ax.set_xlabel('σ_noise')
    ax.set_ylabel('A_drive')
    ax.set_title(lbl, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.055)
    style(ax)

# Row 2: vortex regime amplitude/phase + defect snapshot
ax = fig.add_subplot(gs[1, :2])
im = ax.imshow(np.abs(psi).T, aspect='auto', cmap='magma', origin='lower',
                extent=[0, N_p, 0, N_z], interpolation='bilinear')
ax.axvline(0,   color='#5affc0', lw=1.0, ls='--')
ax.axvline(N_p, color='#5affc0', lw=1.0, ls='--', label='seam')
# Overlay defects from last snapshot
if defect_positions:
    last_defs = defect_positions[-1][1]
    for (n, m, s) in last_defs:
        c = '#5affc0' if s > 0 else '#ff5aa0'
        ax.scatter([n+0.5], [m+0.5], s=80, c=c, marker='o' if s>0 else 's',
                   edgecolor='white', linewidth=0.8)
ax.set_xlabel('protofilament n')
ax.set_ylabel('axial m')
ax.set_title(f'Vortex regime |B| — defects at final time (circles=+, squares=−)',
             fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.046)
style(ax)

# Defect count over time
ax = fig.add_subplot(gs[1, 2:])
ax.plot([d[0] for d in defect_positions], defect_count,
        color='#ffd35a', lw=1.2, label='defect count')
ax2 = ax.twinx()
ax2.plot([d[0] for d in defect_positions], Q_bulk_trace,
         color='#5ad8ff', lw=0.9, alpha=0.85, label='Q_bulk')
ax.set_xlabel('iteration')
ax.set_ylabel('defect count', color='#ffd35a')
ax2.set_ylabel('Q total', color='#5ad8ff')
ax.tick_params(axis='y', colors='#ffd35a')
ax2.tick_params(axis='y', colors='#5ad8ff')
ax.set_title('Defect nucleation and topological charge transport', fontsize=10)
lines1,labels1 = ax.get_legend_handles_labels()
lines2,labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2,
          facecolor='#15151f', edgecolor='#5a5a7a', labelcolor='#d8d8ee',
          fontsize=8, loc='upper right')
style(ax)
for s in ax2.spines.values(): s.set_color('#6a6a8a')

# Row 3: defect world lines (time × n) — the braid projection
ax = fig.add_subplot(gs[2, :2])
if len(plus_traj):
    ax.scatter(plus_traj[:,0], plus_traj[:,1], c='#5affc0', s=8,
               alpha=0.55, label=f'+ defects ({len(plus_traj)})')
if len(minus_traj):
    ax.scatter(minus_traj[:,0], minus_traj[:,1], c='#ff5aa0', s=8,
               alpha=0.55, marker='s', label=f'− defects ({len(minus_traj)})')
ax.set_xlabel('iteration t')
ax.set_ylabel('protofilament n')
ax.set_title('Defect world lines — angular projection (braid diagram)', fontsize=10)
ax.legend(facecolor='#15151f', edgecolor='#5a5a7a', labelcolor='#d8d8ee',
          fontsize=8)
style(ax)

ax = fig.add_subplot(gs[2, 2:])
if len(plus_traj):
    ax.scatter(plus_traj[:,0], plus_traj[:,2], c='#5affc0', s=8,
               alpha=0.55, label='+')
if len(minus_traj):
    ax.scatter(minus_traj[:,0], minus_traj[:,2], c='#ff5aa0', s=8,
               alpha=0.55, marker='s', label='−')
ax.set_xlabel('iteration t')
ax.set_ylabel('axial m')
ax.set_title('Defect world lines — axial projection', fontsize=10)
ax.legend(facecolor='#15151f', edgecolor='#5a5a7a', labelcolor='#d8d8ee',
          fontsize=8)
style(ax)

# Row 4: Lyapunov regimes
ax = fig.add_subplot(gs[3, :])
colors = {'ordered': '#5ad8ff', 'critical': '#ffd35a', 'turbulent': '#ff5aa0'}
for name, (div, lam) in lyap_results.items():
    valid = div > 1e-12
    ax.semilogy(np.where(valid)[0], div[valid], color=colors[name],
                lw=1.0, alpha=0.9, label=f'{name}: λ ≈ {lam:+.4f}')
ax.set_xlabel('iteration')
ax.set_ylabel('||ψ_a − ψ_b||')
ax.set_title('Lyapunov twin-trajectory divergence — chaos-to-locking transition',
             fontsize=10)
ax.legend(facecolor='#15151f', edgecolor='#5a5a7a', labelcolor='#d8d8ee')
style(ax)

fig.suptitle('Phase-topology laboratory — full operator-family characterization',
             color='#e8e8ff', fontsize=13, y=0.995)
plt.savefig('/mnt/user-data/outputs/phase_topology_lab.png', dpi=140,
            facecolor='#0a0a0f', bbox_inches='tight')

print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"Phase diagram swept   (A × σ) = {len(A_grid)}×{len(sigma_grid)}")
print(f"Vortex regime defects: mean={np.mean(defect_count):.2f}  max={max(defect_count)}")
print(f"Lyapunov spectrum:")
for name, (_, lam) in lyap_results.items():
    print(f"  {name:10s} λ_max ≈ {lam:+.4f}    "
          f"({'ordered' if lam<-0.01 else 'critical' if abs(lam)<0.01 else 'chaotic'})")
print("\nFigure saved.")
