"""
Bloom Bridge — Nihiline Recursion grounded in the Phase-Curvature framework
============================================================================
Replaces abstract ψ(t) with closed-form Fresnel Bloom integral
    B̃(t) = e^(−i n₀(t)) · (1 + i κ(t) σ²)^(−1/2)
from Thorne & K1ll, Phase-Curvature Biosentinels, April 2026, Eq. (9).

Computes both detector channels:
    amplitude   |B̃|² = (1 + κ²σ⁴)^(−1/2)         ~ quadratic in κ
    phase       arg B̃ = −n₀ − ½ arctan(κσ²)        ~ linear in κ

Drives κ(t) with deterministic chaotic source (logistic; replace with
p-model cascade for full EFL multifractal lift). Maps cellular states
Q / M / A onto κ regimes per Section 4 of the preprint. Computes the
Δt apoptosis detection window from differential channel response.

Physical units: TFLN broadband THz probe at λ = 100 µm carrier,
sentinel waist σ = 2 µm (matching condition κσ² ~ 1 for κ ~ 0.3 µm⁻²
in mammalian cells with Δn ~ 0.05 between organelles and cytoplasm).
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Physical parameters from preprint Section 8
# ============================================================
lambda_THz  = 100e-6        # 100 µm carrier wavelength (THz)
sigma_um    = 2.0           # sentinel waist (matching condition)
sigma       = sigma_um      # work in µm
kappa_match = 1.0 / sigma**2  # matching curvature scale (µm⁻²) ≈ 0.25

# Time: model cellular passage / biological cycle in seconds
T_total = 1800.0   # 30 minutes — apoptosis detection window
dt      = 0.05     # 50 ms resolution
t       = np.arange(0, T_total, dt)

# ============================================================
# Cellular state κ(t) trajectories — preprint Section 4
# ============================================================
def kappa_quiescent(t, kappa_match, seed=0.4123):
    """State Q: |εσ²| << 1, smooth slow drift near zero curvature."""
    # Slow drift with small chaotic modulation
    base = 0.05 * kappa_match * np.sin(2*np.pi * t / 800.0)
    chaos = np.zeros_like(t)
    s = seed
    for i in range(len(t)):
        s = 3.78 * s * (1 - s)   # mild chaotic regime
        chaos[i] = (s - 0.5) * 0.08 * kappa_match
    return base + chaos

def kappa_mitotic(t, kappa_match, y0=600.0, w=120.0):
    """State M: localized Gaussian curvature spike, κσ² ~ 1 at y0."""
    return kappa_match * np.exp(-((t - y0)**2) / (2 * w**2))

def kappa_apoptotic(t, kappa_match, t_onset=400.0, seed=0.71):
    """
    State A: baseline near zero plus zero-mean stochastic ξ(t) with
    sign fluctuations. We use deterministic chaos with rising
    variance after t_onset to model the apoptosis transition.
    """
    base = 0.02 * kappa_match * np.sin(2*np.pi * t / 1000.0)
    xi = np.zeros_like(t)
    s = seed
    for i in range(len(t)):
        s = 3.95 * s * (1 - s)   # strong chaotic regime
        amp_envelope = 0.05 + 0.55 * (1.0 / (1.0 + np.exp(-(t[i]-t_onset)/80.0)))
        xi[i] = (s - 0.5) * amp_envelope * kappa_match * 2.0
    return base + xi

# ============================================================
# Bloom integral — closed-form Fresnel detector
# ============================================================
def bloom_integral(kappa_t, sigma, n0_t=None):
    """
    B̃(t) = e^(−i n₀(t)) · (1 + i κ(t) σ²)^(−1/2)
    Returns complex array.
    """
    if n0_t is None:
        n0_t = np.zeros_like(kappa_t)
    z = 1.0 + 1j * kappa_t * sigma**2
    return np.exp(-1j * n0_t) * z**(-0.5)

# ============================================================
# Channel observables
# ============================================================
def amplitude_intensity(B):
    return np.abs(B)**2

def phase_arg(B):
    return np.unwrap(np.angle(B))

def sliding_phase_variance(phase, window_steps):
    """σ²_φ(t) over sliding window — the apoptosis biomarker."""
    var = np.zeros_like(phase)
    half = window_steps // 2
    for i in range(len(phase)):
        lo = max(0, i - half)
        hi = min(len(phase), i + half + 1)
        var[i] = np.var(phase[lo:hi])
    return var

def sliding_mean(arr, window_steps):
    mean = np.zeros_like(arr)
    half = window_steps // 2
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        mean[i] = np.mean(arr[lo:hi])
    return mean

# ============================================================
# Simulate three states and compute Δt detection window
# ============================================================
kappa_Q = kappa_quiescent(t, kappa_match)
kappa_M = kappa_mitotic(t, kappa_match, y0=900.0)
kappa_A = kappa_apoptotic(t, kappa_match, t_onset=600.0)

B_Q = bloom_integral(kappa_Q, sigma)
B_M = bloom_integral(kappa_M, sigma)
B_A = bloom_integral(kappa_A, sigma)

window_sec = 30.0
window_steps = int(window_sec / dt)

# Apoptotic state: compute Δt
amp_A = amplitude_intensity(B_A)
phs_A = phase_arg(B_A)
sigma2_phi_A = sliding_phase_variance(phs_A, window_steps)
mean_amp_A   = sliding_mean(amp_A, window_steps)

# Baselines from pre-onset window
baseline_idx = (t < 400.0)
phi_baseline = np.mean(sigma2_phi_A[baseline_idx]) + 3*np.std(sigma2_phi_A[baseline_idx])
amp_baseline = np.mean(mean_amp_A[baseline_idx]) - 3*np.std(mean_amp_A[baseline_idx])

# Detection times
phase_breach = np.where(sigma2_phi_A > phi_baseline)[0]
amp_breach   = np.where(mean_amp_A < amp_baseline)[0]

# Filter post-onset
phase_breach = phase_breach[t[phase_breach] > 400.0]
amp_breach   = amp_breach[t[amp_breach] > 400.0]

t_phase_first = t[phase_breach[0]] if len(phase_breach) else None
t_amp_first   = t[amp_breach[0]]   if len(amp_breach)   else None
if t_phase_first is not None and t_amp_first is not None:
    delta_t = t_amp_first - t_phase_first
else:
    delta_t = None

print("=" * 60)
print("Δt apoptosis detection window")
print("=" * 60)
print(f"Phase-variance breach at t = {t_phase_first:.1f} s" if t_phase_first else "Phase: no breach")
print(f"Amplitude collapse at  t = {t_amp_first:.1f} s" if t_amp_first else "Amplitude: no breach")
if delta_t is not None:
    print(f"Δt = {delta_t:.1f} s  ({delta_t/60:.2f} min)")
    print(f"Preprint window: 5–30 min — {'WITHIN' if 300<=delta_t<=1800 else 'outside'} predicted range")

# ============================================================
# Order asymmetry validation: phase response ~ κ, amp ~ κ²
# ============================================================
kappa_test = np.linspace(-0.5*kappa_match, 0.5*kappa_match, 200)
B_test = bloom_integral(kappa_test, sigma)
phase_resp = np.angle(B_test)
amp_resp   = 1.0 - np.abs(B_test)**2

# Fit phase ~ a*κ and amp ~ b*κ²
p_phase = np.polyfit(kappa_test, phase_resp, 1)
p_amp   = np.polyfit(kappa_test, amp_resp, 2)
print("\n" + "=" * 60)
print("Order asymmetry verification (preprint Eqs. 12, 13)")
print("=" * 60)
print(f"Phase fit linear coeff: {p_phase[0]:+.5f}  (preprint: −σ²/2 = {-sigma**2/2:+.5f})")
print(f"Amp   fit quadratic coeff: {p_amp[0]:+.5f}  (preprint: +σ⁴/2 = {sigma**4/2:+.5f})")
print("→ Phase channel linear in κ; amplitude channel quadratic in κ")
print("→ This is the structural origin of the 2:1 pattern attractor")

# ============================================================
# σ-sweep — resolution-resolved observable f(α; σ)
# ============================================================
sigma_sweep = np.array([0.5, 1.0, 2.0, 4.0, 8.0])  # µm
print("\n" + "=" * 60)
print("σ-sweep (resolution-resolved response, preprint Section 7)")
print("=" * 60)
sigma_results = []
for s_um in sigma_sweep:
    B_s = bloom_integral(kappa_A, s_um)
    amp_var_s = np.var(np.abs(B_s)**2)
    phs_var_s = np.var(np.unwrap(np.angle(B_s)))
    ratio = phs_var_s / (amp_var_s + 1e-12)
    sigma_results.append((s_um, amp_var_s, phs_var_s, ratio))
    print(f"σ = {s_um:5.2f} µm   amp_var = {amp_var_s:.4f}   phs_var = {phs_var_s:7.3f}   phs/amp = {ratio:8.2f}")

# ============================================================
# Visualization
# ============================================================
fig = plt.figure(figsize=(13, 12))
fig.patch.set_facecolor('#0a0a0f')
gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.28)

def style(ax):
    ax.set_facecolor('#0a0a0f')
    for s in ax.spines.values(): s.set_color('#6a6a8a')
    ax.tick_params(colors='#b8b8d0')
    ax.xaxis.label.set_color('#d8d8ee')
    ax.yaxis.label.set_color('#d8d8ee')
    ax.title.set_color('#e8e8ff')
    ax.grid(True, alpha=0.15, color='#5a5a7a')

# Row 1: three cellular states — κ(t) trajectories
ax = fig.add_subplot(gs[0, 0])
ax.plot(t, kappa_Q*sigma**2, color='#5affc0', lw=0.7, label='Q (quiescent)')
ax.plot(t, kappa_M*sigma**2, color='#ffd35a', lw=0.7, label='M (mitotic)')
ax.plot(t, kappa_A*sigma**2, color='#ff5aa0', lw=0.7, label='A (apoptotic)')
ax.axhline(1.0, color='#9090b0', lw=0.4, ls='--', alpha=0.5, label='κσ² = 1 (matching)')
ax.set_xlabel('time (s)')
ax.set_ylabel('κ(t) σ²  (dimensionless)')
ax.set_title('Cellular-state curvature trajectories')
ax.legend(facecolor='#15151f', edgecolor='#5a5a7a', labelcolor='#d8d8ee', loc='upper right', fontsize=8)
style(ax)

# Row 1: complex B̃ trajectories in phase space
ax = fig.add_subplot(gs[0, 1])
ax.plot(B_Q.real, B_Q.imag, color='#5affc0', lw=0.4, alpha=0.7, label='Q')
ax.plot(B_M.real, B_M.imag, color='#ffd35a', lw=0.4, alpha=0.7, label='M')
ax.plot(B_A.real, B_A.imag, color='#ff5aa0', lw=0.4, alpha=0.7, label='A')
ax.set_xlabel('Re B̃')
ax.set_ylabel('Im B̃')
ax.set_title('Bloom-integral phase-space trajectories')
ax.set_aspect('equal')
ax.legend(facecolor='#15151f', edgecolor='#5a5a7a', labelcolor='#d8d8ee')
style(ax)

# Row 2: apoptosis biomarker — the key plot
ax = fig.add_subplot(gs[1, :])
ax2 = ax.twinx()
ax.plot(t, sigma2_phi_A, color='#ff5aa0', lw=1.2, label='σ²_φ phase variance')
ax2.plot(t, mean_amp_A, color='#5ad8ff', lw=1.2, label='⟨|B̃|²⟩ mean intensity')
ax.axhline(phi_baseline, color='#ff5aa0', lw=0.5, ls=':', alpha=0.6)
ax2.axhline(amp_baseline, color='#5ad8ff', lw=0.5, ls=':', alpha=0.6)
if t_phase_first:
    ax.axvline(t_phase_first, color='#ff5aa0', lw=1.0, alpha=0.7,
               label=f'phase breach t={t_phase_first:.0f}s')
if t_amp_first:
    ax.axvline(t_amp_first, color='#5ad8ff', lw=1.0, alpha=0.7,
               label=f'amp collapse t={t_amp_first:.0f}s')
if delta_t:
    ax.axvspan(t_phase_first, t_amp_first, color='#ffd35a', alpha=0.10)
    ax.text((t_phase_first+t_amp_first)/2, ax.get_ylim()[1]*0.85,
            f'Δt = {delta_t/60:.1f} min',
            color='#ffd35a', ha='center', fontsize=11, weight='bold')
ax.set_xlabel('time (s)')
ax.set_ylabel('phase variance σ²_φ', color='#ff5aa0')
ax2.set_ylabel('mean intensity ⟨|B̃|²⟩', color='#5ad8ff')
ax.tick_params(axis='y', colors='#ff5aa0')
ax2.tick_params(axis='y', colors='#5ad8ff')
ax.set_title('Apoptosis biomarker — phase decoherence precedes amplitude collapse (preprint Section 4)')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2,
          facecolor='#15151f', edgecolor='#5a5a7a', labelcolor='#d8d8ee', loc='upper left')
style(ax)
for s in ax2.spines.values(): s.set_color('#6a6a8a')

# Row 3: order asymmetry verification
ax = fig.add_subplot(gs[2, 0])
ax.plot(kappa_test*sigma**2, phase_resp, color='#ff5aa0', lw=1.4, label='arg B̃ (linear in κ)')
ax.plot(kappa_test*sigma**2, np.polyval(p_phase, kappa_test),
        color='#ff5aa0', lw=0.6, ls='--', alpha=0.7, label='linear fit')
ax.set_xlabel('κσ²')
ax.set_ylabel('arg B̃')
ax.set_title('Phase channel — linear in κ')
ax.legend(facecolor='#15151f', edgecolor='#5a5a7a', labelcolor='#d8d8ee')
style(ax)

ax = fig.add_subplot(gs[2, 1])
ax.plot(kappa_test*sigma**2, amp_resp, color='#5ad8ff', lw=1.4, label='1 − |B̃|² (quadratic in κ)')
ax.plot(kappa_test*sigma**2, np.polyval(p_amp, kappa_test),
        color='#5ad8ff', lw=0.6, ls='--', alpha=0.7, label='quadratic fit')
ax.set_xlabel('κσ²')
ax.set_ylabel('1 − |B̃|²')
ax.set_title('Amplitude channel — quadratic in κ (this is the 2:1 attractor)')
ax.legend(facecolor='#15151f', edgecolor='#5a5a7a', labelcolor='#d8d8ee')
style(ax)

# Row 4: σ-sweep — resolution-resolved channel ratio
ax = fig.add_subplot(gs[3, :])
sweep_arr = np.array(sigma_results)
ax.semilogy(sweep_arr[:,0], sweep_arr[:,1], 'o-', color='#5ad8ff', lw=1.4, ms=8, label='var(|B̃|²)')
ax.semilogy(sweep_arr[:,0], sweep_arr[:,2], 's-', color='#ff5aa0', lw=1.4, ms=8, label='var(arg B̃)')
ax2 = ax.twinx()
ax2.plot(sweep_arr[:,0], sweep_arr[:,3], '^--', color='#ffd35a', lw=1.0, ms=7, label='phase/amp ratio')
ax.set_xlabel('sentinel waist σ (µm)')
ax.set_ylabel('channel variance', color='#d8d8ee')
ax2.set_ylabel('phase/amplitude variance ratio', color='#ffd35a')
ax2.tick_params(colors='#ffd35a')
ax.set_title('Resolution-resolved spectroscopy — sweep σ across cellular→organelle scales (preprint Section 7)')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2,
          facecolor='#15151f', edgecolor='#5a5a7a', labelcolor='#d8d8ee', loc='upper left')
style(ax)
for s in ax2.spines.values(): s.set_color('#6a6a8a')

fig.suptitle('Bloom Bridge — nihiline recursion grounded in Phase-Curvature framework',
             color='#e8e8ff', fontsize=13, y=0.995)
plt.savefig('/mnt/user-data/outputs/bloom_bridge.png', dpi=140,
            facecolor='#0a0a0f', bbox_inches='tight')
print("\nFigure saved.")
