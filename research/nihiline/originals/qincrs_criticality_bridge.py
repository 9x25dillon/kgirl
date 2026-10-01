#!/usr/bin/env python3
"""
QINCRS Criticality Bridge — Selection Amplification at Phase Boundaries
========================================================================
Can a system near a critical point amplify an infinitesimal coherent
bias (~10^-50) into a measurable state-selection asymmetry?

This script implements three canonical models of critical amplification:

  Model 1: Ising Ferromagnet at T_c
    - Mean-field Ising model with external field h -> 0
    - Susceptibility chi diverges at T_c: chi ~ |T - T_c|^{-gamma}
    - Amplification = chi * h (linear response near criticality)
    - Question: can divergent chi compensate for h ~ 10^{-50}?

  Model 2: Bistable Genetic Switch (Stochastic)
    - Two-state system with noise-driven transitions
    - Kramers escape rate over barrier Delta_G
    - Bias epsilon tilts the double-well
    - Question: does epsilon ~ 10^{-50} shift occupation probabilities?

  Model 3: Coherence Resonance / Stochastic Resonance
    - Noise + weak periodic signal near a threshold
    - Signal-to-noise ratio peaks at optimal noise level
    - Question: can SR amplify a coherent signal below thermal noise?

For each model, we compute the MINIMUM BIAS required to produce
a 1% shift in state occupation over biologically relevant timescales.
Then we compare with the planetary EM bias.

The answer determines whether the planetary coherence hypothesis
lives or dies.

Authors: K1ll / Dr. Aris Thorne — QINCRS Collaboration
Date: April 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import brentq
from scipy.special import erf
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTS
# ============================================================
KB      = 1.380649e-23     # J/K
HBAR    = 1.0545718e-34    # J·s
EV_TO_J = 1.6021766e-19   # eV -> J
T_BODY  = 310.15           # 37 C in Kelvin
KBT     = KB * T_BODY      # thermal energy at body temp

# Planetary bias from our calculation
PLANETARY_BIAS_DK = 1e-50  # dk/k0 from planetary EM (order of magnitude)
# Convert to energy bias: epsilon ~ kBT * (dk/k0) for thermal systems
PLANETARY_EPSILON_J = KBT * PLANETARY_BIAS_DK

print("=" * 75)
print("QINCRS CRITICALITY BRIDGE")
print("Selection Amplification at Phase Boundaries")
print("=" * 75)
print(f"\nPlanetary EM bias (dk/k0):     {PLANETARY_BIAS_DK:.1e}")
print(f"Thermal energy at 37 C (kBT):  {KBT:.4e} J = {KBT/EV_TO_J*1000:.2f} meV")
print(f"Planetary energy bias:          {PLANETARY_EPSILON_J:.1e} J")
print(f"Planetary bias / kBT:           {PLANETARY_BIAS_DK:.1e}")


# ============================================================
# MODEL 1: MEAN-FIELD ISING AT CRITICALITY
# ============================================================
print("\n" + "=" * 75)
print("MODEL 1: ISING FERROMAGNET NEAR T_c")
print("=" * 75)

def ising_magnetization_mf(T_reduced, h_reduced, N_spins):
    """
    Mean-field Ising magnetization.

    Self-consistency equation: m = tanh(m/T_reduced + h_reduced/T_reduced)

    T_reduced = T / T_c (so T_c = 1)
    h_reduced = h / (J * z) where J = coupling, z = coordination number

    Returns magnetization m in [-1, 1]
    """
    def self_consistency(m):
        return m - np.tanh((m + h_reduced) / T_reduced)

    # For h > 0, solution is positive
    if abs(h_reduced) < 1e-100:
        if T_reduced < 1.0:
            # Below Tc: spontaneous magnetization
            # Find positive solution
            try:
                m = brentq(self_consistency, 0.001, 0.999)
            except:
                m = 0.0
        else:
            m = 0.0
    else:
        try:
            if h_reduced > 0:
                m = brentq(self_consistency, 1e-200, 0.9999)
            else:
                m = brentq(self_consistency, -0.9999, -1e-200)
        except:
            # Linear response: m ~ chi * h
            if T_reduced > 1:
                chi = 1.0 / (T_reduced - 1.0)
            else:
                chi = 1e10  # divergent
            m = min(chi * h_reduced / T_reduced, 0.999)

    return m


def ising_susceptibility(T_reduced):
    """
    Mean-field susceptibility chi = 1 / |T/Tc - 1| (Curie-Weiss law)
    Diverges at T_c.
    For finite system of N spins: chi_max ~ N^{gamma/nu*d} ~ N^{2/3} (3D)
    """
    if abs(T_reduced - 1.0) < 1e-10:
        return 1e10  # regularized divergence
    if T_reduced > 1:
        return 1.0 / (T_reduced - 1.0)
    else:
        return 1.0 / (2.0 * (1.0 - T_reduced))


def ising_finite_size_chi_max(N_spins, d=3):
    """
    Maximum susceptibility for finite-size system.
    chi_max ~ N^{gamma/(nu*d)} = N^{2/3} for 3D Ising (gamma=1.24, nu=0.63)
    """
    gamma_nu_d = 1.24 / (0.63 * d)  # ~ 0.656
    return N_spins ** gamma_nu_d


print("\n--- Ising Model Analysis ---")
print(f"Mean-field critical exponent gamma = 1 (MF), 1.24 (3D)")

# Relevant biological system sizes
system_sizes = {
    'Single protein (100 residues)':     100,
    'Microtubule segment (1000 tubulin)': 1000,
    'Microtubule (13k tubulin)':          13000,
    'Cellular network (10^6 nodes)':      1e6,
    'Neural column (10^5 neurons)':       1e5,
    'Brain network (10^11 neurons)':      1e11,
}

print(f"\n{'System':<42} {'N':<14} {'chi_max':<14} {'m at h=10^-50':<18} {'Amplification':<16}")
print("-" * 104)

ising_results = {}
for sys_name, N in system_sizes.items():
    chi_max = ising_finite_size_chi_max(N)
    # Magnetization response: m = chi * h / kBT (linear response)
    # h_eff = planetary bias in units of coupling energy
    h_eff = PLANETARY_BIAS_DK  # dimensionless bias
    m_response = chi_max * h_eff
    amplification = chi_max

    ising_results[sys_name] = {
        'N': N, 'chi_max': chi_max,
        'm': m_response, 'amp': amplification
    }

    print(f"{sys_name:<42} {N:<14.0e} {chi_max:<14.2e} {m_response:<18.2e} {amplification:<16.2e}")

print(f"\n>>> VERDICT: Even brain-scale networks (N=10^11) give chi_max ~ 10^7")
print(f">>> Amplification of 10^-50 bias: 10^7 * 10^-50 = 10^-43")
print(f">>> ISING CRITICALITY CANNOT BRIDGE THE GAP.")
print(f">>> Required amplification: 10^48. Maximum available: ~10^7.")
print(f">>> Shortfall: ~41 ORDERS OF MAGNITUDE.")


# ============================================================
# MODEL 2: BISTABLE GENETIC SWITCH (KRAMERS)
# ============================================================
print("\n" + "=" * 75)
print("MODEL 2: BISTABLE GENETIC SWITCH (KRAMERS ESCAPE)")
print("=" * 75)

def kramers_rate(delta_G_kBT, attempt_freq=1e9):
    """
    Kramers escape rate over barrier of height Delta_G.
    k = f_attempt * exp(-Delta_G / kBT)

    delta_G_kBT: barrier height in units of kBT
    attempt_freq: characteristic attempt frequency (Hz)
    """
    return attempt_freq * np.exp(-delta_G_kBT)


def bistable_occupation_ratio(delta_G_kBT, epsilon_kBT):
    """
    For a bistable system with barrier Delta_G and asymmetry epsilon:

    P(A) / P(B) = exp(epsilon / kBT)

    The bias epsilon shifts the occupation probability.

    Returns P(A), P(B) where state A is favored by epsilon > 0.
    """
    ratio = np.exp(epsilon_kBT)
    P_A = ratio / (1 + ratio)
    P_B = 1.0 / (1 + ratio)
    return P_A, P_B


print("\n--- Kramers Escape Analysis ---")
print(f"Planetary bias in kBT units: epsilon/kBT = {PLANETARY_BIAS_DK:.1e}")
print(f"(This is the same as dk/k0 since the bias acts on the barrier)")

# Barrier heights for various biological switches
barriers = {
    'Near-zero barrier (epsilon ~ kBT)':  0.01,
    'Weak switch (5 kBT)':                5.0,
    'Moderate switch (10 kBT)':           10.0,
    'Strong switch (20 kBT)':             20.0,
    'Robust genetic toggle (30 kBT)':     30.0,
}

print(f"\n{'Switch Type':<42} {'Barrier (kBT)':<16} {'P(A) unbiased':<16} "
      f"{'P(A) biased':<16} {'Delta P':<16}")
print("-" * 106)

for sw_name, dG in barriers.items():
    P_A_0, P_B_0 = bistable_occupation_ratio(dG, 0.0)
    P_A_eps, P_B_eps = bistable_occupation_ratio(dG, PLANETARY_BIAS_DK)
    delta_P = abs(P_A_eps - P_A_0)

    print(f"{sw_name:<42} {dG:<16.2f} {P_A_0:<16.6f} {P_A_eps:<16.6f} {delta_P:<16.2e}")

print(f"\n>>> VERDICT: Kramers switching faithfully transmits the bias.")
print(f">>> But it does not amplify it. Delta_P ~ epsilon/kBT = 10^-50.")
print(f">>> A bistable switch is NOT a selection amplifier for this scale.")
print(f">>> The occupation shift is real but unmeasurably small.")


# ============================================================
# MODEL 3: STOCHASTIC RESONANCE
# ============================================================
print("\n" + "=" * 75)
print("MODEL 3: STOCHASTIC RESONANCE")
print("=" * 75)

def sr_snr(signal_amplitude, noise_intensity, barrier_height, omega_signal):
    """
    Signal-to-noise ratio in classic stochastic resonance.

    SNR ~ (A^2 / D^2) * exp(-2 * Delta_V / D)

    where A = signal amplitude, D = noise intensity, Delta_V = barrier
    omega = signal frequency

    Peak SNR occurs at D_opt ~ Delta_V / ln(Delta_V / A)

    Returns SNR (dimensionless)
    """
    if noise_intensity <= 0 or barrier_height <= 0:
        return 0.0

    # Kramers rate at this noise level
    r_K = omega_signal * np.exp(-barrier_height / noise_intensity)

    # SR formula (two-state theory, McNamara & Wiesenfeld 1989)
    if r_K > 0 and noise_intensity > 0:
        snr = (np.pi * signal_amplitude**2) / (2 * noise_intensity**2) * \
              np.exp(-barrier_height / noise_intensity)
    else:
        snr = 0.0
    return max(snr, 1e-300)


print("\n--- Stochastic Resonance Analysis ---")
print("Question: can SR amplify a 10^-50 signal to detectable levels?")
print()

# Signal: planetary coherent bias
A_signal = PLANETARY_BIAS_DK  # dimensionless, in units of kBT

# Barrier: biological switch barrier
Delta_V = 5.0  # kBT (relatively shallow for SR to work)

# Noise: thermal fluctuations (D = kBT in these units, so D = 1)
D_range = np.logspace(-2, 2, 1000)  # noise intensity in kBT

# Signal frequency: solar cycle
omega_signal = 2 * np.pi / (11.0 * 365.25 * 24 * 3600)  # rad/s

snr_vals = [sr_snr(A_signal, D, Delta_V, omega_signal) for D in D_range]

# Find peak SNR
peak_idx = np.argmax(snr_vals)
peak_D = D_range[peak_idx]
peak_SNR = snr_vals[peak_idx]

print(f"Signal amplitude (planetary bias): A = {A_signal:.1e} kBT")
print(f"Barrier height: Delta_V = {Delta_V:.1f} kBT")
print(f"Optimal noise intensity: D_opt = {peak_D:.4f} kBT")
print(f"Peak SNR at optimal noise: {peak_SNR:.4e}")
print()
print(f">>> SNR ~ A^2 / D^2 * exp(-Delta_V/D)")
print(f">>> With A = 10^-50: peak SNR ~ (10^-50)^2 * prefactor = 10^-100 * prefactor")
print(f">>> No amount of noise optimization rescues a 10^-100 baseline.")
print(f">>> STOCHASTIC RESONANCE CANNOT BRIDGE THE GAP.")

# Now compute: what is the MINIMUM signal amplitude for SR to produce SNR > 1?
print(f"\n--- Minimum detectable signal via SR ---")

def min_signal_for_snr1(Delta_V, D_opt=None):
    """Find minimum A such that peak SNR >= 1"""
    if D_opt is None:
        # Optimal noise for SR: D_opt ~ Delta_V / ln(something)
        # Approximate: D_opt ~ Delta_V / 2 for modest barriers
        D_opt = Delta_V / 2.0
    # SNR ~ (pi * A^2 / (2 * D^2)) * exp(-DV/D)
    # Set SNR = 1:
    # A^2 = 2 * D^2 / (pi * exp(-DV/D))
    A_min = np.sqrt(2 * D_opt**2 / (np.pi * np.exp(-Delta_V / D_opt)))
    return A_min

for dv in [1.0, 3.0, 5.0, 10.0, 20.0]:
    A_min = min_signal_for_snr1(dv)
    shortfall = np.log10(A_min) - np.log10(A_signal) if A_signal > 0 else float('inf')
    print(f"  Barrier {dv:5.1f} kBT: A_min for SNR=1 = {A_min:.4e} kBT "
          f"(shortfall: {shortfall:.0f} orders of magnitude)")


# ============================================================
# MODEL 4: THE ACTUAL QUESTION — CRITICAL SLOWING DOWN
#           AND INTEGRATION TIME
# ============================================================
print("\n" + "=" * 75)
print("MODEL 4: CRITICAL SLOWING DOWN + TEMPORAL INTEGRATION")
print("=" * 75)

print("""
Key insight: all three models above assume INSTANTANEOUS response.
But biology operates over LONG TIMESCALES.

If a system near criticality integrates a coherent bias over time T,
the effective signal grows as:

  S_eff ~ epsilon * sqrt(T / tau_corr)  (for T >> tau_corr)

where tau_corr is the correlation time, which DIVERGES at criticality:

  tau_corr ~ |T - T_c|^{-nu*z}  (critical slowing down)

Near T_c, tau_corr -> infinity, and the system integrates coherently
for arbitrarily long times.

BUT: the noise also integrates. The noise grows as sqrt(T).
So the signal-to-noise ratio grows as:

  SNR(T) ~ epsilon * sqrt(T / tau_corr) / sqrt(kBT * T / tau_corr)
         = epsilon / sqrt(kBT)

THIS DOES NOT GROW WITH TIME.

The SNR for a biased random walk is:

  SNR = epsilon * T / sqrt(D * T) = epsilon * sqrt(T/D)

where D is the diffusion constant.
""")

def biased_random_walk_snr(epsilon, D, T):
    """
    SNR for a biased random walk after time T.
    Mean displacement: epsilon * T
    RMS fluctuation: sqrt(2 * D * T)
    SNR = epsilon * T / sqrt(2 * D * T) = epsilon * sqrt(T / (2*D))
    """
    return epsilon * np.sqrt(T / (2 * D))


# Integration times
integration_times = {
    'Cell cycle (24 hr)':           24 * 3600,
    'Month':                        30 * 24 * 3600,
    'Year':                         365.25 * 24 * 3600,
    'Human generation (30 yr)':     30 * 365.25 * 24 * 3600,
    'Millennium':                   1000 * 365.25 * 24 * 3600,
    'Geological epoch (1 Myr)':     1e6 * 365.25 * 24 * 3600,
    'Earth history (4.5 Gyr)':      4.5e9 * 365.25 * 24 * 3600,
    'Universe age (13.8 Gyr)':      13.8e9 * 365.25 * 24 * 3600,
}

print(f"\n--- Temporal Integration of Planetary Bias ---")
print(f"Bias: epsilon = {PLANETARY_BIAS_DK:.1e} (dimensionless)")
print(f"Diffusion: D = 1 (normalized to kBT units)")
print(f"\n{'Integration Time':<32} {'T (seconds)':<16} {'SNR':<16} {'Detectable?':<12}")
print("-" * 76)

for t_name, T in integration_times.items():
    snr = biased_random_walk_snr(PLANETARY_BIAS_DK, 1.0, T)
    detectable = "YES" if snr > 1 else "no"
    print(f"{t_name:<32} {T:<16.4e} {snr:<16.4e} {detectable:<12}")

# What integration time would be needed?
T_needed = 2.0 / PLANETARY_BIAS_DK**2  # SNR = 1 when T = 2D/eps^2
print(f"\n>>> Time needed for SNR = 1: T = 2D/eps^2 = {T_needed:.1e} seconds")
print(f">>> That's {T_needed / (365.25*24*3600):.1e} years")
print(f">>> Universe age: {13.8e9:.1e} years")
print(f">>> Shortfall: {np.log10(T_needed/(365.25*24*3600)) - np.log10(13.8e9):.0f} orders of magnitude")


# ============================================================
# MODEL 5: THE HONEST SYNTHESIS — WHAT CAN WORK?
# ============================================================
print("\n" + "=" * 75)
print("MODEL 5: HONEST SYNTHESIS — WHAT MINIMUM BIAS IS REQUIRED?")
print("=" * 75)

print("""
Working backward from the requirement:

To produce a 1% shift in state occupation (Delta_P = 0.01)
in a bistable biological system over one generation (30 years):

  1. Direct bias: epsilon = Delta_P = 0.01 in kBT units
     -> Required field: E_0 ~ 10^8 V/m (from Val-01 adiabatic model)
     -> Available (planetary): E_0 ~ 10^-16 V/m
     -> IMPOSSIBLE

  2. Via stochastic resonance (optimal noise):
     -> Minimum signal for SNR=1: A_min ~ 10^-1 to 10^0 kBT
     -> Available: 10^-50 kBT
     -> IMPOSSIBLE

  3. Via Ising criticality (N ~ 10^11 brain):
     -> Amplification: chi_max ~ 10^7
     -> Required bias: 0.01 / 10^7 = 10^-9 kBT
     -> Available: 10^-50 kBT
     -> IMPOSSIBLE (shortfall: 41 orders of magnitude)

  4. Via temporal integration (biased random walk):
     -> Available time: 10^10 years (universe age)
     -> SNR achieved: 10^-50 * sqrt(10^17) = 10^-50 * 10^8.5 = 10^-41.5
     -> IMPOSSIBLE

  5. ALL MECHANISMS COMBINED (multiplicative):
     -> chi_max * sqrt(T) * SR_gain
     -> 10^7 * 10^8.5 * 10^0 = 10^15.5
     -> Applied to 10^-50 bias: 10^-34.5
     -> STILL IMPOSSIBLE (shortfall: ~35 orders of magnitude)
""")

# ============================================================
# COMPUTE: What IS the minimum bias that biology can detect?
# ============================================================
print("=" * 75)
print("THE REAL QUESTION: What is the minimum detectable bias?")
print("=" * 75)

# Best case scenario: brain-scale critical system, integrating for a year
N_brain = 1e11
chi_brain = ising_finite_size_chi_max(N_brain)
T_year = 365.25 * 24 * 3600
temporal_gain = np.sqrt(T_year)  # biased walk SNR gain

total_amp = chi_brain * temporal_gain

epsilon_min = 0.01 / total_amp  # for 1% shift
epsilon_min_1pct = epsilon_min

print(f"\nBest-case biological amplification:")
print(f"  Critical susceptibility (N=10^11): chi = {chi_brain:.2e}")
print(f"  Temporal integration (1 year):     sqrt(T) = {temporal_gain:.2e}")
print(f"  Total amplification:               {total_amp:.2e}")
print(f"\n  Minimum bias for 1% effect:        epsilon_min = {epsilon_min:.2e} kBT")
print(f"  In eV:                             {epsilon_min * KBT / EV_TO_J:.2e} eV")
print(f"  In V/m (using Val-01 coupling):    ~10^{np.log10(epsilon_min)+8:.0f} V/m")

print(f"\n  Planetary bias available:          {PLANETARY_BIAS_DK:.1e} kBT")
print(f"  Shortfall:                         {np.log10(epsilon_min) - np.log10(PLANETARY_BIAS_DK):.0f} orders of magnitude")

# ============================================================
# FIGURES
# ============================================================
print("\n\nGenerating figures...")

fig = plt.figure(figsize=(18, 20))
gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)

# --- Panel A: Ising susceptibility vs system size ---
ax_a = fig.add_subplot(gs[0, 0])
N_range = np.logspace(1, 12, 200)
chi_vals = [ising_finite_size_chi_max(N) for N in N_range]
ax_a.loglog(N_range, chi_vals, 'b-', linewidth=2.5)

# Mark biological systems
for sys_name, res in ising_results.items():
    ax_a.plot(res['N'], res['chi_max'], 'ro', markersize=8, zorder=5)
    short_name = sys_name.split('(')[0].strip()
    ax_a.annotate(short_name, (res['N'], res['chi_max']),
                  textcoords='offset points', xytext=(10, 5), fontsize=7)

# Required amplification
ax_a.axhline(y=1e48, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
ax_a.text(20, 2e48, 'Required: $10^{48}$', fontsize=10, color='red')

ax_a.set_xlabel('System Size N', fontsize=11)
ax_a.set_ylabel('Maximum Susceptibility $\\chi_{\\rm max}$', fontsize=11)
ax_a.set_title('(a) Ising Criticality: $\\chi_{\\rm max} \\sim N^{0.66}$', fontsize=12, fontweight='bold')
ax_a.set_ylim(1, 1e52)
ax_a.grid(True, alpha=0.3, which='both')

# --- Panel B: Stochastic resonance SNR ---
ax_b = fig.add_subplot(gs[0, 1])

signal_amplitudes = [1e-1, 1e-5, 1e-10, 1e-20, 1e-50]
colors_sr = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#264653']

for A, col in zip(signal_amplitudes, colors_sr):
    snr_vals_plot = [sr_snr(A, D, 5.0, omega_signal) for D in D_range]
    ax_b.loglog(D_range, snr_vals_plot, color=col, linewidth=2,
                label=f'A = {A:.0e}')

ax_b.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=1.5,
             label='SNR = 1 (detection)')
ax_b.set_xlabel('Noise Intensity D (kBT)', fontsize=11)
ax_b.set_ylabel('Signal-to-Noise Ratio', fontsize=11)
ax_b.set_title('(b) Stochastic Resonance: Peak SNR vs Signal Strength', fontsize=12, fontweight='bold')
ax_b.legend(fontsize=8, loc='upper right')
ax_b.set_ylim(1e-120, 1e5)
ax_b.grid(True, alpha=0.3, which='both')

# --- Panel C: Temporal integration ---
ax_c = fig.add_subplot(gs[1, 0])

bias_values = [1e-2, 1e-5, 1e-10, 1e-20, 1e-30, 1e-50]
T_range_sec = np.logspace(0, 20, 200)
T_range_years = T_range_sec / (365.25 * 24 * 3600)

for eps, col in zip(bias_values, ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261', '#264653']):
    snr_time = [biased_random_walk_snr(eps, 1.0, T) for T in T_range_sec]
    ax_c.loglog(T_range_years, snr_time, color=col, linewidth=2,
                label=f'$\\epsilon$ = {eps:.0e}')

ax_c.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
ax_c.axvline(x=30, color='gray', linestyle=':', alpha=0.5, label='1 generation')
ax_c.axvline(x=13.8e9, color='purple', linestyle=':', alpha=0.5, label='Universe age')

ax_c.set_xlabel('Integration Time (years)', fontsize=11)
ax_c.set_ylabel('SNR (biased random walk)', fontsize=11)
ax_c.set_title('(c) Temporal Integration: SNR $\\sim \\epsilon\\sqrt{T}$', fontsize=12, fontweight='bold')
ax_c.legend(fontsize=8, ncol=2)
ax_c.set_xlim(1e-7, 1e12)
ax_c.grid(True, alpha=0.3, which='both')

# --- Panel D: The gap visualization ---
ax_d = fig.add_subplot(gs[1, 1])

mechanisms = [
    ('Planetary\nEM bias', np.log10(PLANETARY_BIAS_DK), '#264653'),
    ('+ Ising\ncriticality\n($N=10^{11}$)', np.log10(PLANETARY_BIAS_DK) + 7, '#457B9D'),
    ('+ Temporal\nintegration\n(30 yr)', np.log10(PLANETARY_BIAS_DK) + 7 + 4.2, '#2A9D8F'),
    ('+ Stochastic\nresonance\n(optimal)', np.log10(PLANETARY_BIAS_DK) + 7 + 4.2 + 0, '#E9C46A'),
    ('ALL\nCOMBINED', np.log10(PLANETARY_BIAS_DK) + 7 + 4.2, '#F4A261'),
]

x_pos = np.arange(len(mechanisms))
values = [m[1] for m in mechanisms]
colors_gap = [m[2] for m in mechanisms]
labels_gap = [m[0] for m in mechanisms]

bars = ax_d.bar(x_pos, values, color=colors_gap, edgecolor='black', linewidth=0.5,
                alpha=0.85, width=0.6, bottom=0)

# Target line
ax_d.axhline(y=np.log10(0.01), color='red', linestyle='-', linewidth=2.5,
             label='Required: $\\Delta P = 1\\%$')
ax_d.axhline(y=np.log10(1e-4), color='orange', linestyle='--', linewidth=1.5,
             label='Measurable: $\\Delta P = 0.01\\%$')

ax_d.set_xticks(x_pos)
ax_d.set_xticklabels(labels_gap, fontsize=8, ha='center')
ax_d.set_ylabel('log$_{10}$(effective bias)', fontsize=11)
ax_d.set_title('(d) The Amplification Gap', fontsize=12, fontweight='bold')
ax_d.legend(fontsize=9, loc='upper right')
ax_d.set_ylim(-55, 0)
ax_d.grid(True, alpha=0.2, axis='y')

# Add gap annotation
gap = np.log10(0.01) - (np.log10(PLANETARY_BIAS_DK) + 7 + 4.2)
ax_d.annotate('', xy=(4, np.log10(0.01)), xytext=(4, np.log10(PLANETARY_BIAS_DK) + 11.2),
              arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax_d.text(4.4, -25, f'GAP:\n~{abs(gap):.0f} orders\nof magnitude',
          fontsize=11, color='red', fontweight='bold')

# --- Panel E: Required field strength for detectable effect ---
ax_e = fig.add_subplot(gs[2, 0])

# From Val-01: dk/k0 = (alpha * q * x0 * E0)^2 / 4 * C
# Invert: E0 = sqrt(4 * dk / (alpha * q * x0)^2 / C)
alpha_gt = 2.8782e20  # most sensitive base pair
x0 = 1e-11
q = 1.6021766e-19

dk_targets = np.logspace(-60, 0, 200)
E0_required = np.sqrt(4 * dk_targets / (alpha_gt * q * x0)**2)

ax_e.loglog(dk_targets, E0_required, 'b-', linewidth=2.5)

# Mark key levels
field_levels = {
    'Planetary surface ($\\sim 10^{-16}$ V/m)': 1e-16,
    'Schumann cavity ($\\sim 10^{-3}$ V/m)': 1e-3,
    'Lab THz ($10^6$ V/m)': 1e6,
    'Focused THz ($10^8$ V/m)': 1e8,
}

for fl_name, fl_val in field_levels.items():
    ax_e.axhline(y=fl_val, color='gray', linestyle=':', alpha=0.5)
    ax_e.text(1e-58, fl_val * 1.5, fl_name, fontsize=8, color='gray')

# Mark detectability
ax_e.axvline(x=1e-4, color='red', linestyle='--', alpha=0.7,
             label='$\\Delta k/k_0 = 10^{-4}$ (detectable)')
ax_e.fill_betweenx([1e-20, 1e12], 1e-4, 1, alpha=0.1, color='green')
ax_e.text(1e-2, 1e-12, 'Detectable\nregime', fontsize=10, color='green', alpha=0.7)

ax_e.set_xlabel('$\\Delta k / k_0$ (tunneling modulation)', fontsize=11)
ax_e.set_ylabel('Required $E_0$ (V/m)', fontsize=11)
ax_e.set_title('(e) Field Strength Required for Given Effect Size', fontsize=12, fontweight='bold')
ax_e.set_xlim(1e-60, 1)
ax_e.set_ylim(1e-20, 1e12)
ax_e.legend(fontsize=9)
ax_e.grid(True, alpha=0.3, which='both')

# --- Panel F: The honest conclusion diagram ---
ax_f = fig.add_subplot(gs[2, 1])
ax_f.axis('off')

conclusion_text = """
HONEST CONCLUSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEAD (by ~35-50 orders of magnitude):
  ✗ Direct EM → proton tunneling
  ✗ Ising criticality amplification
  ✗ Stochastic resonance
  ✗ Temporal integration
  ✗ All of the above combined

ALIVE (model-independent):
  ✓ Spectral fingerprint predictions
  ✓ Coherence factor as discriminator
  ✓ Falsification criteria (shielding, E₀², 
    base ordering)

THE GAP MEANS:
  If astronomical-biological correlations are 
  real, the coupling mechanism is NOT:
    • Electromagnetic
    • Quantum tunneling
    • Any known linear/nonlinear response

  It must be something QUALITATIVELY different:
    • Gravitational (Section VII piezo-vacuum)?
    • Informational (not energetic)?
    • A mechanism we haven't identified?

  OR: the correlations are not real.
"""

ax_f.text(0.05, 0.95, conclusion_text, transform=ax_f.transAxes,
          fontsize=11, verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# --- Panel G: What DOES work at planetary field strengths? ---
ax_g = fig.add_subplot(gs[3, 0])
ax_g.axis('off')

what_works = """
WHAT COULD BRIDGE THE GAP?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. NON-EM COUPLING (Section VII)
   Gravitational piezo-luminescence couples 
   planetary strain directly to vacuum EM
   → bypasses magnetospheric attenuation
   → requires E₀ ~ 10⁻² V/m at target
   → testable with gravitational wave detectors

2. RESONANT BIOLOGICAL ANTENNA
   If microtubules/membranes act as coherent
   EM collectors over macroscopic volumes
   → effective E₀ amplified by √N_antenna
   → N ~ 10⁴⁰ is impossible physically

3. INFORMATION-THEORETIC COUPLING
   The bias doesn't carry energy—it carries 
   timing information that a critical system
   reads as a phase reference
   → requires new physics of measurement
   → most speculative, least constrained

4. THE NULL HYPOTHESIS
   Astronomical-biological correlations are
   coincidental, artifactual, or mediated by
   known non-EM pathways (cosmic rays, UV,
   seasonal temperature, etc.)
   → most parsimonious
   → should be default assumption
"""

ax_g.text(0.05, 0.95, what_works, transform=ax_g.transAxes,
          fontsize=10, verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='#f0f8ff', alpha=0.9))

# --- Panel H: Minimum coupling strength map ---
ax_h = fig.add_subplot(gs[3, 1])

# For each mechanism, what coupling strength makes it work?
mechanisms_needed = {
    'Direct EM\n(current model)':          1e-50,
    'EM + Ising\ncriticality':             1e-43,
    'EM + Ising +\ntemporal (30yr)':       1e-39,
    'Unknown\namplifier\n($10^{35}\\times$)': 1e-15,
    'Required for\n1% biological\neffect':  1e-2,
}

x_mech = np.arange(len(mechanisms_needed))
y_mech = [np.log10(v) for v in mechanisms_needed.values()]
colors_mech = ['#264653', '#457B9D', '#2A9D8F', '#E9C46A', '#E63946']

bars_h = ax_h.bar(x_mech, y_mech, color=colors_mech, edgecolor='black',
                  linewidth=0.5, alpha=0.85, width=0.6)
ax_h.set_xticks(x_mech)
ax_h.set_xticklabels(list(mechanisms_needed.keys()), fontsize=8, ha='center')
ax_h.set_ylabel('log$_{10}$(effective coupling strength)', fontsize=11)
ax_h.set_title('(h) Known vs Required Coupling Strengths', fontsize=12, fontweight='bold')
ax_h.axhline(y=-4, color='orange', linestyle='--', alpha=0.7,
             label='Detection threshold ($10^{-4}$)')
ax_h.legend(fontsize=9)
ax_h.set_ylim(-55, 0)
ax_h.grid(True, alpha=0.2, axis='y')

fig.suptitle('QINCRS Criticality Bridge: Can Phase Boundaries Amplify Planetary Bias?',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig('/home/claude/fig8_criticality_bridge.pdf', dpi=300, bbox_inches='tight')
fig.savefig('/home/claude/fig8_criticality_bridge.png', dpi=300, bbox_inches='tight')
print("  -> fig8_criticality_bridge saved")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 75)
print("FINAL VERDICT")
print("=" * 75)
print(f"""
THE CALCULATION IS COMPLETE.

Starting point:
  Solar-planetary EM fields at Earth's surface: E_0 ~ 10^-16 V/m
  Proton tunneling modulation: dk/k0 ~ 10^-50

We tested every known amplification mechanism:
  Ising criticality (N=10^11):     amplification ~ 10^7
  Temporal integration (30 yr):     amplification ~ 10^4
  Stochastic resonance:            amplification ~ 10^0 (none)
  All combined:                     amplification ~ 10^11

  Required amplification:           10^48
  Available amplification:          10^11
  UNBRIDGEABLE GAP:                 ~35-37 orders of magnitude

CONCLUSION:
  No known physical mechanism—including criticality, stochastic
  resonance, temporal integration, or any combination thereof—
  can amplify the planetary electromagnetic bias to biologically
  detectable levels.

  The planetary coherence hypothesis, in its electromagnetic form,
  is CLEANLY FALSIFIED by its own quantitative framework.

  This is not a failure. This is a result.

  The spectral fingerprint predictions remain as model-independent
  tests for ANY proposed coupling mechanism. The falsification
  criteria remain valid. And the ~35 order-of-magnitude gap
  provides a precise constraint that any future theory must satisfy.

  If astronomical-biological correlations are real, the coupling
  is not electromagnetic. The search space has been narrowed
  from "anything" to "something that provides 10^35 amplification
  over known physics."

  That constraint is the most valuable output of this calculation.
""")

print("=" * 75)
print("Script complete. All figures saved.")
print("=" * 75)


if __name__ == '__main__':
    pass

# Run main
main()
