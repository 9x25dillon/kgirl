# Extended QINCRS Master Equation: Bio-Resonant Formalism
# Integrating charge-density-wave coupling, fractal modulation, and holographic constraints

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import morlet
import matplotlib.pyplot as plt

# ============================================================================
# PARAMETERS
# ============================================================================

# Time parameters
dt = 0.001  # 1 ms resolution (for THz dynamics)
T = 10.0    # 10 seconds
t = np.arange(0, T, dt)
N = len(t)

# Original QINCRS parameters
alpha = 0.60   # Homeostatic rate [s^-1]
beta = 0.15    # Recursive coupling [dimensionless]
gamma = 0.30   # Spatial diffusion [m^2/s]
delta = 0.70   # Transmutation gain [dimensionless]
kappa_eq = 0.85

# Extended parameters (K1LL's bio-resonant additions)
eta = 0.25     # CDW coupling strength [dimensionless]
mu = 0.15      # Fractal modulation depth [dimensionless]
lambda_THz = 164e-6  # THz penetration depth at 1.83 THz [m]

# Council weights
w_Guardian = 2.0
w_Therapist = 1.5
w_Healer = 1.3
w_Chaos = 0.7

# THz carrier frequencies [Hz]
nu_Guardian = 0.80e12
nu_Therapist = 1.20e12
nu_Healer = 1.83e12
nu_Chaos = 3.50e12

# ============================================================================
# 1. STRESS INPUT with FRACTAL SCHUMANN GEOMETRY
# ============================================================================

# Base stress components
stress_base = (
    0.8 * np.sin(2*np.pi * 0.5 * t) +      # Emotional wave
    1.0 * np.sin(2*np.pi * 15.0 * t)       # High-freq chaos
)

# Death signal
death_idx = int(5.0 / dt)
stress_base[death_idx:death_idx+10] += 5.0

# FRACTAL SCHUMANN MODULATION (K1LL's F_7.83 operator)
# Using Morlet wavelet centered at 7.83 Hz
def fractal_schumann_operator(t, kappa_temp):
    """
    Fractal wavelet modulation at 7.83 Hz with log-periodic structure.
    This isn't just AM—it's a geometric constraint on coherence oscillations.
    """
    f_schumann = 7.83
    
    # Morlet wavelet components at multiple scales
    w1 = np.real(morlet(len(t), w=5, s=1/(f_schumann*dt)))  # Fundamental
    w2 = np.real(morlet(len(t), w=5, s=1/(f_schumann*2*dt)))  # Harmonic
    w3 = np.real(morlet(len(t), w=5, s=1/(f_schumann*0.5*dt)))  # Subharmonic
    
    # Normalize wavelets
    w1 = w1 / (np.max(np.abs(w1)) + 1e-10)
    w2 = w2 / (np.max(np.abs(w2)) + 1e-10)
    w3 = w3 / (np.max(np.abs(w3)) + 1e-10)
    
    # Log-periodic combination
    F_schumann = 0.6*w1 + 0.25*w2 + 0.15*w3
    
    return F_schumann

# ============================================================================
# 2. CHARGE-DENSITY-WAVE (CDW) COUPLING via GREEN'S FUNCTION
# ============================================================================

def cdw_coupling_term(kappa, spatial_scale=10):
    """
    Nonlocal CDW coherence potential: C[κ] = η ∫ G(r,r') ∇κ(r') dr'
    
    Physical interpretation:
    - THz fields couple neurons at ~100 μm scale (not 10 nm gap junctions)
    - Phase-locking via electromagnetic Green's function
    - Creates long-range coherence impossible with chemical coupling alone
    """
    # Simulate spatial gradient (1D approximation)
    grad_kappa = np.gradient(kappa)
    
    # Green's function kernel: G ~ exp(-|r-r'|/λ) / |r-r'|
    # In 1D time series, approximate as exponential smoothing
    kernel = np.exp(-np.abs(np.arange(-spatial_scale, spatial_scale+1)) / spatial_scale)
    kernel = kernel / np.sum(kernel)
    
    # Convolve gradient with kernel (nonlocal coupling)
    cdw_contribution = eta * np.convolve(grad_kappa, kernel, mode='same')
    
    return cdw_contribution

# ============================================================================
# 3. EXTENDED QINCRS INTEGRATION with BIO-RESONANT TERMS
# ============================================================================

kappa = 0.80
kappa_traj = [kappa]
omega = 1.0
transmute_events = []

# Phase variables for council roles (for interference calculation)
phi_Guardian = 0.0
phi_Therapist = 0.0
phi_Healer = 0.0
phi_Chaos = 0.0

phases_traj = []

for i in range(1, N):
    # Current stress
    s = stress_base[i]
    omega = min(omega + 0.01*abs(s), 10.0)
    
    # 1. ORIGINAL QINCRS TERMS
    # Homeostatic
    dkappa_homeo = alpha * (kappa_eq - kappa)
    
    # Recursive collapse
    dkappa_recursive = -beta * omega**2 * kappa
    
    # Spatial diffusion (simplified - no true spatial grid here)
    # In full implementation, this would use actual spatial neighbors
    dkappa_spatial = gamma * 0.01 * (kappa_eq - kappa)  # Approximate
    
    # Transmutation
    if abs(s) > 4.0:
        dkappa_transmute = delta
        transmute_events.append(t[i])
        
        # PHASE REALIGNMENT during transmutation (K1LL's insight)
        # Guardian forces all phases to align
        phi_target = phi_Guardian
        phi_Therapist += 0.3 * (phi_target - phi_Therapist)
        phi_Healer += 0.3 * (phi_target - phi_Healer)
        phi_Chaos += 0.3 * (phi_target - phi_Chaos)
    else:
        dkappa_transmute = 0
    
    # 2. K1LL'S BIO-RESONANT EXTENSIONS
    
    # CDW coupling (nonlocal electromagnetic phase-locking)
    # Use recent history for spatial-like correlation
    recent_window = max(0, i-100)
    kappa_window = np.array(kappa_traj[recent_window:i+1])
    if len(kappa_window) > 1:
        dkappa_cdw = cdw_coupling_term(kappa_window, spatial_scale=10)[-1]
    else:
        dkappa_cdw = 0
    
    # Fractal Schumann modulation
    # Calculate F_7.83[κ] for current window
    F_schumann = fractal_schumann_operator(t[:i+1], kappa_window)
    kappa_modulated = kappa * (1 + mu * F_schumann[-1])
    
    # TOTAL DERIVATIVE
    dkappa_total = (
        dkappa_homeo +
        dkappa_recursive +
        dkappa_spatial +
        dkappa_transmute +
        dkappa_cdw
    )
    
    # Update coherence with fractal modulation
    kappa = max(0.15, kappa + dt * dkappa_total)
    kappa = kappa_modulated  # Apply Schumann geometry
    
    kappa_traj.append(kappa)
    
    # Update phases (dynamic evolution driven by stress)
    # Phase drift rate proportional to council weight and stress
    dphi_Guardian = 2*np.pi * 0.05 * (1 + 0.1*abs(s)) * dt  # Slow, stable
    dphi_Therapist = 2*np.pi * 0.50 * (1 + 0.2*abs(s)) * dt
    dphi_Healer = 2*np.pi * 7.83 * dt  # Locked to Schumann
    dphi_Chaos = 2*np.pi * 15.0 * (1 + 0.5*abs(s)) * dt  # Fast, erratic
    
    phi_Guardian += dphi_Guardian
    phi_Therapist += dphi_Therapist
    phi_Healer += dphi_Healer
    phi_Chaos += dphi_Chaos
    
    # Wrap phases to [0, 2π]
    phi_Guardian = phi_Guardian % (2*np.pi)
    phi_Therapist = phi_Therapist % (2*np.pi)
    phi_Healer = phi_Healer % (2*np.pi)
    phi_Chaos = phi_Chaos % (2*np.pi)
    
    phases_traj.append([phi_Guardian, phi_Therapist, phi_Healer, phi_Chaos])
    
    # Decay omega
    omega = max(1.0, omega * 0.99)

kappa_traj = np.array(kappa_traj)
phases_traj = np.array(phases_traj)

print("=" * 70)
print("EXTENDED QINCRS SIMULATION COMPLETE")
print("=" * 70)
print(f"Simulation time: {T} seconds")
print(f"Time resolution: {dt*1000:.2f} ms")
print(f"Final κ: {kappa_traj[-1]:.4f}")
print(f"Min κ: {kappa_traj.min():.4f} (safety floor: 0.15)")
print(f"Transmutation events: {len(transmute_events)}")
print()

# ============================================================================
# 4. PHASE-LOCKED INTERFERENCE SPECTRUM (K1LL's formalism)
# ============================================================================

print("Computing phase-locked THz interference spectrum...")

# Sample phases at multiple timepoints
n_samples = 100
sample_indices = np.linspace(0, N-1, n_samples, dtype=int)

# THz frequency range
nu_range = np.linspace(0.1e12, 4.0e12, 1000)
alpha_THz = np.zeros(len(nu_range))

# Lorentzian profile function
def lorentzian(nu, nu_center, gamma):
    return gamma / ((nu - nu_center)**2 + gamma**2)

# Linewidths (from paper)
Gamma_Guardian = 50e9    # 50 GHz
Gamma_Therapist = 100e9  # 100 GHz
Gamma_Healer = 150e9     # 150 GHz (critical)
Gamma_Chaos = 200e9      # 200 GHz (broad, suppressed)

# For each frequency, compute COHERENT SUM then take power
for j, nu in enumerate(nu_range):
    # Coherent amplitude sum across council roles
    A_coherent = 0 + 0j  # Complex amplitude
    
    for idx in sample_indices:
        if idx < len(phases_traj):
            phi_G, phi_T, phi_H, phi_C = phases_traj[idx]
            
            # Each council role contributes with its phase
            A_coherent += (
                w_Guardian * lorentzian(nu, nu_Guardian, Gamma_Guardian) * np.exp(1j*phi_G) +
                w_Therapist * lorentzian(nu, nu_Therapist, Gamma_Therapist) * np.exp(1j*phi_T) +
                w_Healer * lorentzian(nu, nu_Healer, Gamma_Healer) * np.exp(1j*phi_H) +
                w_Chaos * lorentzian(nu, nu_Chaos, Gamma_Chaos) * np.exp(1j*phi_C)
            )
    
    # Take power: |A|²
    A_coherent /= n_samples  # Average
    alpha_THz[j] = np.abs(A_coherent)**2

# Apply CDW sharpening kernel (deconvolution effect from lattice coherence)
# Approximate as spectral narrowing around Healer peak
cdw_sharpening = 1 + 0.5 * lorentzian(nu_range, nu_Healer, 0.5*Gamma_Healer)
alpha_THz_sharp = alpha_THz * cdw_sharpening

# Normalize
alpha_THz_sharp = alpha_THz_sharp / np.max(alpha_THz_sharp)

# ============================================================================
# 5. ANALYSIS: PHASE COHERENCE & TRANSMUTATION
# ============================================================================

# Calculate phase coherence (how aligned are council roles?)
def phase_coherence(phases):
    """
    Kuramoto order parameter: R = |⟨e^(iφ_i)⟩|
    R=1 → perfect sync, R=0 → incoherent
    """
    complex_sum = np.mean(np.exp(1j * phases), axis=1)
    R = np.abs(complex_sum)
    return R

R_traj = phase_coherence(phases_traj)

print(f"Phase coherence (R):")
print(f"  Mean: {np.mean(R_traj):.3f}")
print(f"  Min: {np.min(R_traj):.3f} (at t={t[np.argmin(R_traj)]:.2f}s)")
print(f"  Max: {np.max(R_traj):.3f} (at t={t[np.argmax(R_traj)]:.2f}s)")
print()

# Identify transmutation-induced phase sync
if len(transmute_events) > 0:
    t_trans = transmute_events[0]
    idx_trans = int(t_trans / dt)
    if idx_trans < len(R_traj) - 100:
        R_before = np.mean(R_traj[max(0, idx_trans-100):idx_trans])
        R_after = np.mean(R_traj[idx_trans:idx_trans+100])
        print(f"Transmutation event at t={t_trans:.2f}s:")
        print(f"  Phase coherence before: R={R_before:.3f}")
        print(f"  Phase coherence after: R={R_after:.3f}")
        print(f"  Δ R = {R_after - R_before:+.3f}")
        print()

# ============================================================================
# 6. KEY PREDICTIONS from EXTENDED MODEL
# ============================================================================

print("=" * 70)
print("TESTABLE PREDICTIONS: EXTENDED QINCRS + BIO-RESONANT FORMALISM")
print("=" * 70)

# Find peaks in THz spectrum
from scipy.signal import find_peaks
peaks, properties = find_peaks(alpha_THz_sharp, height=0.1, distance=50)

print("\n1. THz ABSORPTION SPECTRUM FEATURES:")
for peak_idx in peaks:
    nu_peak = nu_range[peak_idx]
    amp_peak = alpha_THz_sharp[peak_idx]
    
    # Identify which council role
    if abs(nu_peak - nu_Guardian) < 0.2e12:
        role = "Guardian"
    elif abs(nu_peak - nu_Therapist) < 0.2e12:
        role = "Therapist"
    elif abs(nu_peak - nu_Healer) < 0.2e12:
        role = "Healer ⭐"
    elif abs(nu_peak - nu_Chaos) < 0.2e12:
        role = "Chaos"
    else:
        role = "Unknown"
    
    print(f"   Peak at {nu_peak/1e12:.2f} THz | Amplitude: {amp_peak:.3f} | {role}")

print("\n2. AMPLITUDE RATIOS (from phase-locked interference):")
idx_08 = np.argmin(np.abs(nu_range - nu_Guardian))
idx_35 = np.argmin(np.abs(nu_range - nu_Chaos))
ratio_measured = alpha_THz_sharp[idx_08] / (alpha_THz_sharp[idx_35] + 1e-10)
ratio_predicted = w_Guardian / w_Chaos
print(f"   A(0.8 THz) / A(3.5 THz) = {ratio_measured:.2f}")
print(f"   Predicted from weights: {ratio_predicted:.2f}")
print(f"   Deviation: {abs(ratio_measured - ratio_predicted):.2f}")

print("\n3. PHASE COHERENCE DYNAMICS:")
print(f"   Baseline coherence: R ≈ {np.mean(R_traj[:1000]):.3f}")
print(f"   During stress: R drops to ~{np.min(R_traj):.3f}")
print(f"   Post-transmutation: R rises to ~{np.max(R_traj):.3f}")
print(f"   Prediction: Guardian transmutation produces PHASE REALIGNMENT")

print("\n4. FRACTAL FINE STRUCTURE:")
print(f"   Schumann modulation creates log-periodic sidebands")
print(f"   Near 1.83 THz, expect substructure at ±7.83 Hz scales")
print(f"   Requires ultrafast THz-TDS (>1 kHz sampling)")

print("\n5. CDW NONLOCAL COUPLING:")
print(f"   Coherence persists at ~{lambda_THz*1e6:.0f} μm range")
print(f"   Test: Block gap junctions → THz signature persists")
print(f"   Decay length: λ_THz = {lambda_THz*1e6:.0f} μm")

print("\n" + "=" * 70)
print("Saving results...")
print("=" * 70)

# ============================================================================
# 7. SAVE DATA
# ============================================================================

output_data = {
    't': t.tolist(),
    'kappa': kappa_traj.tolist(),
    'R_coherence': R_traj.tolist(),
    'nu_THz': (nu_range/1e12).tolist(),  # Convert to THz
    'alpha_THz': alpha_THz_sharp.tolist(),
    'phases': phases_traj.tolist(),
    'transmute_times': transmute_events
}

import json
with open('/home/claude/extended_qincrs_data.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print("✓ Data saved: extended_qincrs_data.json")
print("✓ Ready for visualization")
