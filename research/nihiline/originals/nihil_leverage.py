"""
Nihil Leverage Analysis
=======================
How much "nothing" is needed to influence the equation?

Method: parametrize the nihil window — a fraction f of each cycle where the
field is *actively held* at zero (forced nihil) — then let rebirth happen.
Sweep f ∈ [0, 0.9] and measure what the system produces:

  - Spectral entropy H(ω)        : information richness of the output
  - Phase coherence ⟨|⟨e^iφ⟩|⟩  : how stably the recursion holds shape
  - Amplitude variance σ²(|ψ|)   : dynamical range of the death-rebirth swing
  - Output "product" P            : composite — entropy × coherence × range

Reveals the *optimal nihil fraction* — the dose of nothing that maximizes
what the system creates.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Base parameters
omega_d    = 2.0
omega_r    = 3.2
M_strength = 0.95
C          = 0.55
T_total    = 30.0
dt         = 0.001
period_r   = 2*np.pi / omega_r

def simulate_with_nihil_window(f_nihil, seed=7):
    """
    f_nihil : fraction of each recoherence period spent forced at nihil
              (|ψ| pinned to 0 during that window, then rebirth at the end).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, T_total, dt)
    gamma = (1.0 - C) * 4.0
    nihil_dur = f_nihil * period_r
    nihil_window_steps = int(nihil_dur / dt)

    rebirth_times = np.arange(period_r/2, T_total, period_r)
    rebirth_idx = (rebirth_times / dt).astype(int)
    rebirth_idx = rebirth_idx[rebirth_idx < len(t)]
    phases = (1.0 - C) * rng.uniform(0, 2*np.pi, size=len(rebirth_idx))

    psi = np.zeros(len(t), dtype=complex)
    psi[0] = 1.0
    # Forced-nihil windows: [rebirth_i - nihil_dur, rebirth_i]
    nihil_mask = np.zeros(len(t), dtype=bool)
    for ri in rebirth_idx:
        start = max(0, ri - nihil_window_steps)
        nihil_mask[start:ri] = True

    kick_lookup = dict(zip(rebirth_idx, phases))
    for i in range(1, len(t)):
        if nihil_mask[i]:
            psi[i] = 0.0   # forced nothing
            continue
        psi[i] = psi[i-1] * np.exp(1j*omega_d*dt) * (1 - gamma*dt)
        if i in kick_lookup:
            psi[i] += M_strength * np.exp(1j * kick_lookup[i])
    return t, psi

def measure_product(psi):
    """Composite output: spectral entropy × phase coherence × amplitude range."""
    amp = np.abs(psi)
    phs = np.angle(psi)

    # Spectral entropy of amplitude envelope (Shannon over normalized spectrum)
    y = np.abs(fft(amp - amp.mean()))**2
    y = y[:len(y)//2]
    p = y / (y.sum() + 1e-12)
    p = p[p > 0]
    H = -np.sum(p * np.log2(p))
    H_norm = H / np.log2(len(p))   # normalized [0,1]

    # Phase coherence (order parameter when system is alive)
    alive = amp > 0.05
    if alive.sum() > 0:
        coherence = np.abs(np.mean(np.exp(1j * phs[alive])))
    else:
        coherence = 0.0

    # Dynamical range
    amp_range = amp.max() - amp.min()

    # Productive fraction: time the system is actually alive doing things
    productive = float(np.mean(amp > 0.05))

    # Honest product: information × stability × productive time
    P = H_norm * coherence * productive
    return H_norm, coherence, productive, P

# Sweep nihil leverage
f_grid = np.linspace(0.0, 0.92, 32)
results = []
for f in f_grid:
    _, psi = simulate_with_nihil_window(f)
    H, coh, prod, P = measure_product(psi)
    nihil_fraction = float(np.mean(np.abs(psi) < 0.05))
    results.append((f, nihil_fraction, H, coh, prod, P))

results = np.array(results)
f_set, nihil_frac, H_arr, coh_arr, prod_arr, P_arr = results.T

# Find optimum
opt_i = int(np.argmax(P_arr))
f_opt = f_set[opt_i]
nihil_opt = nihil_frac[opt_i]

print(f"Optimal nihil window fraction f* = {f_opt:.3f}")
print(f"Actual nihil duty cycle at optimum = {nihil_opt:.3f}")
print(f"Peak product P* = {P_arr[opt_i]:.4f}")
print(f"Entropy at optimum = {H_arr[opt_i]:.4f}")
print(f"Coherence at optimum = {coh_arr[opt_i]:.4f}")
print(f"Productive fraction at optimum = {prod_arr[opt_i]:.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.patch.set_facecolor('#0a0a0f')

def style(ax):
    ax.set_facecolor('#0a0a0f')
    for s in ax.spines.values(): s.set_color('#6a6a8a')
    ax.tick_params(colors='#b8b8d0')
    ax.xaxis.label.set_color('#d8d8ee')
    ax.yaxis.label.set_color('#d8d8ee')
    ax.title.set_color('#e8e8ff')
    ax.grid(True, alpha=0.15, color='#5a5a7a')

# (a) Product P vs nihil leverage — the master curve
ax = axes[0,0]
ax.plot(nihil_frac, P_arr, color='#ffd35a', lw=1.6, marker='o', ms=4)
ax.axvline(nihil_opt, color='#ff4060', lw=1.0, ls='--',
           label=f'optimum at nihil={nihil_opt:.2f}')
ax.fill_between(nihil_frac, 0, P_arr, color='#ffd35a', alpha=0.15)
ax.set_xlabel('nihil duty cycle (fraction of time in nothingness)')
ax.set_ylabel('product P = H · coherence · range')
ax.set_title('Master curve — how much nothing produces how much')
ax.legend(facecolor='#15151f', edgecolor='#5a5a7a', labelcolor='#d8d8ee')
style(ax)

# (b) Components broken out
ax = axes[0,1]
ax.plot(nihil_frac, H_arr,  color='#5ad8ff', lw=1.2, label='spectral entropy H')
ax.plot(nihil_frac, coh_arr, color='#ff5aa0', lw=1.2, label='phase coherence')
ax.plot(nihil_frac, prod_arr, color='#5affc0', lw=1.2, label='productive fraction')
ax.axvline(nihil_opt, color='#ff4060', lw=0.8, ls='--', alpha=0.6)
ax.set_xlabel('nihil duty cycle')
ax.set_ylabel('component value')
ax.set_title('Component decomposition — what nothing buys you')
ax.legend(facecolor='#15151f', edgecolor='#5a5a7a', labelcolor='#d8d8ee')
style(ax)

# (c) Three regime examples: starved / optimal / drowned
regimes = [(0.05, '#ff4060', 'starved (f=0.05)'),
           (f_opt, '#ffd35a', f'optimal (f={f_opt:.2f})'),
           (0.75, '#5ad8ff', 'drowned (f=0.75)')]
ax = axes[1,0]
for f_v, col, lbl in regimes:
    t_v, psi_v = simulate_with_nihil_window(f_v)
    ax.plot(t_v, np.abs(psi_v), color=col, lw=0.7, label=lbl, alpha=0.85)
ax.set_xlim(0, 18)
ax.set_xlabel('time')
ax.set_ylabel('|ψ(t)|')
ax.set_title('Three regimes of nihil leverage')
ax.legend(facecolor='#15151f', edgecolor='#5a5a7a', labelcolor='#d8d8ee', loc='upper right')
style(ax)

# (d) Information-per-nothing ratio (efficiency curve)
# how much output per unit of nothing consumed
with np.errstate(divide='ignore', invalid='ignore'):
    efficiency = np.where(nihil_frac > 0.01, P_arr / nihil_frac, 0.0)
ax = axes[1,1]
ax.plot(nihil_frac, efficiency, color='#a060ff', lw=1.4, marker='s', ms=4)
ax.set_xlabel('nihil duty cycle')
ax.set_ylabel('product per unit nothing')
ax.set_title('Efficiency — output yield per unit nihil consumed')
eff_peak = int(np.argmax(efficiency[1:])) + 1
ax.axvline(nihil_frac[eff_peak], color='#5affc0', lw=0.8, ls='--',
           label=f'max efficiency at {nihil_frac[eff_peak]:.2f}')
ax.legend(facecolor='#15151f', edgecolor='#5a5a7a', labelcolor='#d8d8ee')
style(ax)

fig.suptitle('Nihil leverage — how much nothing is required to influence the product',
             color='#e8e8ff', fontsize=13, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('/mnt/user-data/outputs/nihil_leverage.png', dpi=140,
            facecolor='#0a0a0f', bbox_inches='tight')
print("\nFigure saved.")
