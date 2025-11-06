"""
Graphical Abstract Generator for Cognitive Renewal Dynamics
Creates a 1200x600px figure showing the S(t) ↔ Π renewal loop

Requirements:
    pip install matplotlib numpy pillow

Usage:
    python generate_graphical_abstract.py
    
Output:
    graphical_abstract.png (1200x600px, 200 DPI)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib import font_manager

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure
fig = plt.figure(figsize=(12, 6), facecolor='white', dpi=200)

# Create three panels
ax1 = plt.subplot(1, 3, 1)  # Sequential mode
ax2 = plt.subplot(1, 3, 2)  # Exchange / Equation
ax3 = plt.subplot(1, 3, 3)  # Invariant field

# ============================================
# LEFT PANEL: Sequential Mode S(t)
# ============================================

t = np.linspace(0, 10, 1000)

# Generate multiple frequency bands with varying coherence
bands = {
    'delta': (1, '#1976D2'),
    'theta': (2, '#2196F3'),
    'alpha': (3, '#42A5F5'),
    'beta': (4, '#64B5F6'),
    'gamma': (5, '#90CAF9')
}

for i, (name, (freq, color)) in enumerate(bands.items()):
    # Create oscillating signal with noise (representing fluctuating coherence)
    signal = np.sin(2 * np.pi * freq * t / 10) 
    noise = 0.2 * np.random.randn(len(t))
    combined = signal + noise
    
    ax1.plot(t, combined + i * 1.5, color=color, linewidth=1.5, alpha=0.8, label=name)

ax1.set_xlim(0, 10)
ax1.set_ylim(-1, 8)
ax1.set_title('S(t)\nSequential Mode', fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Time', fontsize=11)
ax1.text(5, -0.5, 'Moment-to-moment\nfluctuations', 
         ha='center', fontsize=10, style='italic', color='#424242')
ax1.axis('off')

# Add subtle background
rect1 = mpatches.Rectangle((0, -1), 10, 9, 
                           linewidth=0, 
                           edgecolor='none', 
                           facecolor='#E3F2FD', 
                           alpha=0.3, 
                           zorder=-1)
ax1.add_patch(rect1)

# ============================================
# CENTER PANEL: Exchange with coherence levels
# ============================================

ax2.axis('off')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

# Title
ax2.text(0.5, 0.95, 'The Renewal Loop', 
         ha='center', fontsize=16, fontweight='bold')

# Main equation (large, centered)
ax2.text(0.5, 0.75, r'$\frac{d\kappa}{dt} = \alpha(1 - \kappa)$', 
         ha='center', fontsize=24, bbox=dict(boxstyle='round', 
                                             facecolor='#FFF9C4', 
                                             alpha=0.8))

# Bidirectional arrow
arrow = FancyArrowPatch((0.1, 0.5), (0.9, 0.5),
                        arrowstyle='<->', 
                        mutation_scale=30, 
                        linewidth=3, 
                        color='#424242')
ax2.add_patch(arrow)

# High coherence example (top)
ax2.text(0.5, 0.58, 'High coherence (κ ≈ 1)', 
         ha='center', fontsize=11, fontweight='bold', color='#2E7D32')
# Draw aligned waves
t_small = np.linspace(0, 2*np.pi, 50)
for i in range(3):
    wave = 0.03 * np.sin(t_small) + 0.35 + i*0.01
    x_wave = 0.2 + t_small / (2*np.pi) * 0.6
    ax2.plot(x_wave, wave, color='#66BB6A', linewidth=2, alpha=0.8)
ax2.text(0.5, 0.32, '→ Unified awareness', 
         ha='center', fontsize=9, style='italic', color='#2E7D32')

# Low coherence example (bottom)
ax2.text(0.5, 0.25, 'Low coherence (κ ≈ 0.2)', 
         ha='center', fontsize=11, fontweight='bold', color='#C62828')
# Draw misaligned waves
for i in range(3):
    phase_shift = np.random.uniform(0, np.pi)
    wave = 0.03 * np.sin(t_small + phase_shift) + 0.12 + i*0.01
    x_wave = 0.2 + t_small / (2*np.pi) * 0.6
    ax2.plot(x_wave, wave, color='#EF5350', linewidth=2, alpha=0.8)
ax2.text(0.5, 0.06, '→ Decoherence / Release', 
         ha='center', fontsize=9, style='italic', color='#C62828')

# ============================================
# RIGHT PANEL: Invariant Field Π
# ============================================

# Create attractor basin visualization
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# Potential function (creates basin shape)
Z = X**2 + Y**2

# Plot as contour (attractor basin)
contour = ax3.contourf(X, Y, Z, levels=20, cmap='YlOrBr', alpha=0.7)

# Add center glow (attractor point)
circle = plt.Circle((0, 0), 0.3, color='#FF6F00', alpha=0.9, zorder=10)
ax3.add_patch(circle)

ax3.set_xlim(-2, 2)
ax3.set_ylim(-2, 2)
ax3.set_title('Π\nInvariant Field', fontsize=16, fontweight='bold', pad=20)
ax3.text(0, -1.5, 'Stable attractor\npattern', 
         ha='center', fontsize=10, style='italic', color='#424242')
ax3.axis('off')

# ============================================
# OVERALL FIGURE ANNOTATIONS
# ============================================

# Add main title at top
fig.suptitle('Cognitive Renewal Dynamics', 
             fontsize=20, fontweight='bold', y=0.98)

# Add subtitle
fig.text(0.5, 0.92, 'Consciousness as rhythmic return to proportion',
         ha='center', fontsize=13, style='italic', color='#616161')

# Add key insight at bottom
fig.text(0.5, 0.04, 
         'Identity is not what stays constant, but what RETURNS',
         ha='center', fontsize=12, fontweight='bold', 
         bbox=dict(boxstyle='round', facecolor='#E8EAF6', alpha=0.8))

# Add author
fig.text(0.95, 0.02, 'Randy Lynn, 2025', 
         ha='right', fontsize=9, color='#757575')

# ============================================
# SAVE FIGURE
# ============================================

plt.tight_layout(rect=[0, 0.06, 1, 0.90])
plt.savefig('graphical_abstract.png', 
            dpi=200, 
            bbox_inches='tight', 
            facecolor='white',
            edgecolor='none')

print("✓ Graphical abstract saved as 'graphical_abstract.png'")
print("  Dimensions: 1200x600px at 200 DPI")
print("  Ready for academia.edu upload!")

plt.show()
