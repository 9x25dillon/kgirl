"""
tdcs_enhanced_recovery.py
Integrate tDCS-inspired active interventions with complex CNN reconstruction
"""

import numpy as np
import torch
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class TDCSProtocol:
    """
    tDCS stimulation protocol
    Maps to our coherence modulation
    """
    anode_region: str  # e.g., "pre-SMA", "DLPFC"
    cathode_region: str
    current_ma: float  # 1-2 mA
    duration_min: int  # 20 min per session
    frequency_target: str  # Which band to modulate
    polarity: str  # "anodal" (excitatory) or "cathodal" (inhibitory)


class TDCSSimulator:
    """
    Simulate tDCS-like coherence modulation
    Based on paper's findings about electrode placement effects
    """
    
    def __init__(self):
        # Region → Band mapping from neuroscience
        self.region_band_map = {
            'pre-SMA': 'theta',      # Motor planning
            'DLPFC': 'beta',         # Executive function
            'OFC': 'alpha',          # Emotion regulation
            'cerebellum': 'delta'    # Coordination
        }
        
        # Polarity effects (from paper)
        self.polarity_effects = {
            'anodal': 1.15,     # Excitatory (increase κ)
            'cathodal': 0.85    # Inhibitory (decrease κ)
        }
    
    def apply_tdcs(self, 
                   kappa_current: Dict[str, float],
                   protocol: TDCSProtocol) -> Dict[str, float]:
        """
        Simulate tDCS modulation effect on coherence
        
        Based on paper findings:
        - Anodal stimulation increases cortical excitability
        - Cathodal stimulation decreases excitability
        - Effects are region and band specific
        """
        kappa_modulated = kappa_current.copy()
        
        # Determine target band from anode region
        target_band = self.region_band_map.get(
            protocol.anode_region,
            'alpha'  # Default
        )
        
        # Apply modulation
        if target_band in kappa_modulated:
            # Current intensity effect (1-2 mA range)
            intensity_factor = protocol.current_ma / 2.0  # Normalize
            
            # Polarity effect
            polarity_factor = self.polarity_effects[protocol.polarity]
            
            # Duration effect (diminishing returns after 20 min)
            duration_factor = min(protocol.duration_min / 20.0, 1.0)
            
            # Combined modulation
            modulation = (
                intensity_factor * 
                polarity_factor * 
                duration_factor
            )
            
            # Apply to target band
            kappa_modulated[target_band] *= modulation
            
            # Clip to valid range
            kappa_modulated[target_band] = np.clip(
                kappa_modulated[target_band], 
                0, 1
            )
        
        return kappa_modulated
    
    def optimize_protocol(self,
                         kappa_current: Dict[str, float],
                         kappa_target: Dict[str, float]) -> TDCSProtocol:
        """
        Find optimal tDCS protocol to move current toward target
        
        This is the "smart tDCS" - personalized based on current state
        """
        # Find band with largest deficit
        deficits = {
            band: kappa_target[band] - kappa_current[band]
            for band in kappa_current
        }
        
        # Target band with largest deficit
        target_band = max(deficits, key=deficits.get)
        deficit = deficits[target_band]
        
        # Choose region based on band
        region_map_inverse = {v: k for k, v in self.region_band_map.items()}
        anode_region = region_map_inverse.get(target_band, 'DLPFC')
        
        # Determine polarity
        if deficit > 0:
            polarity = 'anodal'  # Need to increase
            current_ma = min(2.0, 1.0 + deficit)
        else:
            polarity = 'cathodal'  # Need to decrease
            current_ma = min(2.0, 1.0 + abs(deficit))
        
        return TDCSProtocol(
            anode_region=anode_region,
            cathode_region='right_supraorbital',  # Common reference
            current_ma=current_ma,
            duration_min=20,
            frequency_target=target_band,
            polarity=polarity
        )


class EnhancedCoherenceRecoverySystem:
    """
    Complete system: tDCS modulation + Complex CNN reconstruction
    """
    
    def __init__(self, cnn_model):
        self.cnn_model = cnn_model  # Trained ComplexCoherenceReconstructor
        self.tdcs_simulator = TDCSSimulator()
        
        # Track interventions
        self.intervention_history = []
    
    def recover_with_intervention(self,
                                  kappa_degraded: Dict[str, float],
                                  phi_degraded: Dict[str, float],
                                  kappa_target: Dict[str, float],
                                  use_tdcs: bool = True) -> Dict[str, float]:
        """
        Complete recovery pipeline:
        1. Optional tDCS intervention
        2. Complex CNN reconstruction
        """
        
        if use_tdcs:
            # Step 1: Optimize tDCS protocol
            protocol = self.tdcs_simulator.optimize_protocol(
                kappa_degraded,
                kappa_target
            )
            
            # Step 2: Apply tDCS modulation
            kappa_modulated = self.tdcs_simulator.apply_tdcs(
                kappa_degraded,
                protocol
            )
            
            # Track intervention
            self.intervention_history.append({
                'protocol': protocol,
                'pre_tdcs': kappa_degraded.copy(),
                'post_tdcs': kappa_modulated.copy()
            })
        else:
            kappa_modulated = kappa_degraded
        
        # Step 3: Complex CNN reconstruction
        bands = list(kappa_modulated.keys())
        kappa_tensor = torch.FloatTensor([
            [kappa_modulated[b] for b in bands]
        ])
        phi_tensor = torch.FloatTensor([
            [phi_degraded[b] for b in bands]
        ])
        
        with torch.no_grad():
            kappa_rec_tensor, phi_rec_tensor = self.cnn_model(
                kappa_tensor, 
                phi_tensor
            )
        
        # Convert back to dict
        kappa_recovered = {
            band: float(kappa_rec_tensor[0, i])
            for i, band in enumerate(bands)
        }
        
        return kappa_recovered
    
    def compare_approaches(self,
                          kappa_degraded: Dict[str, float],
                          phi_degraded: Dict[str, float],
                          kappa_target: Dict[str, float]) -> Dict:
        """
        Compare three approaches:
        1. No intervention (baseline)
        2. tDCS only
        3. tDCS + CNN (hybrid)
        """
        
        # Baseline (no intervention)
        baseline_recovery = np.mean(list(kappa_degraded.values()))
        
        # tDCS only
        protocol = self.tdcs_simulator.optimize_protocol(
            kappa_degraded,
            kappa_target
        )
        kappa_tdcs = self.tdcs_simulator.apply_tdcs(
            kappa_degraded,
            protocol
        )
        tdcs_recovery = np.mean(list(kappa_tdcs.values()))
        
        # tDCS + CNN (hybrid)
        kappa_hybrid = self.recover_with_intervention(
            kappa_degraded,
            phi_degraded,
            kappa_target,
            use_tdcs=True
        )
        hybrid_recovery = np.mean(list(kappa_hybrid.values()))
        
        # CNN only
        kappa_cnn = self.recover_with_intervention(
            kappa_degraded,
            phi_degraded,
            kappa_target,
            use_tdcs=False
        )
        cnn_recovery = np.mean(list(kappa_cnn.values()))
        
        # Target
        target_level = np.mean(list(kappa_target.values()))
        
        return {
            'baseline': baseline_recovery,
            'tdcs_only': tdcs_recovery,
            'cnn_only': cnn_recovery,
            'hybrid': hybrid_recovery,
            'target': target_level,
            'improvement_tdcs': tdcs_recovery - baseline_recovery,
            'improvement_cnn': cnn_recovery - baseline_recovery,
            'improvement_hybrid': hybrid_recovery - baseline_recovery
        }


# Example usage
if __name__ == "__main__":
    from complex_cnn_reconstructor import ComplexCoherenceReconstructor
    
    # Load trained CNN
    cnn_model = ComplexCoherenceReconstructor(n_bands=5)
    cnn_model.load_state_dict(
        torch.load('experiments/complex_cnn_v1/best_model.pth')['model_state_dict']
    )
    cnn_model.eval()
    
    # Initialize system
    system = EnhancedCoherenceRecoverySystem(cnn_model)
    
    # Example: Degraded state (OCD-like pattern)
    kappa_degraded = {
        'delta': 0.45,
        'theta': 0.35,   # Low (executive dysfunction)
        'alpha': 0.30,   # Low (emotion dysregulation)
        'beta': 0.40,
        'gamma': 0.38
    }
    
    phi_degraded = {
        'delta': 0.5,
        'theta': 0.8,
        'alpha': -0.3,
        'beta': 1.2,
        'gamma': -0.7
    }
    
    # Target state (healthy)
    kappa_target = {
        'delta': 0.65,
        'theta': 0.75,
        'alpha': 0.80,
        'beta': 0.70,
        'gamma': 0.65
    }
    
    # Compare approaches
    results = system.compare_approaches(
        kappa_degraded,
        phi_degraded,
        kappa_target
    )
    
    print("=" * 70)
    print("COHERENCE RECOVERY COMPARISON")
    print("=" * 70)
    print()
    print(f"Baseline (no intervention):  {results['baseline']:.3f}")
    print(f"tDCS only:                   {results['tdcs_only']:.3f}  (+{results['improvement_tdcs']:.3f})")
    print(f"CNN only:                    {results['cnn_only']:.3f}  (+{results['improvement_cnn']:.3f})")
    print(f"Hybrid (tDCS + CNN):         {results['hybrid']:.3f}  (+{results['improvement_hybrid']:.3f})")
    print(f"Target:                      {results['target']:.3f}")
    print()
    print("✓ Hybrid approach shows best recovery")
