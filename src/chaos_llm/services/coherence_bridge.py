"""
coherence_bridge.py
Bridge between NewThought and Unified Coherence Recovery systems
Enables advanced coherence recovery for thought vectors
"""

from typing import Dict, List, Optional
import numpy as np

from .newthought import Thought, ThoughtCascade, newthought_service
from .unified_coherence_recovery import (
    UnifiedCoherenceRecoverySystem,
    FrequencyBand,
    unified_coherence_recovery_system
)


class CoherenceBridge:
    """
    Bridge between NewThought and Unified Coherence Recovery

    Maps thought coherence to EEG frequency bands and enables
    advanced quantum-inspired coherence recovery.
    """

    def __init__(self):
        self.newthought = newthought_service
        self.coherence_system = unified_coherence_recovery_system

        # Mapping: thought depth -> dominant frequency band
        self.depth_to_band = {
            0: FrequencyBand.GAMMA,  # Surface thoughts - high frequency
            1: FrequencyBand.BETA,   # Active thinking
            2: FrequencyBand.ALPHA,  # Relaxed awareness
            3: FrequencyBand.THETA,  # Deep insight
            4: FrequencyBand.DELTA,  # Foundational thoughts
            5: FrequencyBand.DELTA,  # Deepest level
        }

    def thought_to_frequency_bands(self, thought: Thought) -> Dict[FrequencyBand, float]:
        """
        Map thought coherence to EEG frequency bands

        Strategy:
        - Dominant band based on recursion depth
        - Coherence score distributed across nearby bands
        - Entropy affects band spread
        """
        kappa = {}

        # Get dominant band for this depth
        dominant_band = self.depth_to_band.get(
            thought.depth,
            FrequencyBand.ALPHA  # Default to alpha
        )

        # Base coherence from thought
        base_coherence = thought.coherence_score

        # Entropy affects distribution spread
        spread_factor = thought.entropy  # Higher entropy = more spread

        # Distribute coherence across bands
        bands = list(FrequencyBand)
        dominant_idx = bands.index(dominant_band)

        for idx, band in enumerate(bands):
            distance_from_dominant = abs(idx - dominant_idx)

            if distance_from_dominant == 0:
                # Dominant band gets most coherence
                kappa[band] = base_coherence * (1.0 - spread_factor * 0.5)
            else:
                # Nearby bands get proportional coherence
                falloff = np.exp(-distance_from_dominant / (1 + spread_factor))
                kappa[band] = base_coherence * falloff * spread_factor

        # Normalize to ensure valid range [0, 1]
        for band in kappa:
            kappa[band] = np.clip(kappa[band], 0.0, 1.0)

        return kappa

    def thought_to_phases(self, thought: Thought) -> Dict[FrequencyBand, float]:
        """
        Extract phase information from thought embedding

        Uses thought vector components to derive phases
        """
        phi = {}

        # Use thought embedding to derive phases
        embedding_chunks = np.array_split(thought.embedding, len(FrequencyBand))

        for idx, band in enumerate(FrequencyBand):
            chunk = embedding_chunks[idx]
            # Phase from complex representation
            phase_component = np.angle(np.mean(chunk + 1j * np.roll(chunk, 1)))
            # Normalize to [0, 2π]
            phi[band] = (phase_component + np.pi) % (2 * np.pi)

        return phi

    def frequency_bands_to_thought_coherence(
        self,
        kappa: Dict[FrequencyBand, float],
        original_thought: Thought
    ) -> float:
        """
        Convert recovered frequency band coherences back to thought coherence

        Strategy:
        - Weight by depth-appropriate band
        - Average across relevant bands
        """
        dominant_band = self.depth_to_band.get(
            original_thought.depth,
            FrequencyBand.ALPHA
        )

        # Weighted average: dominant band has higher weight
        bands = list(FrequencyBand)
        dominant_idx = bands.index(dominant_band)

        weighted_sum = 0.0
        total_weight = 0.0

        for idx, band in enumerate(bands):
            distance_from_dominant = abs(idx - dominant_idx)
            weight = np.exp(-distance_from_dominant / 2.0)  # Gaussian weighting

            weighted_sum += kappa[band] * weight
            total_weight += weight

        recovered_coherence = weighted_sum / total_weight if total_weight > 0 else 0.5

        return float(np.clip(recovered_coherence, 0.0, 1.0))

    async def recover_thought_coherence(
        self,
        thought: Thought,
        timestamp: float
    ) -> Optional[Thought]:
        """
        Use unified coherence recovery to restore degraded thought

        Process:
        1. Map thought to frequency bands
        2. Apply unified coherence recovery
        3. Map recovered bands back to thought
        4. Return recovered thought
        """
        # Step 1: Map to frequency bands
        kappa_current = self.thought_to_frequency_bands(thought)
        phi_current = self.thought_to_phases(thought)

        # Step 2: Apply coherence recovery
        kappa_recovered = self.coherence_system.process(
            kappa_current=kappa_current,
            phi_current=phi_current,
            t_current=timestamp
        )

        if kappa_recovered is None:
            # Emergency decouple - recovery failed
            return None

        # Step 3: Map back to thought
        recovered_coherence = self.frequency_bands_to_thought_coherence(
            kappa_recovered,
            thought
        )

        # Create recovered thought
        recovered_thought = Thought(
            content=thought.content,
            embedding=thought.embedding.copy(),
            coherence_score=recovered_coherence,
            entropy=1.0 - recovered_coherence,  # Inverse relationship
            depth=thought.depth,
            timestamp=timestamp,
            parent_id=thought.parent_id,
            metadata={
                **thought.metadata,
                'recovery_applied': True,
                'original_coherence': thought.coherence_score,
                'recovery_gain': recovered_coherence - thought.coherence_score,
                'frequency_bands': {
                    band.value: kappa_recovered[band]
                    for band in FrequencyBand
                }
            }
        )

        return recovered_thought

    async def recover_thought_cascade(
        self,
        cascade: ThoughtCascade,
        timestamp: float
    ) -> ThoughtCascade:
        """
        Apply coherence recovery to entire thought cascade

        Recovers degraded thoughts while maintaining cascade structure
        """
        recovered_children = []

        for child_thought in cascade.children:
            # Only recover thoughts with low coherence
            if child_thought.coherence_score < 0.6:
                recovered = await self.recover_thought_coherence(
                    child_thought,
                    timestamp
                )

                if recovered is not None:
                    recovered_children.append(recovered)
                else:
                    # Keep original if recovery failed
                    recovered_children.append(child_thought)
            else:
                # Already coherent, keep as is
                recovered_children.append(child_thought)

        # Update cascade
        recovered_cascade = ThoughtCascade(
            root_thought=cascade.root_thought,
            children=recovered_children,
            cascade_coherence=np.mean([t.coherence_score for t in recovered_children]) if recovered_children else 0.0,
            emergence_patterns=cascade.emergence_patterns.copy(),
            total_entropy=np.sum([t.entropy for t in recovered_children])
        )

        # Add recovery pattern
        recovery_count = sum(
            1 for t in recovered_children
            if t.metadata.get('recovery_applied', False)
        )

        if recovery_count > 0:
            recovered_cascade.emergence_patterns.append('coherence_recovery_applied')

        return recovered_cascade

    def get_system_statistics(self) -> Dict:
        """Get combined statistics from both systems"""
        newthought_stats = self.newthought.get_statistics()
        coherence_history = self.coherence_system.export_history()

        return {
            'newthought': newthought_stats,
            'coherence_recovery': {
                'events': len(coherence_history),
                'successful_recoveries': sum(
                    1 for e in coherence_history
                    if e['type'] == 'successful_renewal'
                ),
                'emergency_decouples': sum(
                    1 for e in coherence_history
                    if e['type'] == 'audit_failure'
                ),
                'system_active': self.coherence_system.capsule is not None
            }
        }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

coherence_bridge = CoherenceBridge()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        print("=" * 70)
        print("COHERENCE BRIDGE DEMONSTRATION")
        print("=" * 70)
        print()

        # Create a thought with low coherence
        degraded_thought = Thought(
            content="Quantum computing enables parallel computation through superposition",
            embedding=np.random.randn(768) * 0.1,  # Weak embedding
            coherence_score=0.35,  # Low coherence
            entropy=0.65,
            depth=2,
            timestamp=1.0
        )

        print("Original thought:")
        print(f"  Content: {degraded_thought.content[:50]}...")
        print(f"  Coherence: {degraded_thought.coherence_score:.3f}")
        print(f"  Entropy: {degraded_thought.entropy:.3f}")
        print(f"  Depth: {degraded_thought.depth}")
        print()

        # Map to frequency bands
        print("Mapping to frequency bands...")
        kappa = coherence_bridge.thought_to_frequency_bands(degraded_thought)
        for band, value in kappa.items():
            print(f"  {band.value:6s}: κ={value:.3f}")
        print()

        # Apply recovery
        print("Applying coherence recovery...")
        recovered_thought = await coherence_bridge.recover_thought_coherence(
            degraded_thought,
            timestamp=2.0
        )
        print()

        if recovered_thought:
            print("✓ Recovery successful!")
            print(f"  Original coherence: {degraded_thought.coherence_score:.3f}")
            print(f"  Recovered coherence: {recovered_thought.coherence_score:.3f}")
            print(f"  Improvement: {recovered_thought.coherence_score - degraded_thought.coherence_score:+.3f}")
            print()

            # Show frequency band contributions
            if 'frequency_bands' in recovered_thought.metadata:
                print("  Recovered frequency bands:")
                for band, value in recovered_thought.metadata['frequency_bands'].items():
                    print(f"    {band:6s}: κ={value:.3f}")
        else:
            print("✗ Recovery failed (emergency decouple)")

        print()
        print("=" * 70)

    asyncio.run(demo())
