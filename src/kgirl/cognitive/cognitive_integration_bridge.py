#!/usr/bin/env python3
"""
Cognitive Integration Bridge
============================
Bridge module connecting holographic memory system with LiMps 
Cognitive Communication Organism without modifying existing code.

This module acts as an adapter layer that:
- Maps cognitive states between systems
- Enables holographic memory access from cognitive organisms
- Integrates emergent cognitive features
- Maintains backward compatibility
"""

import sys
import os
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

# Import holographic memory system
from holographic_memory_system import (
    EnhancedCognitiveMemoryOrchestrator,
    HolographicAssociativeMemory,
    FractalMemoryEncoder,
    QuantumHolographicStorage,
    EmergentMemoryPatterns
)

# Import LiMps components (will import from existing system)
try:
    from cognitive_communication_organism import (
        CognitiveCommunicationOrganism,
        CognitiveState,
        CognitiveLevel,
        CommunicationContext
    )
    LIMPS_AVAILABLE = True
except ImportError:
    LIMPS_AVAILABLE = False
    logging.warning("LiMps cognitive_communication_organism not available")

# Import emergent cognitive network if available
try:
    sys.path.append('/home/kill/numbskull')
    from emergent_cognitive_system import (
        EmergentCognitiveOrchestrator,
        QuantumOptimizationStep,
        SwarmCognitiveStep
    )
    EMERGENT_AVAILABLE = True
except ImportError:
    EMERGENT_AVAILABLE = False
    logging.warning("Emergent cognitive system not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IntegratedCognitiveState:
    """Unified cognitive state spanning holographic memory and LiMps"""
    # LiMps cognitive state
    cognitive_level: str
    stability_score: float
    entropy_score: float
    complexity_score: float
    
    # Holographic memory state
    memory_integration_level: float
    memory_resilience: float
    emergence_detected: bool
    
    # Cross-system metrics
    holographic_coherence: float
    quantum_amplitude: float
    fractal_dimension: float
    
    # Metadata
    timestamp: float
    context: Dict[str, Any]


class CognitiveStateMapper:
    """Maps cognitive states between different systems"""
    
    def __init__(self):
        self.mapping_history = []
        self.coherence_threshold = 0.5
        
    def limps_to_holographic(self, cognitive_state: 'CognitiveState') -> Dict:
        """Convert LiMps CognitiveState to holographic memory context"""
        holographic_context = {
            'emotional_valence': cognitive_state.stability_score,
            'cognitive_significance': cognitive_state.complexity_score,
            'entropy_level': cognitive_state.entropy_score,
            'temporal_context': cognitive_state.temporal_context,
            'fractal_dimension': cognitive_state.fractal_dimension,
            'stability': cognitive_state.stability_score
        }
        
        return holographic_context
    
    def holographic_to_limps(self, memory_result: Dict) -> Dict:
        """Convert holographic memory results to LiMps cognitive metrics"""
        limps_metrics = {
            'stability_score': memory_result.get('memory_resilience', 0.5),
            'complexity_score': memory_result.get('cognitive_integration_level', 0.5),
            'entropy_score': memory_result.get('emergence_analysis', {}).get('cognitive_emergence_level', 0.5),
            'coherence_score': memory_result.get('cognitive_integration_level', 0.5),
            'fractal_dimension': memory_result.get('memory_integration', {}).get('fractal', {}).get('fractal_dimension', 1.0)
        }
        
        return limps_metrics
    
    def create_integrated_state(self, 
                               limps_state: Optional['CognitiveState'],
                               memory_state: Dict) -> IntegratedCognitiveState:
        """Create unified cognitive state from both systems"""
        
        if limps_state:
            cognitive_level = limps_state.level.name
            stability = limps_state.stability_score
            entropy = limps_state.entropy_score
            complexity = limps_state.complexity_score
        else:
            cognitive_level = "NEURAL_COGNITION"
            stability = 0.5
            entropy = 0.5
            complexity = 0.5
        
        integrated_state = IntegratedCognitiveState(
            cognitive_level=cognitive_level,
            stability_score=stability,
            entropy_score=entropy,
            complexity_score=complexity,
            memory_integration_level=memory_state.get('cognitive_integration_level', 0.0),
            memory_resilience=memory_state.get('memory_resilience', 0.0),
            emergence_detected=memory_state.get('emergence_detected', False),
            holographic_coherence=self._calculate_holographic_coherence(memory_state),
            quantum_amplitude=self._extract_quantum_amplitude(memory_state),
            fractal_dimension=memory_state.get('memory_integration', {}).get('fractal', {}).get('fractal_dimension', 1.0),
            timestamp=np.datetime64('now').astype(float),
            context=memory_state.get('memory_integration', {})
        )
        
        self.mapping_history.append(integrated_state)
        return integrated_state
    
    def _calculate_holographic_coherence(self, memory_state: Dict) -> float:
        """Calculate holographic coherence from memory state"""
        integration = memory_state.get('memory_integration', {})
        
        # Coherence based on presence and quality of holographic encoding
        holographic_present = integration.get('holographic') is not None
        fractal_quality = integration.get('fractal', {}).get('self_similarity', 0.5)
        
        coherence = (float(holographic_present) + fractal_quality) / 2
        return coherence
    
    def _extract_quantum_amplitude(self, memory_state: Dict) -> float:
        """Extract quantum amplitude from memory state"""
        # Placeholder - would extract from actual quantum state
        return memory_state.get('memory_integration', {}).get('quantum_amplitude', 0.5)


class CognitiveHolographicBridge:
    """Main bridge between LiMps Cognitive Organism and Holographic Memory"""
    
    def __init__(self, 
                 cognitive_organism: Optional['CognitiveCommunicationOrganism'] = None,
                 memory_orchestrator: Optional[EnhancedCognitiveMemoryOrchestrator] = None):
        
        # Initialize memory orchestrator
        if memory_orchestrator is None:
            self.memory = EnhancedCognitiveMemoryOrchestrator()
        else:
            self.memory = memory_orchestrator
        
        # Reference to cognitive organism (if provided)
        self.organism = cognitive_organism
        
        # State mapper
        self.state_mapper = CognitiveStateMapper()
        
        # Processing history
        self.processing_history = []
        self.cognitive_memory_associations = {}
        
        logger.info("Cognitive Holographic Bridge initialized")
    
    def process_with_memory(self, 
                           communication_context: Dict,
                           cognitive_state: Optional['CognitiveState'] = None) -> Dict:
        """Process communication context with integrated holographic memory"""
        
        # Convert cognitive state to holographic context
        if cognitive_state:
            holographic_context = self.state_mapper.limps_to_holographic(cognitive_state)
        else:
            holographic_context = {
                'emotional_valence': 0.5,
                'cognitive_significance': 0.5,
                'stability': 0.5
            }
        
        # Extract data from communication context
        if isinstance(communication_context.get('message_content'), str):
            # Convert string to numeric data for holographic encoding
            data = self._text_to_numeric(communication_context['message_content'])
        else:
            # Use provided numeric data
            data = communication_context.get('data', np.random.random(256))
        
        # Store in holographic memory
        experience = {
            'data': data,
            'context': communication_context.get('message_content', 'Unknown'),
            'emotional_intensity': holographic_context.get('emotional_valence', 0.5)
        }
        
        memory_result = self.memory.integrated_memory_processing(experience, holographic_context)
        
        # Recall similar past experiences
        recall_query = {
            'data': data,
            'similarity_threshold': 0.6,
            'scale_preference': 'adaptive'
        }
        
        recall_result = self.memory.emergent_memory_recall(recall_query, 'integrated')
        
        # Create integrated cognitive state
        integrated_state = self.state_mapper.create_integrated_state(
            cognitive_state, memory_result
        )
        
        # Store association
        memory_key = memory_result['memory_integration']['holographic']
        self.cognitive_memory_associations[memory_key] = {
            'communication_context': communication_context,
            'cognitive_state': cognitive_state,
            'integrated_state': integrated_state,
            'timestamp': np.datetime64('now')
        }
        
        # Build comprehensive result
        result = {
            'memory_storage': memory_result,
            'memory_recall': recall_result,
            'integrated_state': integrated_state,
            'holographic_key': memory_key,
            'emergence_metrics': {
                'emergence_detected': memory_result['emergence_detected'],
                'cognitive_integration': memory_result['cognitive_integration_level'],
                'memory_resilience': memory_result['memory_resilience'],
                'holographic_coherence': integrated_state.holographic_coherence
            },
            'recommendations': self._generate_recommendations(memory_result, recall_result)
        }
        
        self.processing_history.append(result)
        
        logger.info(f"Processed with memory - Emergence: {result['emergence_metrics']['emergence_detected']}")
        
        return result
    
    def recall_similar_cognitive_states(self, 
                                       current_state: 'CognitiveState',
                                       similarity_threshold: float = 0.7) -> List[Dict]:
        """Recall similar cognitive states from holographic memory"""
        
        # Convert current state to holographic query
        holographic_context = self.state_mapper.limps_to_holographic(current_state)
        
        # Create query data from cognitive metrics
        query_data = np.array([
            current_state.stability_score,
            current_state.entropy_score,
            current_state.complexity_score,
            current_state.coherence_score,
            current_state.fractal_dimension
        ])
        
        # Pad to required dimension
        query_data = np.pad(query_data, (0, 256 - len(query_data)), mode='edge')
        
        query = {
            'data': query_data,
            'similarity_threshold': similarity_threshold,
            'scale_preference': 'adaptive'
        }
        
        recall_result = self.memory.emergent_memory_recall(query, 'integrated')
        
        # Map results back to cognitive context
        similar_states = []
        for match in recall_result.get('holographic', [])[:5]:  # Top 5
            memory_key = match['memory_key']
            if memory_key in self.cognitive_memory_associations:
                association = self.cognitive_memory_associations[memory_key]
                similar_states.append({
                    'memory_key': memory_key,
                    'similarity': match['similarity'],
                    'past_context': association['communication_context'],
                    'past_cognitive_state': association['cognitive_state'],
                    'emotional_context': match['emotional_context']
                })
        
        return similar_states
    
    def enhance_cognitive_decision(self,
                                  communication_context: Dict,
                                  proposed_decision: Dict) -> Dict:
        """Enhance cognitive decision using memory-based insights"""
        
        # Recall similar past situations
        if isinstance(communication_context.get('message_content'), str):
            data = self._text_to_numeric(communication_context['message_content'])
        else:
            data = communication_context.get('data', np.random.random(256))
        
        query = {
            'data': data,
            'similarity_threshold': 0.6
        }
        
        recall_result = self.memory.emergent_memory_recall(query, 'integrated')
        
        # Extract insights from recalled memories
        insights = self._extract_decision_insights(recall_result)
        
        # Enhance decision with memory insights
        enhanced_decision = {
            **proposed_decision,
            'memory_informed': True,
            'confidence_adjustment': insights['confidence_modifier'],
            'recommended_strategy': insights['strategy_recommendation'],
            'emergence_prediction': recall_result.get('emergence_prediction', {}),
            'similar_past_outcomes': insights['past_outcomes']
        }
        
        return enhanced_decision
    
    def get_cognitive_trajectory_analysis(self) -> Dict:
        """Analyze cognitive trajectory across integrated system"""
        
        if not self.processing_history:
            return {'status': 'No processing history available'}
        
        # Analyze emergence patterns over time
        emergence_events = [
            h['emergence_metrics']['emergence_detected'] 
            for h in self.processing_history
        ]
        
        # Analyze integration levels
        integration_levels = [
            h['emergence_metrics']['cognitive_integration']
            for h in self.processing_history
        ]
        
        # Analyze holographic coherence
        coherence_levels = [
            h['emergence_metrics']['holographic_coherence']
            for h in self.processing_history
        ]
        
        analysis = {
            'total_processes': len(self.processing_history),
            'emergence_rate': np.mean(emergence_events),
            'average_integration': np.mean(integration_levels),
            'integration_trend': np.polyfit(range(len(integration_levels)), integration_levels, 1)[0] if len(integration_levels) > 1 else 0,
            'average_coherence': np.mean(coherence_levels),
            'coherence_stability': 1.0 - np.std(coherence_levels),
            'metacognitive_state': self.memory.memory_metacognition,
            'cognitive_efficiency': self._calculate_system_efficiency()
        }
        
        return analysis
    
    def _text_to_numeric(self, text: str) -> np.ndarray:
        """Convert text to numeric representation for holographic encoding"""
        # Simple character-based encoding
        if not text:
            return np.random.random(256)
        
        # Use character codes
        char_values = np.array([ord(c) for c in text[:256]])
        
        # Normalize to [0, 1] range
        char_values = char_values / 255.0
        
        # Pad to required length
        if len(char_values) < 256:
            char_values = np.pad(char_values, (0, 256 - len(char_values)), mode='wrap')
        
        return char_values
    
    def _generate_recommendations(self, memory_result: Dict, recall_result: Dict) -> Dict:
        """Generate recommendations based on memory processing"""
        
        emergence_level = memory_result['emergence_analysis'].get('cognitive_emergence_level', 0)
        integration_level = memory_result['cognitive_integration_level']
        
        recommendations = {
            'modulation_strategy': 'adaptive',
            'cognitive_mode': 'explorative' if emergence_level > 0.6 else 'conservative',
            'memory_consolidation_needed': integration_level < 0.4,
            'emergence_attention': emergence_level > 0.7
        }
        
        # Specific recommendations based on recall
        if recall_result.get('integrated', {}).get('recall_confidence', 0) > 0.8:
            recommendations['use_past_patterns'] = True
            recommendations['pattern_source'] = 'holographic_memory'
        
        return recommendations
    
    def _extract_decision_insights(self, recall_result: Dict) -> Dict:
        """Extract decision-making insights from recall results"""
        
        integrated = recall_result.get('integrated', {})
        
        insights = {
            'confidence_modifier': integrated.get('recall_confidence', 0.5) - 0.5,  # -0.5 to +0.5
            'strategy_recommendation': self._determine_strategy(recall_result),
            'past_outcomes': []
        }
        
        # Extract past outcomes from best matches
        for match in integrated.get('best_matches', [])[:3]:
            insights['past_outcomes'].append({
                'source': match['source'],
                'similarity': match['similarity'],
                'outcome_quality': match.get('emotional_context', 0.5)
            })
        
        return insights
    
    def _determine_strategy(self, recall_result: Dict) -> str:
        """Determine recommended strategy based on recall"""
        
        emergence_confidence = recall_result.get('emergence_prediction', {}).get('emergence_forecast_confidence', 0.5)
        
        if emergence_confidence > 0.7:
            return 'emergent_adaptation'
        elif emergence_confidence > 0.4:
            return 'balanced_approach'
        else:
            return 'conservative_known_patterns'
    
    def _calculate_system_efficiency(self) -> float:
        """Calculate overall integrated system efficiency"""
        
        if not self.processing_history:
            return 0.0
        
        recent_processes = self.processing_history[-10:]  # Last 10
        
        efficiencies = [
            (p['emergence_metrics']['cognitive_integration'] + 
             p['emergence_metrics']['holographic_coherence']) / 2
            for p in recent_processes
        ]
        
        return float(np.mean(efficiencies))


class EmergentCognitiveBridge:
    """Bridge to emergent cognitive network for advanced processing"""
    
    def __init__(self):
        self.emergent_available = EMERGENT_AVAILABLE
        
        if EMERGENT_AVAILABLE:
            self.emergent_orchestrator = EmergentCognitiveOrchestrator()
            logger.info("Emergent cognitive bridge initialized with full capabilities")
        else:
            self.emergent_orchestrator = None
            logger.warning("Emergent cognitive network not available - limited functionality")
    
    def process_emergent_cognition(self, input_data: torch.Tensor) -> Dict:
        """Process input through emergent cognitive network"""
        
        if not self.emergent_available:
            return {'status': 'Emergent network unavailable', 'fallback': True}
        
        try:
            # Execute cognitive cycle
            cycle_result = self.emergent_orchestrator.execute_cognitive_cycle(input_data)
            
            return {
                'status': 'success',
                'quantum_state': cycle_result.get('quantum_state'),
                'swarm_results': cycle_result.get('swarm_results'),
                'neural_results': cycle_result.get('neural_results'),
                'emergence_metrics': cycle_result.get('emergence_metrics'),
                'fallback': False
            }
        
        except Exception as e:
            logger.error(f"Emergent cognition processing error: {e}")
            return {'status': 'error', 'error': str(e), 'fallback': True}


def create_integrated_bridge(cognitive_organism: Optional['CognitiveCommunicationOrganism'] = None) -> CognitiveHolographicBridge:
    """Factory function to create integrated cognitive-holographic bridge"""
    
    bridge = CognitiveHolographicBridge(cognitive_organism=cognitive_organism)
    
    logger.info("Integrated cognitive-holographic bridge created successfully")
    logger.info(f"LiMps available: {LIMPS_AVAILABLE}")
    logger.info(f"Emergent network available: {EMERGENT_AVAILABLE}")
    
    return bridge


if __name__ == "__main__":
    # Demonstration of bridge functionality
    print("=== Cognitive Integration Bridge Demo ===\n")
    
    # Create bridge
    bridge = create_integrated_bridge()
    
    # Test processing with synthetic communication context
    test_context = {
        'message_content': "Test cognitive communication with holographic memory integration",
        'channel_conditions': {'SNR': 15.0, 'bandwidth': 1e6},
        'priority_level': 7
    }
    
    result = bridge.process_with_memory(test_context)
    
    print(f"Processing Result:")
    print(f"  Holographic Key: {result['holographic_key']}")
    print(f"  Emergence Detected: {result['emergence_metrics']['emergence_detected']}")
    print(f"  Cognitive Integration: {result['emergence_metrics']['cognitive_integration']:.3f}")
    print(f"  Memory Resilience: {result['emergence_metrics']['memory_resilience']:.3f}")
    print(f"  Holographic Coherence: {result['emergence_metrics']['holographic_coherence']:.3f}")
    
    print(f"\nRecommendations:")
    for key, value in result['recommendations'].items():
        print(f"  {key}: {value}")
    
    # Analyze trajectory
    print(f"\n=== Cognitive Trajectory Analysis ===")
    analysis = bridge.get_cognitive_trajectory_analysis()
    for key, value in analysis.items():
        if key != 'metacognitive_state':
            print(f"  {key}: {value}")

