#!/usr/bin/env python3
"""
Recursive AI System - Complete Integration
Integrates all components into a cohesive recursive AI architecture
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np

# Import all components
from recursive_ai_core import RecursiveProcessor, RecursiveCognitionRequest
from matrix_processor import MatrixProcessor, MatrixConfig
from fractal_resonance import FractalResonanceSystem, ResonanceConfig
from distributed_knowledge_base import DistributedKnowledgeBase, KnowledgeBaseConfig
from emergent_visualizer import EmergentVisualizer, VisualizationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RecursiveAIConfig:
    """Configuration for the complete recursive AI system"""
    # Core settings
    max_recursion_depth: int = 5
    hallucination_temperature: float = 0.8
    coherence_threshold: float = 0.6
    
    # Matrix processing
    embedding_dimension: int = 768
    matrix_optimization_level: int = 2
    
    # Resonance simulation
    resonance_frequency: float = 1.0
    fractal_depth: int = 3
    
    # Knowledge base
    knowledge_db_path: str = "recursive_ai_knowledge.db"
    faiss_index_path: str = "recursive_ai_faiss_index"
    
    # Visualization
    visualization_resolution: int = 100
    enable_visualization: bool = True

@dataclass
class RecursiveAIResult:
    """Result from recursive AI processing"""
    input_text: str
    recursion_depth: int
    insights: List[Dict[str, Any]]
    compiled_knowledge: Dict[str, Any]
    resonance_field: Optional[Any] = None
    emergent_patterns: List[Any] = field(default_factory=list)
    processing_time: float = 0.0
    cognitive_state: Dict[str, Any] = field(default_factory=dict)

class RecursiveAISystem:
    """Complete recursive AI system integrating all components"""
    
    def __init__(self, config: RecursiveAIConfig = None):
        self.config = config or RecursiveAIConfig()
        self.is_initialized = False
        
        # Initialize components
        self.recursive_processor = RecursiveProcessor(
            max_depth=self.config.max_recursion_depth,
            temperature=self.config.hallucination_temperature,
            coherence_threshold=self.config.coherence_threshold
        )
        
        self.matrix_processor = MatrixProcessor(
            MatrixConfig(
                embedding_dim=self.config.embedding_dimension,
                optimization_level=self.config.matrix_optimization_level
            )
        )
        
        self.resonance_system = FractalResonanceSystem(
            ResonanceConfig(
                resonance_frequency=self.config.resonance_frequency,
                fractal_depth=self.config.fractal_depth
            )
        )
        
        self.knowledge_base = DistributedKnowledgeBase(
            KnowledgeBaseConfig(
                db_path=self.config.knowledge_db_path,
                faiss_index_path=self.config.faiss_index_path,
                embedding_dimension=self.config.embedding_dimension
            )
        )
        
        self.visualizer = EmergentVisualizer(
            VisualizationConfig(
                resolution=self.config.visualization_resolution
            )
        ) if self.config.enable_visualization else None
        
        # System state
        self.processing_history: List[RecursiveAIResult] = []
        self.total_insights_generated = 0
        self.total_patterns_detected = 0
        
    async def initialize(self) -> bool:
        """Initialize all system components"""
        logger.info("Initializing Recursive AI System...")
        
        try:
            # Initialize recursive processor
            await self.recursive_processor.initialize()
            logger.info("✓ Recursive processor initialized")
            
            # Initialize knowledge base
            if not await self.knowledge_base.initialize():
                logger.warning("⚠ Knowledge base initialization failed, continuing without persistence")
            
            logger.info("✓ Knowledge base initialized")
            
            self.is_initialized = True
            logger.info("🎉 Recursive AI System fully initialized!")
            return True
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            return False
    
    async def process_recursive_cognition(self, input_text: str, 
                                        depth: int = None) -> RecursiveAIResult:
        """Main processing pipeline for recursive cognition"""
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        start_time = time.time()
        depth = depth or self.config.max_recursion_depth
        
        logger.info(f"🔄 Processing recursive cognition: '{input_text[:50]}...' (depth={depth})")
        
        try:
            # Step 1: Recursive analysis
            logger.info("Step 1: Recursive analysis...")
            recursive_result = await self.recursive_processor.recursive_cognition(
                RecursiveCognitionRequest(
                    input_text=input_text,
                    depth=depth,
                    temperature=self.config.hallucination_temperature,
                    coherence_threshold=self.config.coherence_threshold
                )
            )
            
            insights = recursive_result.insights
            logger.info(f"✓ Generated {len(insights)} insights")
            
            # Step 2: Matrix processing and compilation
            logger.info("Step 2: Matrix processing...")
            insight_texts = [insight["text"] for insight in insights]
            matrix_result = await self.matrix_processor.process_matrices(
                insight_texts,
                [{"id": i, "depth": insight.get("depth", 0), "type": insight.get("type", "unknown")} 
                 for i, insight in enumerate(insights)]
            )
            
            logger.info(f"✓ Matrix processing complete: {matrix_result['statistics']['total_vectors']} vectors")
            
            # Step 3: Fractal resonance simulation
            logger.info("Step 3: Fractal resonance simulation...")
            if matrix_result.get("vectors") is not None:
                vectors = np.array(matrix_result["vectors"])
                resonance_field = await self.resonance_system.process_resonance(
                    [vectors[i] for i in range(min(5, len(vectors)))]  # Limit to 5 vectors for performance
                )
                logger.info(f"✓ Resonance field created: strength={resonance_field.resonance_strength:.3f}")
            else:
                resonance_field = None
                logger.warning("⚠ No vectors available for resonance simulation")
            
            # Step 4: Knowledge base integration
            logger.info("Step 4: Knowledge base integration...")
            for i, insight in enumerate(insights):
                if matrix_result.get("vectors") is not None and i < len(matrix_result["vectors"]):
                    embedding = np.array(matrix_result["vectors"][i])
                    await self.knowledge_base.add_knowledge_node(
                        content=insight["text"],
                        embedding=embedding,
                        source="recursive_cognition",
                        metadata={
                            "depth": insight.get("depth", 0),
                            "type": insight.get("type", "unknown"),
                            "coherence": insight.get("coherence", 0.0),
                            "timestamp": time.time()
                        }
                    )
            
            logger.info("✓ Knowledge base updated")
            
            # Step 5: Emergent pattern detection and visualization
            emergent_patterns = []
            if self.visualizer and matrix_result.get("vectors") is not None:
                logger.info("Step 5: Emergent pattern detection...")
                self.visualizer.visualize_knowledge_graph(
                    insights, 
                    np.array(matrix_result["vectors"])
                )
                emergent_patterns = self.visualizer.patterns_history[-len(insights):] if self.visualizer.patterns_history else []
                logger.info(f"✓ Detected {len(emergent_patterns)} emergent patterns")
            
            # Compile final result
            processing_time = time.time() - start_time
            
            result = RecursiveAIResult(
                input_text=input_text,
                recursion_depth=depth,
                insights=insights,
                compiled_knowledge=matrix_result,
                resonance_field=resonance_field,
                emergent_patterns=emergent_patterns,
                processing_time=processing_time,
                cognitive_state=recursive_result.cognitive_state
            )
            
            # Update system state
            self.processing_history.append(result)
            self.total_insights_generated += len(insights)
            self.total_patterns_detected += len(emergent_patterns)
            
            logger.info(f"🎉 Processing complete in {processing_time:.2f}s")
            logger.info(f"📊 Generated {len(insights)} insights, {len(emergent_patterns)} patterns")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            raise
    
    async def search_knowledge(self, query: str, k: int = 5) -> List[Any]:
        """Search the knowledge base"""
        if not self.is_initialized:
            raise RuntimeError("System not initialized")
        
        logger.info(f"🔍 Searching knowledge base: '{query}'")
        
        # Generate query embedding (simplified)
        query_embedding = np.random.randn(self.config.embedding_dimension)
        
        results = await self.knowledge_base.search_knowledge(query, query_embedding, k)
        logger.info(f"✓ Found {len(results)} relevant knowledge nodes")
        
        return results
    
    async def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        # Knowledge base statistics
        kb_stats = self.knowledge_base.get_statistics()
        
        # Resonance statistics
        resonance_stats = self.resonance_system.get_resonance_statistics()
        
        # Visualization statistics
        viz_stats = self.visualizer.get_emergence_statistics() if self.visualizer else {}
        
        # Processing history statistics
        if self.processing_history:
            avg_processing_time = np.mean([r.processing_time for r in self.processing_history])
            avg_insights_per_query = np.mean([len(r.insights) for r in self.processing_history])
            avg_patterns_per_query = np.mean([len(r.emergent_patterns) for r in self.processing_history])
        else:
            avg_processing_time = 0.0
            avg_insights_per_query = 0.0
            avg_patterns_per_query = 0.0
        
        return {
            "system_status": "initialized" if self.is_initialized else "not_initialized",
            "total_queries_processed": len(self.processing_history),
            "total_insights_generated": self.total_insights_generated,
            "total_patterns_detected": self.total_patterns_detected,
            "average_processing_time": float(avg_processing_time),
            "average_insights_per_query": float(avg_insights_per_query),
            "average_patterns_per_query": float(avg_patterns_per_query),
            "knowledge_base": kb_stats,
            "resonance_system": resonance_stats,
            "visualization": viz_stats,
            "configuration": {
                "max_recursion_depth": self.config.max_recursion_depth,
                "embedding_dimension": self.config.embedding_dimension,
                "resonance_frequency": self.config.resonance_frequency,
                "fractal_depth": self.config.fractal_depth
            }
        }
    
    async def visualize_system_state(self, save_path: str = None) -> None:
        """Visualize current system state"""
        if not self.visualizer:
            logger.warning("Visualization not enabled")
            return
        
        logger.info("🎨 Generating system visualization...")
        
        # Get recent insights for visualization
        recent_insights = []
        recent_embeddings = []
        
        for result in self.processing_history[-3:]:  # Last 3 queries
            recent_insights.extend(result.insights)
            if result.compiled_knowledge.get("vectors"):
                recent_embeddings.extend(result.compiled_knowledge["vectors"])
        
        if recent_insights and recent_embeddings:
            # Visualize knowledge graph
            self.visualizer.visualize_knowledge_graph(
                recent_insights,
                np.array(recent_embeddings),
                f"{save_path}_knowledge_graph.html" if save_path else None
            )
            
            # Visualize emergence timeline
            self.visualizer.visualize_emergence_timeline(
                f"{save_path}_timeline.html" if save_path else None
            )
            
            # Visualize 3D fractals
            self.visualizer.visualize_3d_fractal(
                "mandelbrot",
                f"{save_path}_mandelbrot.html" if save_path else None
            )
            
            logger.info("✓ System visualization complete")
        else:
            logger.warning("⚠ No data available for visualization")
    
    async def close(self):
        """Close the system and cleanup resources"""
        logger.info("🔄 Closing Recursive AI System...")
        
        try:
            # Close knowledge base
            await self.knowledge_base.close()
            
            # Close recursive processor
            if hasattr(self.recursive_processor, 'close'):
                await self.recursive_processor.close()
            
            logger.info("✓ System closed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error during system close: {e}")

# Demo function
async def demo_recursive_ai_system():
    """Demonstrate the complete recursive AI system"""
    # Create configuration
    config = RecursiveAIConfig(
        max_recursion_depth=3,
        embedding_dimension=128,
        enable_visualization=True
    )
    
    # Initialize system
    system = RecursiveAISystem(config)
    
    if not await system.initialize():
        logger.error("Failed to initialize system")
        return
    
    # Process some queries
    queries = [
        "Quantum computing uses superposition and entanglement to process information",
        "Neural networks learn patterns through recursive weight adjustments",
        "Consciousness emerges from recursive cognitive processes"
    ]
    
    for i, query in enumerate(queries):
        print(f"\n{'='*60}")
        print(f"Query {i+1}: {query}")
        print('='*60)
        
        try:
            result = await system.process_recursive_cognition(query, depth=3)
            
            print(f"Processing time: {result.processing_time:.2f}s")
            print(f"Insights generated: {len(result.insights)}")
            print(f"Emergent patterns: {len(result.emergent_patterns)}")
            
            if result.resonance_field:
                print(f"Resonance strength: {result.resonance_field.resonance_strength:.3f}")
                print(f"Coherence measure: {result.resonance_field.coherence_measure:.3f}")
            
            # Show top insights
            print("\nTop insights:")
            for j, insight in enumerate(result.insights[:3]):
                print(f"  {j+1}. {insight['text']} (coherence: {insight.get('coherence', 0):.3f})")
                
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
    
    # Get system statistics
    print(f"\n{'='*60}")
    print("SYSTEM STATISTICS")
    print('='*60)
    stats = await system.get_system_statistics()
    print(json.dumps(stats, indent=2))
    
    # Visualize system state
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print('='*60)
    await system.visualize_system_state("recursive_ai_demo")
    
    # Close system
    await system.close()

if __name__ == "__main__":
    asyncio.run(demo_recursive_ai_system())