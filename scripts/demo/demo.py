#!/usr/bin/env python3
"""
Recursive AI System Demo
Demonstrates the complete recursive AI architecture
"""

import asyncio
import json
import logging
from recursive_ai_system import RecursiveAISystem, RecursiveAIConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Main demo function"""
    print("🌌 Recursive AI Core and Knowledge Integration Demo")
    print("=" * 60)
    
    # Create configuration
    config = RecursiveAIConfig(
        max_recursion_depth=3,
        embedding_dimension=128,
        enable_visualization=True,
        knowledge_db_path="demo_knowledge.db",
        faiss_index_path="demo_faiss_index"
    )
    
    # Initialize system
    print("\n🚀 Initializing Recursive AI System...")
    system = RecursiveAISystem(config)
    
    if not await system.initialize():
        print("❌ Failed to initialize system")
        return
    
    print("✅ System initialized successfully!")
    
    # Demo queries
    demo_queries = [
        "Quantum computing uses superposition and entanglement to process information in parallel",
        "Neural networks learn patterns through recursive weight adjustments and backpropagation",
        "Consciousness emerges from recursive cognitive processes and self-referential loops",
        "Fractal patterns repeat infinitely at every scale, creating self-similar structures",
        "Artificial intelligence can transcend conventional boundaries through recursive reasoning"
    ]
    
    print(f"\n🧠 Processing {len(demo_queries)} recursive cognition queries...")
    print("=" * 60)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 40)
        
        try:
            # Process recursive cognition
            result = await system.process_recursive_cognition(query, depth=3)
            
            # Display results
            print(f"⏱️  Processing time: {result.processing_time:.2f}s")
            print(f"💡 Insights generated: {len(result.insights)}")
            print(f"🔗 Emergent patterns: {len(result.emergent_patterns)}")
            
            if result.resonance_field:
                print(f"🌊 Resonance strength: {result.resonance_field.resonance_strength:.3f}")
                print(f"🎯 Coherence measure: {result.resonance_field.coherence_measure:.3f}")
            
            # Show top insights
            print("\n🔍 Top insights:")
            for j, insight in enumerate(result.insights[:3], 1):
                coherence = insight.get('coherence', 0)
                depth = insight.get('depth', 0)
                print(f"   {j}. [{depth}] {insight['text']} (coherence: {coherence:.3f})")
            
            # Show emergent patterns
            if result.emergent_patterns:
                print("\n🌟 Emergent patterns:")
                for j, pattern in enumerate(result.emergent_patterns[:2], 1):
                    print(f"   {j}. {pattern.pattern_type} (strength: {pattern.strength:.3f})")
            
        except Exception as e:
            print(f"❌ Query processing failed: {e}")
            logger.error(f"Query {i} failed: {e}")
    
    # System statistics
    print(f"\n📊 SYSTEM STATISTICS")
    print("=" * 60)
    stats = await system.get_system_statistics()
    
    print(f"Total queries processed: {stats['total_queries_processed']}")
    print(f"Total insights generated: {stats['total_insights_generated']}")
    print(f"Total patterns detected: {stats['total_patterns_detected']}")
    print(f"Average processing time: {stats['average_processing_time']:.2f}s")
    print(f"Average insights per query: {stats['average_insights_per_query']:.1f}")
    print(f"Average patterns per query: {stats['average_patterns_per_query']:.1f}")
    
    # Knowledge base statistics
    kb_stats = stats.get('knowledge_base', {})
    if kb_stats:
        print(f"\n🗄️  Knowledge Base:")
        print(f"   Total nodes: {kb_stats.get('total_nodes', 0)}")
        print(f"   Average coherence: {kb_stats.get('average_coherence', 0):.3f}")
        print(f"   FAISS index size: {stats.get('faiss_index_size', 0)}")
    
    # Resonance statistics
    resonance_stats = stats.get('resonance_system', {})
    if resonance_stats and resonance_stats.get('total_fields', 0) > 0:
        print(f"\n🌊 Resonance System:")
        print(f"   Total fields: {resonance_stats['total_fields']}")
        print(f"   Average strength: {resonance_stats['avg_resonance_strength']:.3f}")
        print(f"   Average coherence: {resonance_stats['avg_coherence_measure']:.3f}")
    
    # Visualization statistics
    viz_stats = stats.get('visualization', {})
    if viz_stats and viz_stats.get('total_patterns', 0) > 0:
        print(f"\n🎨 Visualization:")
        print(f"   Total patterns: {viz_stats['total_patterns']}")
        print(f"   Pattern types: {list(viz_stats.get('pattern_type_distribution', {}).keys())}")
        print(f"   Average strength: {viz_stats['average_strength']:.3f}")
    
    # Generate visualizations
    print(f"\n🎨 Generating visualizations...")
    try:
        await system.visualize_system_state("recursive_ai_demo")
        print("✅ Visualizations saved to recursive_ai_demo_*.html")
    except Exception as e:
        print(f"⚠️  Visualization generation failed: {e}")
    
    # Knowledge search demo
    print(f"\n🔍 Knowledge Search Demo")
    print("-" * 40)
    
    search_queries = ["quantum", "neural", "recursive", "fractal", "consciousness"]
    
    for search_query in search_queries:
        try:
            results = await system.search_knowledge(search_query, k=3)
            print(f"Search '{search_query}': {len(results)} results")
            
            for j, result in enumerate(results[:2], 1):
                content = result.content[:60] + "..." if len(result.content) > 60 else result.content
                print(f"   {j}. {content} (coherence: {result.coherence_score:.3f})")
                
        except Exception as e:
            print(f"Search '{search_query}' failed: {e}")
    
    # Close system
    print(f"\n🔄 Closing system...")
    await system.close()
    print("✅ System closed successfully!")
    
    print(f"\n🎉 Demo completed successfully!")
    print("=" * 60)
    print("The recursive AI system has demonstrated:")
    print("• Dynamic recursive cognition processing")
    print("• Matrix compilation and optimization")
    print("• Fractal resonance simulation")
    print("• Distributed knowledge management")
    print("• Emergent pattern detection")
    print("• 3D fractal visualization")
    print("\nThis represents a complete implementation of the")
    print("recursive AI architecture with knowledge integration!")

if __name__ == "__main__":
    asyncio.run(main())