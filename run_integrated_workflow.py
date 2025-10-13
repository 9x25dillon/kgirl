#!/usr/bin/env python3
"""
Integrated Workflow Runner
==========================

Demonstrates the full integration of:
- LFM2-8B-A1B (local LLM)
- Numbskull embedding pipeline
- Dual LLM orchestration

This script provides a complete example of how to use the
numbskull-enhanced orchestrator in production.

Usage:
    python run_integrated_workflow.py
    python run_integrated_workflow.py --config config_lfm2.json
    python run_integrated_workflow.py --query "Your question here"

Author: Assistant
License: MIT
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

from numbskull_dual_orchestrator import (
    create_numbskull_orchestrator,
    NUMBSKULL_AVAILABLE
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config_lfm2.json") -> Dict[str, Any]:
    """Load configuration from JSON file"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"Config file {config_path} not found, using defaults")
        return get_default_config()
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        logger.info(f"✅ Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        "local_llm": {
            "base_url": "http://127.0.0.1:8080",
            "mode": "llama-cpp",
            "model": "LFM2-8B-A1B",
            "timeout": 120,
            "max_retries": 3
        },
        "resource_llm": None,  # Use local fallback
        "orchestrator_settings": {
            "temperature": 0.7,
            "max_tokens": 512,
            "style": "concise",
            "use_numbskull": True,
            "use_semantic": True,
            "use_mathematical": True,
            "use_fractal": True,
            "fusion_method": "weighted_average",
            "embedding_enhancement": "metadata"
        },
        "numbskull_config": {
            "use_semantic": True,
            "use_mathematical": True,
            "use_fractal": True,
            "fusion_method": "weighted_average",
            "semantic_weight": 0.4,
            "mathematical_weight": 0.3,
            "fractal_weight": 0.3,
            "parallel_processing": True,
            "cache_embeddings": True
        }
    }


async def run_single_query(
    orchestrator,
    query: str,
    resource_paths: List[str] = None,
    inline_resources: List[str] = None
) -> Dict[str, Any]:
    """Run a single query through the integrated workflow"""
    
    resource_paths = resource_paths or []
    inline_resources = inline_resources or []
    
    logger.info("=" * 80)
    logger.info("RUNNING INTEGRATED WORKFLOW")
    logger.info("=" * 80)
    logger.info(f"Query: {query}")
    logger.info(f"Resource paths: {resource_paths}")
    logger.info(f"Inline resources: {len(inline_resources)} item(s)")
    logger.info("-" * 80)
    
    try:
        # Run the orchestration
        result = await orchestrator.run_with_embeddings(
            user_prompt=query,
            resource_paths=resource_paths,
            inline_resources=inline_resources
        )
        
        logger.info("=" * 80)
        logger.info("RESULT")
        logger.info("=" * 80)
        
        # Display summary
        logger.info("\n--- RESOURCE SUMMARY ---")
        logger.info(result["summary"])
        
        # Display embedding info
        if result.get("embedding_result"):
            embedding_meta = result["embedding_result"]["metadata"]
            logger.info("\n--- EMBEDDING ANALYSIS ---")
            logger.info(f"Components: {', '.join(embedding_meta['components_used'])}")
            logger.info(f"Dimension: {embedding_meta['embedding_dim']}")
            logger.info(f"Processing time: {embedding_meta['processing_time']:.3f}s")
            logger.info(f"Cached: {result['embedding_result'].get('cached', False)}")
        
        # Display statistics
        if result.get("embedding_stats"):
            stats = result["embedding_stats"]
            logger.info("\n--- EMBEDDING STATISTICS ---")
            logger.info(f"Total embeddings: {stats['total_embeddings']}")
            logger.info(f"Cache hits: {stats['cache_hits']}")
            logger.info(f"Cache size: {stats['cache_size']}")
            logger.info(f"Avg embedding time: {stats.get('avg_embedding_time', 0):.3f}s")
            logger.info(f"Cache hit rate: {stats.get('cache_hit_rate', 0):.2%}")
            
            if stats.get("components_used"):
                logger.info("Components usage:")
                for comp, count in stats["components_used"].items():
                    logger.info(f"  - {comp}: {count}")
        
        # Display final answer
        logger.info("\n--- FINAL ANSWER (LFM2-8B-A1B) ---")
        logger.info(result["final"])
        
        logger.info("=" * 80)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Query failed: {e}", exc_info=True)
        raise


async def run_demo_suite(orchestrator):
    """Run a suite of demonstration queries"""
    
    logger.info("\n" + "=" * 80)
    logger.info("DEMO SUITE: Testing Integrated Workflow")
    logger.info("=" * 80 + "\n")
    
    demos = [
        {
            "name": "Simple Text Query",
            "query": "What are the main components of this system?",
            "resources": ["README.md"],
            "inline": ["This system integrates AI embeddings with LLM orchestration."]
        },
        {
            "name": "Mathematical Expression",
            "query": "Analyze the mathematical complexity of the algorithm",
            "resources": [],
            "inline": ["Algorithm: f(x) = x^2 + 2x + 1, complexity O(n log n)"]
        },
        {
            "name": "Multi-Resource Query",
            "query": "Summarize the key features and architecture",
            "resources": ["README.md", "requirements.txt"],
            "inline": ["Focus on: embeddings, orchestration, and LLM integration"]
        }
    ]
    
    results = []
    
    for i, demo in enumerate(demos, 1):
        logger.info(f"\n--- DEMO {i}/{len(demos)}: {demo['name']} ---\n")
        
        try:
            result = await run_single_query(
                orchestrator,
                query=demo["query"],
                resource_paths=demo["resources"],
                inline_resources=demo["inline"]
            )
            results.append({
                "demo": demo["name"],
                "success": True,
                "result": result
            })
        except Exception as e:
            logger.error(f"Demo {i} failed: {e}")
            results.append({
                "demo": demo["name"],
                "success": False,
                "error": str(e)
            })
        
        # Brief pause between demos
        await asyncio.sleep(1)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DEMO SUITE SUMMARY")
    logger.info("=" * 80)
    
    successful = sum(1 for r in results if r["success"])
    logger.info(f"Completed: {successful}/{len(results)} demos")
    
    for result in results:
        status = "✅" if result["success"] else "❌"
        logger.info(f"{status} {result['demo']}")
    
    return results


async def interactive_mode(orchestrator):
    """Run in interactive mode for testing"""
    
    logger.info("\n" + "=" * 80)
    logger.info("INTERACTIVE MODE")
    logger.info("=" * 80)
    logger.info("Enter queries to test the integrated workflow.")
    logger.info("Commands:")
    logger.info("  - 'quit' or 'exit': Exit interactive mode")
    logger.info("  - 'stats': Show embedding statistics")
    logger.info("  - 'clear': Clear embedding cache")
    logger.info("=" * 80 + "\n")
    
    while True:
        try:
            query = input("\nEnter query: ").strip()
            
            if query.lower() in ['quit', 'exit']:
                logger.info("Exiting interactive mode...")
                break
            
            if query.lower() == 'stats':
                stats = orchestrator.get_embedding_stats()
                logger.info(f"\nEmbedding Statistics:\n{json.dumps(stats, indent=2)}")
                continue
            
            if query.lower() == 'clear':
                orchestrator.clear_embedding_cache()
                logger.info("✅ Embedding cache cleared")
                continue
            
            if not query:
                continue
            
            # Ask for optional resources
            resource_input = input("Resource paths (comma-separated, or press Enter): ").strip()
            resource_paths = [p.strip() for p in resource_input.split(',')] if resource_input else []
            
            inline_input = input("Inline context (or press Enter): ").strip()
            inline_resources = [inline_input] if inline_input else []
            
            # Run query
            await run_single_query(
                orchestrator,
                query=query,
                resource_paths=resource_paths,
                inline_resources=inline_resources
            )
            
        except KeyboardInterrupt:
            logger.info("\n\nInterrupted. Exiting...")
            break
        except EOFError:
            logger.info("\n\nEOF received. Exiting...")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}", exc_info=True)


async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Run integrated LFM2 + Numbskull + Dual LLM workflow"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config_lfm2.json',
        help='Path to configuration file (default: config_lfm2.json)'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Single query to run (skips demo suite)'
    )
    parser.add_argument(
        '--resources',
        type=str,
        nargs='+',
        help='Resource file paths'
    )
    parser.add_argument(
        '--inline',
        type=str,
        help='Inline context/resources'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo suite'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    args = parser.parse_args()
    
    # Check numbskull availability
    if not NUMBSKULL_AVAILABLE:
        logger.error("❌ Numbskull pipeline not available!")
        logger.error("Please ensure /home/kill/numbskull is accessible and contains the embedding pipeline.")
        sys.exit(1)
    
    logger.info("✅ Numbskull pipeline available")
    
    # Load configuration
    config = load_config(args.config)
    
    # Create orchestrator
    logger.info("Initializing numbskull-enhanced orchestrator...")
    
    try:
        orchestrator = create_numbskull_orchestrator(
            local_configs=[config["local_llm"]],
            remote_config=config.get("resource_llm"),
            settings=config.get("orchestrator_settings", {}),
            numbskull_config=config.get("numbskull_config")
        )
        logger.info("✅ Orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize orchestrator: {e}", exc_info=True)
        sys.exit(1)
    
    try:
        # Run based on arguments
        if args.interactive:
            await interactive_mode(orchestrator)
        elif args.query:
            # Single query mode
            resource_paths = args.resources or []
            inline_resources = [args.inline] if args.inline else []
            
            await run_single_query(
                orchestrator,
                query=args.query,
                resource_paths=resource_paths,
                inline_resources=inline_resources
            )
        elif args.demo:
            # Demo suite mode
            await run_demo_suite(orchestrator)
        else:
            # Default: run demo suite
            logger.info("No mode specified, running demo suite...")
            logger.info("Use --help for options\n")
            await run_demo_suite(orchestrator)
        
        logger.info("\n✅ Workflow completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Workflow failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Cleanup
        try:
            await orchestrator.close()
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")


if __name__ == "__main__":
    asyncio.run(main())

