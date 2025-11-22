#!/usr/bin/env python3
"""
Integrated WaveCaster Runner
============================

Complete integration of Enhanced WaveCaster with:
- Numbskull hybrid embeddings
- Dual LLM orchestration
- All 10 component adapters
- Signal processing
- Complete cognitive architecture

This brings together EVERYTHING into a unified wavecasting system.

Usage:
    python integrated_wavecaster_runner.py --text "Your message"
    python integrated_wavecaster_runner.py --llm --prompt "Generate content"
    python integrated_wavecaster_runner.py --demo

Author: Assistant
License: MIT
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add numbskull to path
numbskull_path = Path("/home/kill/numbskull")
if numbskull_path.exists() and str(numbskull_path) not in sys.path:
    sys.path.insert(0, str(numbskull_path))

# Import enhanced wavecaster
from enhanced_wavecaster import EnhancedWaveCaster, create_default_config

# Import our integrated components
from numbskull_dual_orchestrator import create_numbskull_orchestrator
from neuro_symbolic_numbskull_adapter import NeuroSymbolicNumbskullAdapter
from signal_processing_numbskull_adapter import SignalProcessingNumbskullAdapter
from unified_cognitive_orchestrator import UnifiedCognitiveOrchestrator

import signal_processing as dsp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegratedWaveCasterSystem:
    """
    Complete integrated system combining:
    - Enhanced WaveCaster
    - Numbskull embeddings
    - Dual LLM orchestration
    - All component adapters
    - Full cognitive architecture
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize integrated system"""
        logger.info("=" * 70)
        logger.info("INTEGRATED WAVECASTER SYSTEM INITIALIZING")
        logger.info("=" * 70)
        
        self.config = config or self._default_config()
        
        # 1. Enhanced WaveCaster (base system)
        logger.info("\n1. Initializing Enhanced WaveCaster...")
        try:
            self.wavecaster = EnhancedWaveCaster(self.config.get("wavecaster", {}))
            logger.info("   ✅ Enhanced WaveCaster ready")
        except Exception as e:
            logger.warning(f"   ⚠️  WaveCaster init failed: {e}")
            self.wavecaster = None
        
        # 2. Numbskull + Dual LLM Orchestrator
        logger.info("2. Initializing Numbskull + Dual LLM...")
        try:
            self.numbskull_orchestrator = create_numbskull_orchestrator(
                local_configs=self.config.get("local_llm", [{"base_url": "http://127.0.0.1:8080", "mode": "llama-cpp"}]),
                remote_config=self.config.get("remote_llm"),
                settings=self.config.get("orchestrator_settings", {}),
                numbskull_config=self.config.get("numbskull", {})
            )
            logger.info("   ✅ Numbskull + Dual LLM ready")
        except Exception as e:
            logger.warning(f"   ⚠️  Numbskull orchestrator init failed: {e}")
            self.numbskull_orchestrator = None
        
        # 3. Neuro-Symbolic Adapter
        logger.info("3. Initializing Neuro-Symbolic Adapter...")
        try:
            self.neuro_symbolic = NeuroSymbolicNumbskullAdapter(
                use_numbskull=True,
                numbskull_config=self.config.get("numbskull", {})
            )
            logger.info("   ✅ Neuro-Symbolic adapter ready")
        except Exception as e:
            logger.warning(f"   ⚠️  Neuro-Symbolic init failed: {e}")
            self.neuro_symbolic = None
        
        # 4. Signal Processing Adapter
        logger.info("4. Initializing Signal Processing Adapter...")
        try:
            self.signal_adapter = SignalProcessingNumbskullAdapter(
                use_numbskull=True,
                numbskull_config=self.config.get("numbskull", {})
            )
            logger.info("   ✅ Signal Processing adapter ready")
        except Exception as e:
            logger.warning(f"   ⚠️  Signal adapter init failed: {e}")
            self.signal_adapter = None
        
        logger.info("\n" + "=" * 70)
        logger.info("INTEGRATED WAVECASTER SYSTEM READY")
        logger.info("=" * 70)
        self._print_status()
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "local_llm": [{
                "base_url": "http://127.0.0.1:8080",
                "mode": "llama-cpp",
                "model": "LFM2-8B-A1B",
                "timeout": 120
            }],
            "numbskull": {
                "use_semantic": False,
                "use_mathematical": False,
                "use_fractal": True,
                "fusion_method": "weighted_average"
            },
            "orchestrator_settings": {
                "temperature": 0.7,
                "max_tokens": 512,
                "style": "concise",
                "use_numbskull": True
            },
            "wavecaster": {}
        }
    
    def _print_status(self):
        """Print system status"""
        logger.info("\n🎯 System Components:")
        logger.info(f"  Enhanced WaveCaster:      {'✅ Active' if self.wavecaster else '❌ Inactive'}")
        logger.info(f"  Numbskull Orchestrator:   {'✅ Active' if self.numbskull_orchestrator else '❌ Inactive'}")
        logger.info(f"  Neuro-Symbolic Adapter:   {'✅ Active' if self.neuro_symbolic else '❌ Inactive'}")
        logger.info(f"  Signal Processing:        {'✅ Active' if self.signal_adapter else '❌ Inactive'}")
        logger.info("")
    
    async def run_complete_wavecaster_workflow(
        self,
        text: Optional[str] = None,
        llm_prompt: Optional[str] = None,
        resource_files: List[str] = None,
        inline_resources: List[str] = None,
        output_dir: Path = Path("wavecaster_output")
    ) -> Dict[str, Any]:
        """
        Complete integrated wavecaster workflow
        
        Args:
            text: Direct text to cast (or use llm_prompt)
            llm_prompt: LLM prompt to generate text
            resource_files: Files for LLM context
            inline_resources: Inline resources for LLM
            output_dir: Output directory
        
        Returns:
            Complete workflow results
        """
        logger.info("\n" + "=" * 70)
        logger.info("INTEGRATED WAVECASTER WORKFLOW")
        logger.info("=" * 70)
        
        workflow_results = {
            "stages": {},
            "final_output": None,
            "signals_generated": False
        }
        
        content_to_cast = text
        
        # Stage 1: Generate content with LLM if needed
        if llm_prompt and self.numbskull_orchestrator:
            logger.info("\n--- Stage 1: LLM Content Generation with Embeddings ---")
            try:
                llm_result = await self.numbskull_orchestrator.run_with_embeddings(
                    user_prompt=llm_prompt,
                    resource_paths=resource_files or [],
                    inline_resources=inline_resources or []
                )
                
                content_to_cast = llm_result.get("final", "")
                workflow_results["stages"]["llm_generation"] = {
                    "content_length": len(content_to_cast),
                    "embeddings_used": llm_result.get("numbskull_enabled", False),
                    "summary_length": len(llm_result.get("summary", ""))
                }
                
                logger.info(f"✅ Generated {len(content_to_cast)} characters with LLM")
                
            except Exception as e:
                logger.warning(f"⚠️  LLM generation failed: {e}")
                content_to_cast = llm_prompt  # Fallback to prompt as content
        elif llm_prompt:
            logger.info("⚠️  No LLM orchestrator, using prompt as direct text")
            content_to_cast = llm_prompt
        
        if not content_to_cast:
            logger.error("❌ No content to cast!")
            return workflow_results
        
        logger.info(f"\nContent to cast: {content_to_cast[:100]}...")
        
        # Stage 2: Neuro-Symbolic Analysis with Embeddings
        if self.neuro_symbolic:
            logger.info("\n--- Stage 2: Neuro-Symbolic Analysis ---")
            try:
                analysis = await self.neuro_symbolic.analyze_with_embeddings(
                    content_to_cast,
                    enable_all_modules=True
                )
                
                workflow_results["stages"]["neuro_symbolic"] = {
                    "modules_analyzed": len(analysis["modules"]),
                    "insights": len(analysis["insights"]),
                    "recommendations": analysis["recommendations"]
                }
                
                logger.info(f"✅ Analyzed with {len(analysis['modules'])} modules")
                
            except Exception as e:
                logger.warning(f"⚠️  Neuro-symbolic analysis failed: {e}")
        
        # Stage 3: Embedding-Guided Modulation Selection
        if self.signal_adapter:
            logger.info("\n--- Stage 3: Modulation Selection ---")
            try:
                scheme, selection_analysis = await self.signal_adapter.select_modulation_from_embedding(
                    content_to_cast
                )
                
                workflow_results["stages"]["modulation_selection"] = {
                    "scheme": scheme.name,
                    "method": selection_analysis.get("method", "default"),
                    "reason": selection_analysis.get("reason", "N/A")
                }
                
                logger.info(f"✅ Selected modulation: {scheme.name}")
                logger.info(f"   Reason: {selection_analysis.get('reason', 'N/A')}")
                
            except Exception as e:
                logger.warning(f"⚠️  Modulation selection failed: {e}")
                scheme = dsp.ModulationScheme.QPSK  # Default
        else:
            scheme = dsp.ModulationScheme.QPSK
            logger.info("⚠️  Using default QPSK modulation")
        
        # Stage 4: Signal Generation and Casting
        logger.info("\n--- Stage 4: Signal Generation ---")
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Use wavecaster if available, otherwise use signal adapter
            if self.wavecaster:
                result = self.wavecaster.cast_text_direct(
                    text=content_to_cast,
                    scheme=scheme,
                    output_dir=output_dir,
                    use_adaptive=True
                )
                
                workflow_results["stages"]["signal_generation"] = {
                    "method": "enhanced_wavecaster",
                    "paths": result.get("paths", {}),
                    "config": result.get("config", {})
                }
                
                logger.info("✅ Signals generated with Enhanced WaveCaster")
                
            elif self.signal_adapter:
                result = await self.signal_adapter.encode_embedding_to_signal(
                    content_to_cast,
                    output_dir=output_dir
                )
                
                workflow_results["stages"]["signal_generation"] = {
                    "method": "signal_adapter",
                    "signal_generated": result.get("signal_generated", False),
                    "modulation": result.get("modulation_scheme", "N/A")
                }
                
                logger.info("✅ Signals generated with Signal Adapter")
            
            workflow_results["signals_generated"] = True
            
        except Exception as e:
            logger.error(f"❌ Signal generation failed: {e}")
            workflow_results["stages"]["signal_generation"] = {"error": str(e)}
        
        # Compile final output
        workflow_results["final_output"] = {
            "content": content_to_cast,
            "content_length": len(content_to_cast),
            "modulation_scheme": scheme.name if isinstance(scheme, dsp.ModulationScheme) else str(scheme),
            "output_directory": str(output_dir),
            "stages_completed": list(workflow_results["stages"].keys())
        }
        
        logger.info("\n" + "=" * 70)
        logger.info("INTEGRATED WAVECASTER WORKFLOW COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Stages completed: {len(workflow_results['stages'])}")
        logger.info(f"Signals generated: {workflow_results['signals_generated']}")
        
        return workflow_results
    
    async def close(self):
        """Clean up resources"""
        if self.neuro_symbolic:
            await self.neuro_symbolic.close()
        if self.signal_adapter:
            await self.signal_adapter.close()
        if self.numbskull_orchestrator:
            await self.numbskull_orchestrator.close()
        logger.info("✅ Integrated WaveCaster system closed")


async def demo_integrated_wavecaster():
    """Comprehensive demo of integrated wavecaster system"""
    
    print("\n" + "=" * 70)
    print("INTEGRATED WAVECASTER SYSTEM DEMO")
    print("Complete LiMp + Numbskull + WaveCaster Integration")
    print("=" * 70)
    
    # Create integrated system
    system = IntegratedWaveCasterSystem()
    
    # Demo scenarios
    scenarios = [
        {
            "name": "Direct Text Casting",
            "text": "Emergency communication: All systems operational. Network stability confirmed.",
            "llm_prompt": None,
            "output_dir": Path("output/demo1_direct")
        },
        {
            "name": "Simple Message",
            "text": "Testing integrated wavecaster with Numbskull embeddings and dual LLM orchestration.",
            "llm_prompt": None,
            "output_dir": Path("output/demo2_simple")
        },
        {
            "name": "Mathematical Content",
            "text": "Solve the quadratic equation: x^2 - 5x + 6 = 0. Solutions are x = 2 and x = 3.",
            "llm_prompt": None,
            "output_dir": Path("output/demo3_math")
        }
    ]
    
    # Run scenarios
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*70}")
        print(f"SCENARIO {i}/{len(scenarios)}: {scenario['name']}")
        print(f"{'='*70}")
        
        result = await system.run_complete_wavecaster_workflow(
            text=scenario["text"],
            llm_prompt=scenario["llm_prompt"],
            output_dir=scenario["output_dir"]
        )
        
        print(f"\n📊 Results:")
        print(f"  Stages completed: {len(result['stages'])}")
        print(f"  Signals generated: {result['signals_generated']}")
        print(f"  Content length: {result['final_output']['content_length']} chars")
        print(f"  Modulation: {result['final_output']['modulation_scheme']}")
        
        if result.get("stages", {}).get("neuro_symbolic"):
            ns = result["stages"]["neuro_symbolic"]
            print(f"  Neuro-Symbolic: {ns['modules_analyzed']} modules, {ns['insights']} insights")
    
    # Cleanup
    await system.close()
    
    print(f"\n{'='*70}")
    print("✅ INTEGRATED WAVECASTER DEMO COMPLETE")
    print(f"{'='*70}")
    print("\nCheck output/ directory for generated signals!")
    print("(Note: Full signal generation requires all services running)")


async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Integrated WaveCaster with complete LiMp + Numbskull integration"
    )
    parser.add_argument(
        '--text',
        type=str,
        help='Direct text to cast into signals'
    )
    parser.add_argument(
        '--llm',
        action='store_true',
        help='Use LLM to generate content'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        help='LLM prompt for content generation'
    )
    parser.add_argument(
        '--resources',
        type=str,
        nargs='+',
        help='Resource files for LLM context'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='wavecaster_output',
        help='Output directory'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demonstration scenarios'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    
    # Create system
    system = IntegratedWaveCasterSystem(config)
    
    try:
        if args.demo:
            # Run demo
            await demo_integrated_wavecaster()
        elif args.text or args.prompt:
            # Run single workflow
            result = await system.run_complete_wavecaster_workflow(
                text=args.text,
                llm_prompt=args.prompt if args.llm else None,
                resource_files=args.resources or [],
                output_dir=Path(args.output)
            )
            
            print("\n" + "=" * 70)
            print("WORKFLOW RESULTS")
            print("=" * 70)
            print(json.dumps(result, indent=2, default=str))
        else:
            # Show help
            parser.print_help()
            print("\n💡 Quick start:")
            print("  python integrated_wavecaster_runner.py --demo")
            print("  python integrated_wavecaster_runner.py --text 'Your message'")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    finally:
        await system.close()


if __name__ == "__main__":
    asyncio.run(main())

