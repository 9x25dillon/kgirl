#!/usr/bin/env python3
"""
Enhanced Dual LLM WaveCaster with TA ULS Integration
====================================================

This is the main integration module that combines:
- TA ULS Transformer architecture
- Dual LLM orchestration system  
- Neuro-symbolic adaptive reflective engine
- Advanced signal processing and modulation
- Comprehensive CLI interface

Author: Assistant
License: MIT
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import our modules
from tauls_transformer import TAULSLanguageModel, demo_tauls_model
from dual_llm_orchestrator import (
    DualLLMOrchestrator, HTTPConfig, OrchestratorSettings,
    LocalLLM, ResourceLLM, create_orchestrator
)
from neuro_symbolic_engine import (
    MirrorCastEngine, AdaptiveLinkPlanner, 
    demo_neuro_symbolic_engine
)
from signal_processing import (
    ModulationScheme, FEC, ModConfig, FrameConfig, SecurityConfig,
    full_process_and_save, demo_signal_processing, play_audio
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("enhanced_wavecaster")

class EnhancedWaveCaster:
    """Main class integrating all components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.mirror_engine = MirrorCastEngine()
        self.adaptive_planner = AdaptiveLinkPlanner(
            db_path=config.get("db_path", "reflective_db.json")
        )
        
        # Initialize orchestrator if LLM configs provided
        self.orchestrator = None
        if "llm" in config:
            self.orchestrator = self._create_orchestrator(config["llm"])
    
    def _create_orchestrator(self, llm_config: Dict[str, Any]) -> Optional[DualLLMOrchestrator]:
        """Create LLM orchestrator from configuration"""
        try:
            local_configs = llm_config.get("local", [])
            remote_config = llm_config.get("remote")
            settings = llm_config.get("settings", {})
            
            return create_orchestrator(local_configs, remote_config, settings)
        except Exception as e:
            logger.error(f"Failed to create orchestrator: {e}")
            return None
    
    def cast_text_direct(
        self, 
        text: str,
        scheme: ModulationScheme,
        output_dir: Path,
        use_adaptive: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Direct text to waveform casting"""
        
        logger.info(f"Direct casting: {len(text)} characters using {scheme.name}")
        
        # Neuro-symbolic analysis
        analysis = self.mirror_engine.cast(text)
        
        # Configuration
        mcfg = ModConfig(**kwargs.get("modulation", {}))
        fcfg = FrameConfig(**kwargs.get("framing", {}))
        sec = SecurityConfig(**kwargs.get("security", {}))
        fec_scheme = FEC[kwargs.get("fec", "HAMMING74")]
        
        # Adaptive planning
        if use_adaptive:
            config_dict, explanation = self.adaptive_planner.plan(text, analysis)
            # Update modulation config based on adaptive planning
            if "symbol_rate" in config_dict:
                mcfg.symbol_rate = config_dict["symbol_rate"]
            logger.info(f"Adaptive planning: {explanation}")
        else:
            explanation = "No adaptive planning used"
        
        # Process and save
        paths = full_process_and_save(
            text=text,
            outdir=output_dir,
            scheme=scheme,
            mcfg=mcfg,
            fcfg=fcfg,
            sec=sec,
            fec_scheme=fec_scheme,
            want_wav=kwargs.get("want_wav", True),
            want_iq=kwargs.get("want_iq", False),
            title=f"Enhanced WaveCaster - {scheme.name}"
        )
        
        return {
            "text": text,
            "analysis": analysis,
            "explanation": explanation,
            "config": {
                "modulation": mcfg.__dict__,
                "framing": fcfg.__dict__,
                "security": sec.__dict__,
                "fec": fec_scheme.name
            },
            "paths": {
                "wav": str(paths.wav) if paths.wav else None,
                "iq": str(paths.iq) if paths.iq else None,
                "meta": str(paths.meta) if paths.meta else None,
                "png": str(paths.png) if paths.png else None
            },
            "processing_time": time.time()
        }
    
    def cast_with_llm(
        self,
        prompt: str,
        resource_files: List[str],
        inline_resources: List[str],
        scheme: ModulationScheme,
        output_dir: Path,
        **kwargs
    ) -> Dict[str, Any]:
        """LLM-orchestrated casting"""
        
        if not self.orchestrator:
            raise RuntimeError("No LLM orchestrator configured")
        
        logger.info(f"LLM orchestration: prompt='{prompt[:50]}...', resources={len(resource_files)}")
        
        # Run dual LLM orchestration
        llm_result = self.orchestrator.run(prompt, resource_files, inline_resources)
        
        # Cast the generated text
        cast_result = self.cast_text_direct(
            text=llm_result["final"],
            scheme=scheme,
            output_dir=output_dir,
            **kwargs
        )
        
        # Combine results
        return {
            **cast_result,
            "llm_orchestration": {
                "prompt": prompt,
                "resource_files": resource_files,
                "summary": llm_result["summary"],
                "final_text": llm_result["final"]
            }
        }
    
    def learn_adaptive(
        self,
        texts: List[str],
        episodes: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """Run adaptive learning episodes"""
        
        logger.info(f"Starting adaptive learning: {episodes} episodes, {len(texts)} texts")
        
        results = []
        
        for episode in range(episodes):
            text = texts[episode % len(texts)]
            
            # Analysis and planning
            analysis = self.mirror_engine.cast(text)
            config_dict, explanation = self.adaptive_planner.plan(text, analysis)
            
            # Simulate transmission (in real implementation, this would be actual modem)
            import numpy as np
            success = np.random.random() > 0.3  # 70% success rate for demo
            
            # Update planner
            self.adaptive_planner.reward_and_record(
                text=text,
                config=config_dict,
                explanation=explanation,
                success=success,
                entropy=analysis["entropy"],
                complexity=analysis["endpoints"]["metadata"]["complexity"],
                harmony=analysis["love"]["harmony_index"]
            )
            
            results.append({
                "episode": episode + 1,
                "text_hash": analysis["endpoints"]["artifact_id"],
                "config": config_dict,
                "success": success,
                "explanation": explanation
            })
            
            if episode % 5 == 0:
                logger.info(f"Episode {episode + 1}/{episodes} complete")
        
        success_rate = sum(r["success"] for r in results) / len(results)
        logger.info(f"Learning complete. Success rate: {success_rate:.1%}")
        
        return {
            "episodes": results,
            "success_rate": success_rate,
            "agent_stats": self.adaptive_planner.agent.get_stats(),
            "db_stats": self.adaptive_planner.db.get_stats()
        }

def create_default_config() -> Dict[str, Any]:
    """Create default configuration"""
    return {
        "db_path": "reflective_db.json",
        "llm": {
            "local": [
                {
                    "base_url": "http://127.0.0.1:8080",
                    "mode": "llama-cpp",
                    "model": "local-model"
                }
            ],
            "remote": {
                "base_url": "https://api.openai.com",
                "api_key": None,  # Set via environment or CLI
                "model": "gpt-4o-mini"
            },
            "settings": {
                "temperature": 0.7,
                "max_tokens": 512,
                "style": "concise"
            }
        },
        "modulation": {
            "sample_rate": 48000,
            "symbol_rate": 1200,
            "amplitude": 0.7
        },
        "framing": {
            "use_crc32": True,
            "use_crc16": False
        },
        "security": {
            "password": None,
            "watermark": None,
            "hmac_key": None
        }
    }

def build_parser() -> argparse.ArgumentParser:
    """Build comprehensive CLI parser"""
    
    parser = argparse.ArgumentParser(
        prog="enhanced_wavecaster",
        description="Enhanced Dual LLM WaveCaster with TA ULS Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct text modulation
  python enhanced_wavecaster.py modulate --text "Hello World" --scheme qpsk --wav
  
  # LLM-orchestrated casting
  python enhanced_wavecaster.py cast --prompt "Summarize the key points" \\
    --resource-file document.txt --scheme ofdm --adaptive
  
  # Adaptive learning
  python enhanced_wavecaster.py learn --episodes 20 --texts "Test message 1" "Test message 2"
  
  # Component demos
  python enhanced_wavecaster.py demo --component tauls
  python enhanced_wavecaster.py demo --component neuro-symbolic
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Commands")
    
    # Common arguments
    def add_common_args(p):
        p.add_argument("--config", type=str, help="Configuration file (JSON)")
        p.add_argument("--output-dir", type=str, default="output", help="Output directory")
        p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    def add_modulation_args(p):
        p.add_argument("--scheme", choices=[s.name.lower() for s in ModulationScheme], 
                      default="qpsk", help="Modulation scheme")
        p.add_argument("--sample-rate", type=int, default=48000)
        p.add_argument("--symbol-rate", type=int, default=1200)
        p.add_argument("--amplitude", type=float, default=0.7)
        p.add_argument("--wav", action="store_true", help="Generate WAV file")
        p.add_argument("--iq", action="store_true", help="Generate IQ file")
        p.add_argument("--play", action="store_true", help="Play audio")
    
    def add_security_args(p):
        p.add_argument("--password", type=str, help="Encryption password")
        p.add_argument("--watermark", type=str, help="Watermark string")
        p.add_argument("--hmac-key", type=str, help="HMAC key")
        p.add_argument("--fec", choices=[f.name.lower() for f in FEC], 
                      default="hamming74", help="FEC scheme")
    
    # Modulate command
    mod_parser = subparsers.add_parser("modulate", help="Direct text modulation")
    add_common_args(mod_parser)
    add_modulation_args(mod_parser)
    add_security_args(mod_parser)
    mod_parser.add_argument("--text", type=str, required=True, help="Text to modulate")
    mod_parser.add_argument("--adaptive", action="store_true", help="Use adaptive planning")
    
    # Cast command (LLM orchestration)
    cast_parser = subparsers.add_parser("cast", help="LLM-orchestrated casting")
    add_common_args(cast_parser)
    add_modulation_args(cast_parser)
    add_security_args(cast_parser)
    cast_parser.add_argument("--prompt", type=str, required=True, help="LLM prompt")
    cast_parser.add_argument("--resource-file", nargs="*", default=[], help="Resource files")
    cast_parser.add_argument("--resource-text", nargs="*", default=[], help="Inline resources")
    cast_parser.add_argument("--adaptive", action="store_true", help="Use adaptive planning")
    
    # LLM configuration
    cast_parser.add_argument("--local-url", type=str, default="http://127.0.0.1:8080")
    cast_parser.add_argument("--local-mode", choices=["openai-chat", "llama-cpp", "textgen-webui"], 
                           default="llama-cpp")
    cast_parser.add_argument("--remote-url", type=str, help="Remote LLM URL")
    cast_parser.add_argument("--remote-key", type=str, help="Remote LLM API key")
    
    # Learn command
    learn_parser = subparsers.add_parser("learn", help="Adaptive learning")
    add_common_args(learn_parser)
    learn_parser.add_argument("--texts", nargs="+", required=True, help="Training texts")
    learn_parser.add_argument("--episodes", type=int, default=10, help="Learning episodes")
    learn_parser.add_argument("--db-path", type=str, default="reflective_db.json")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Component demonstrations")
    add_common_args(demo_parser)
    demo_parser.add_argument("--component", 
                           choices=["tauls", "neuro-symbolic", "signal-processing", "all"],
                           default="all", help="Component to demo")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze text with neuro-symbolic engine")
    add_common_args(analyze_parser)
    analyze_parser.add_argument("--text", type=str, required=True, help="Text to analyze")
    analyze_parser.add_argument("--plot", action="store_true", help="Generate plots")
    
    return parser

def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from file or create default"""
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config {config_path}: {e}")
    
    return create_default_config()

def update_config_from_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Update configuration with command line arguments"""
    
    # Modulation settings
    if hasattr(args, 'sample_rate'):
        config["modulation"]["sample_rate"] = args.sample_rate
    if hasattr(args, 'symbol_rate'):
        config["modulation"]["symbol_rate"] = args.symbol_rate
    if hasattr(args, 'amplitude'):
        config["modulation"]["amplitude"] = args.amplitude
    
    # Security settings
    if hasattr(args, 'password') and args.password:
        config["security"]["password"] = args.password
    if hasattr(args, 'watermark') and args.watermark:
        config["security"]["watermark"] = args.watermark
    if hasattr(args, 'hmac_key') and args.hmac_key:
        config["security"]["hmac_key"] = args.hmac_key
    
    # LLM settings
    if hasattr(args, 'local_url'):
        config["llm"]["local"][0]["base_url"] = args.local_url
    if hasattr(args, 'local_mode'):
        config["llm"]["local"][0]["mode"] = args.local_mode
    if hasattr(args, 'remote_url') and args.remote_url:
        config["llm"]["remote"]["base_url"] = args.remote_url
    if hasattr(args, 'remote_key') and args.remote_key:
        config["llm"]["remote"]["api_key"] = args.remote_key
    
    return config

def cmd_modulate(args: argparse.Namespace) -> int:
    """Handle modulate command"""
    config = load_config(args.config)
    config = update_config_from_args(config, args)
    
    wavecaster = EnhancedWaveCaster(config)
    
    try:
        result = wavecaster.cast_text_direct(
            text=args.text,
            scheme=ModulationScheme[args.scheme.upper()],
            output_dir=Path(args.output_dir),
            use_adaptive=args.adaptive,
            modulation=config["modulation"],
            framing=config["framing"],
            security=config["security"],
            fec=args.fec.upper(),
            want_wav=args.wav or not args.iq,
            want_iq=args.iq
        )
        
        print(json.dumps(result, indent=2, default=str))
        
        # Play audio if requested
        if args.play and result["paths"]["wav"]:
            try:
                import soundfile as sf
                data, sr = sf.read(result["paths"]["wav"])
                play_audio(data, sr)
            except Exception as e:
                logger.warning(f"Audio playback failed: {e}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Modulation failed: {e}")
        return 1

def cmd_cast(args: argparse.Namespace) -> int:
    """Handle cast command"""
    config = load_config(args.config)
    config = update_config_from_args(config, args)
    
    wavecaster = EnhancedWaveCaster(config)
    
    try:
        result = wavecaster.cast_with_llm(
            prompt=args.prompt,
            resource_files=args.resource_file,
            inline_resources=args.resource_text,
            scheme=ModulationScheme[args.scheme.upper()],
            output_dir=Path(args.output_dir),
            modulation=config["modulation"],
            framing=config["framing"],
            security=config["security"],
            fec=args.fec.upper(),
            want_wav=args.wav or not args.iq,
            want_iq=args.iq
        )
        
        print(json.dumps(result, indent=2, default=str))
        
        # Play audio if requested
        if args.play and result["paths"]["wav"]:
            try:
                import soundfile as sf
                data, sr = sf.read(result["paths"]["wav"])
                play_audio(data, sr)
            except Exception as e:
                logger.warning(f"Audio playback failed: {e}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Casting failed: {e}")
        return 1

def cmd_learn(args: argparse.Namespace) -> int:
    """Handle learn command"""
    config = load_config(args.config)
    if args.db_path:
        config["db_path"] = args.db_path
    
    wavecaster = EnhancedWaveCaster(config)
    
    try:
        result = wavecaster.learn_adaptive(
            texts=args.texts,
            episodes=args.episodes
        )
        
        print(json.dumps(result, indent=2, default=str))
        return 0
        
    except Exception as e:
        logger.error(f"Learning failed: {e}")
        return 1

def cmd_demo(args: argparse.Namespace) -> int:
    """Handle demo command"""
    
    if args.component in ["tauls", "all"]:
        logger.info("=== TA ULS Transformer Demo ===")
        try:
            demo_tauls_model()
        except Exception as e:
            logger.error(f"TA ULS demo failed: {e}")
    
    if args.component in ["neuro-symbolic", "all"]:
        logger.info("=== Neuro-Symbolic Engine Demo ===")
        try:
            demo_neuro_symbolic_engine()
        except Exception as e:
            logger.error(f"Neuro-symbolic demo failed: {e}")
    
    if args.component in ["signal-processing", "all"]:
        logger.info("=== Signal Processing Demo ===")
        try:
            demo_signal_processing()
        except Exception as e:
            logger.error(f"Signal processing demo failed: {e}")
    
    return 0

def cmd_analyze(args: argparse.Namespace) -> int:
    """Handle analyze command"""
    config = load_config(args.config)
    wavecaster = EnhancedWaveCaster(config)
    
    try:
        analysis = wavecaster.mirror_engine.cast(args.text)
        print(json.dumps(analysis, indent=2, default=str))
        
        if args.plot:
            from neuro_symbolic_engine import plot_fractal_layers
            plot_fractal_layers(analysis["fractal"], "analysis_fractal.png")
            logger.info("Saved fractal plot: analysis_fractal.png")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1

def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point"""
    parser = build_parser()
    args = parser.parse_args(argv)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Route to command handlers
    if args.command == "modulate":
        return cmd_modulate(args)
    elif args.command == "cast":
        return cmd_cast(args)
    elif args.command == "learn":
        return cmd_learn(args)
    elif args.command == "demo":
        return cmd_demo(args)
    elif args.command == "analyze":
        return cmd_analyze(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())