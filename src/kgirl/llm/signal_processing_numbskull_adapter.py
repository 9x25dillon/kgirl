#!/usr/bin/env python3
"""
Signal Processing + Numbskull Integration Adapter
=================================================

Deep integration between Signal Processing and Numbskull:
- Embedding-based modulation scheme selection
- Pattern-aware signal generation
- Embedding transmission and encoding
- Robust signal processing with error correction

Author: Assistant
License: MIT
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add numbskull to path
numbskull_path = Path("/home/kill/numbskull")
if numbskull_path.exists() and str(numbskull_path) not in sys.path:
    sys.path.insert(0, str(numbskull_path))

try:
    from advanced_embedding_pipeline import HybridEmbeddingPipeline, HybridConfig
    NUMBSKULL_AVAILABLE = True
except ImportError:
    NUMBSKULL_AVAILABLE = False

import signal_processing as dsp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalProcessingNumbskullAdapter:
    """
    Adapter integrating Signal Processing with Numbskull embeddings
    
    Provides:
    - Embedding-guided modulation selection
    - Pattern-based signal generation
    - Embedding encoding into signals
    - Robust transmission with FEC
    """
    
    def __init__(
        self,
        use_numbskull: bool = True,
        numbskull_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize adapter"""
        logger.info("=" * 70)
        logger.info("SIGNAL PROCESSING + NUMBSKULL ADAPTER")
        logger.info("=" * 70)
        
        # Initialize Numbskull
        self.numbskull = None
        if use_numbskull and NUMBSKULL_AVAILABLE:
            config = HybridConfig(**(numbskull_config or {}))
            self.numbskull = HybridEmbeddingPipeline(config)
            logger.info("✅ Numbskull pipeline integrated")
        else:
            logger.warning("⚠️  Operating without Numbskull embeddings")
        
        # Signal processing components available
        self.modulators = dsp.Modulators()
        logger.info("✅ Signal modulators ready")
        logger.info("=" * 70)
    
    async def select_modulation_from_embedding(
        self,
        text: str
    ) -> Tuple[dsp.ModulationScheme, Dict[str, Any]]:
        """
        Select optimal modulation scheme based on embedding analysis
        
        Args:
            text: Input text
        
        Returns:
            (ModulationScheme, analysis_dict)
        """
        logger.info("\n📡 Embedding-Based Modulation Selection")
        
        # Default scheme
        scheme = dsp.ModulationScheme.QPSK
        analysis = {"method": "default", "reason": "no embedding available"}
        
        if self.numbskull:
            try:
                # Generate embedding
                emb_result = await self.numbskull.embed(text)
                embedding = emb_result["fused_embedding"]
                
                # Analyze embedding characteristics
                norm = float(np.linalg.norm(embedding))
                variance = float(np.var(embedding))
                complexity = len(emb_result["metadata"]["components_used"])
                
                # Select scheme based on characteristics
                if variance > 0.1:
                    # High variance = complex signal = use robust scheme
                    scheme = dsp.ModulationScheme.OFDM
                    reason = "High variance detected, using OFDM for robustness"
                elif complexity >= 3:
                    # Multiple components = rich content = use QAM
                    scheme = dsp.ModulationScheme.QAM16
                    reason = "Multi-component embeddings, using QAM16 for efficiency"
                elif norm < 0.5:
                    # Low energy = simple content = use BFSK
                    scheme = dsp.ModulationScheme.BFSK
                    reason = "Low complexity, using BFSK for simplicity"
                else:
                    # Medium complexity = use QPSK
                    scheme = dsp.ModulationScheme.QPSK
                    reason = "Balanced characteristics, using QPSK"
                
                analysis = {
                    "method": "embedding_guided",
                    "norm": norm,
                    "variance": variance,
                    "complexity": complexity,
                    "reason": reason
                }
                
                logger.info(f"  ✅ Selected {scheme.name}: {reason}")
                
            except Exception as e:
                logger.warning(f"  ⚠️  Embedding analysis failed: {e}, using default")
        
        return scheme, analysis
    
    async def encode_embedding_to_signal(
        self,
        text: str,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Encode text with embeddings into modulated signal
        
        Args:
            text: Text to encode
            output_dir: Optional output directory
        
        Returns:
            Encoding results
        """
        logger.info("\n🎵 Encoding Text to Signal with Embeddings")
        
        results = {
            "text_length": len(text),
            "embedding_info": None,
            "modulation_scheme": None,
            "signal_generated": False
        }
        
        # Select modulation based on embedding
        scheme, analysis = await self.select_modulation_from_embedding(text)
        results["modulation_scheme"] = scheme.name
        results["selection_analysis"] = analysis
        
        # Generate signal
        try:
            # Configuration
            mod_config = dsp.ModConfig(
                sample_rate=48000,
                symbol_rate=1200,
                amplitude=0.7
            )
            frame_config = dsp.FrameConfig()
            security_config = dsp.SecurityConfig()
            fec_scheme = dsp.FEC.HAMMING74
            
            # Encode text to bits
            bits = dsp.encode_text(text, frame_config, security_config, fec_scheme)
            logger.info(f"  ✅ Encoded to {len(bits)} bits")
            
            # Modulate to signal
            audio_signal, iq_signal = dsp.bits_to_signals(bits, scheme, mod_config)
            
            if audio_signal is not None:
                results["signal_generated"] = True
                results["signal_length"] = len(audio_signal)
                results["sample_rate"] = mod_config.sample_rate
                logger.info(f"  ✅ Generated {len(audio_signal)} samples at {mod_config.sample_rate}Hz")
                
                # Optionally save
                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(exist_ok=True)
                    wav_path = output_dir / "encoded_signal.wav"
                    dsp.write_wav_mono(wav_path, audio_signal, mod_config.sample_rate)
                    results["output_file"] = str(wav_path)
                    logger.info(f"  ✅ Saved to {wav_path}")
            
        except Exception as e:
            logger.error(f"  ❌ Signal generation failed: {e}")
            results["error"] = str(e)
        
        return results
    
    async def embedding_to_constellation(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        Create constellation diagram from embeddings
        
        Args:
            text: Input text
        
        Returns:
            Constellation data
        """
        logger.info("\n⭐ Embedding to Constellation Mapping")
        
        if not self.numbskull:
            logger.warning("  ⚠️  Numbskull not available")
            return {"error": "Numbskull not available"}
        
        try:
            # Generate embedding
            emb_result = await self.numbskull.embed(text)
            embedding = emb_result["fused_embedding"]
            
            # Map embedding to constellation points
            # Use first N dimensions as I/Q pairs
            n_symbols = min(64, len(embedding) // 2)
            symbols = []
            
            for i in range(n_symbols):
                I = float(embedding[i*2])
                Q = float(embedding[i*2+1]) if i*2+1 < len(embedding) else 0.0
                symbols.append(I + 1j * Q)
            
            symbols_array = np.array(symbols, dtype=np.complex64)
            
            # Normalize
            symbols_array = symbols_array / (np.abs(symbols_array).max() + 1e-10)
            
            logger.info(f"  ✅ Created {n_symbols} constellation points")
            
            return {
                "symbols": symbols_array.tolist(),
                "num_symbols": n_symbols,
                "embedding_dim": len(embedding),
                "components": emb_result["metadata"]["components_used"]
            }
            
        except Exception as e:
            logger.error(f"  ❌ Constellation mapping failed: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Clean up resources"""
        if self.numbskull:
            await self.numbskull.close()
        logger.info("✅ Signal processing adapter closed")


async def demo_signal_adapter():
    """Demonstration of signal processing + Numbskull integration"""
    print("\n" + "=" * 70)
    print("SIGNAL PROCESSING + NUMBSKULL ADAPTER DEMO")
    print("=" * 70)
    
    # Create adapter
    adapter = SignalProcessingNumbskullAdapter(
        use_numbskull=NUMBSKULL_AVAILABLE,
        numbskull_config={"use_fractal": True, "cache_embeddings": True}
    )
    
    # Test cases
    test_texts = [
        "Simple message for basic modulation",
        "Complex multi-layer neural network architecture with attention mechanisms",
        "x^2 + 2x + 1 = 0"
    ]
    
    # Test modulation selection
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: Modulation Selection")
        print(f"{'='*70}")
        print(f"Text: {text[:60]}...")
        
        scheme, analysis = await adapter.select_modulation_from_embedding(text)
        print(f"Selected: {scheme.name}")
        print(f"Reason: {analysis.get('reason', 'N/A')}")
    
    # Test signal encoding
    print(f"\n{'='*70}")
    print("TEST: Signal Encoding")
    print(f"{'='*70}")
    result = await adapter.encode_embedding_to_signal(test_texts[0])
    print(f"Signal generated: {result['signal_generated']}")
    if result.get('signal_length'):
        print(f"Signal length: {result['signal_length']} samples")
        print(f"Modulation: {result['modulation_scheme']}")
    
    # Test constellation mapping
    print(f"\n{'='*70}")
    print("TEST: Constellation Mapping")
    print(f"{'='*70}")
    constellation = await adapter.embedding_to_constellation(test_texts[1])
    if 'num_symbols' in constellation:
        print(f"Symbols: {constellation['num_symbols']}")
        print(f"Components: {constellation.get('components', 'N/A')}")
    
    # Cleanup
    await adapter.close()
    
    print(f"\n{'='*70}")
    print("✅ DEMO COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(demo_signal_adapter())

