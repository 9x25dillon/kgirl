#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced WaveCaster System
======================================================

Tests all major components and integration points.

Author: Assistant
License: MIT
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

# Import our modules
from tauls_transformer import (
    KFPLayer, TAULSControlUnit, EntropyRegulationModule, 
    TAULSTransformerBlock, TAULSLanguageModel
)
from dual_llm_orchestrator import (
    LocalSummarizer, HTTPConfig, OrchestratorSettings
)
from neuro_symbolic_engine import (
    EntropyAnalyzer, DianneReflector, MatrixTransformer,
    MirrorCastEngine, AdaptiveLinkPlanner
)
from signal_processing import (
    ModulationScheme, FEC, ModConfig, FrameConfig, SecurityConfig,
    hamming74_encode, hamming74_decode, to_bits, from_bits,
    Modulators, encode_text, decode_bits
)
from enhanced_wavecaster import EnhancedWaveCaster, create_default_config

class TestTAULSTransformer(unittest.TestCase):
    """Test TA ULS Transformer components"""
    
    def setUp(self):
        self.dim = 64
        self.batch_size = 4
        self.seq_len = 32
        
    def test_kfp_layer(self):
        """Test Kinetic Force Principle layer"""
        layer = KFPLayer(self.dim)
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        
        output, fluctuation = layer(x)
        
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(fluctuation.shape, (self.dim,))
        self.assertTrue(torch.all(torch.isfinite(output)))
    
    def test_tauls_control_unit(self):
        """Test TA ULS control unit"""
        control_unit = TAULSControlUnit(self.dim, self.dim * 2, self.dim)
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        
        result = control_unit(x)
        
        self.assertIn('control_output', result)
        self.assertEqual(result['control_output'].shape, x.shape)
        self.assertIn('control_mixing', result)
    
    def test_entropy_regulation(self):
        """Test entropy regulation module"""
        module = EntropyRegulationModule(self.dim)
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        stress = torch.randn(self.batch_size, self.seq_len, 1)
        
        output, info = module(x, stress)
        
        self.assertEqual(output.shape, x.shape)
        self.assertIn('current_entropy', info)
        self.assertIn('target_intensity', info)
    
    def test_tauls_transformer_block(self):
        """Test complete TA ULS transformer block"""
        d_model = 64
        n_heads = 4
        d_ff = 256
        
        block = TAULSTransformerBlock(d_model, n_heads, d_ff)
        x = torch.randn(self.batch_size, self.seq_len, d_model)
        
        result = block(x)
        
        self.assertIn('output', result)
        self.assertEqual(result['output'].shape, x.shape)
        self.assertIn('attention_weights', result)
        self.assertIn('control_info', result)
        self.assertIn('entropy_info', result)
        self.assertIn('stability_info', result)
    
    def test_tauls_language_model(self):
        """Test complete TA ULS language model"""
        vocab_size = 1000
        d_model = 64
        n_heads = 4
        n_layers = 2
        max_seq_len = 128
        
        model = TAULSLanguageModel(vocab_size, d_model, n_heads, n_layers, max_seq_len)
        input_ids = torch.randint(0, vocab_size, (self.batch_size, self.seq_len))
        
        result = model(input_ids)
        
        self.assertIn('logits', result)
        self.assertEqual(result['logits'].shape, (self.batch_size, self.seq_len, vocab_size))
        self.assertIn('hidden_states', result)
        self.assertIn('stability_metrics', result)
        self.assertEqual(len(result['stability_metrics']), n_layers)

class TestDualLLMOrchestrator(unittest.TestCase):
    """Test dual LLM orchestration system"""
    
    def test_local_summarizer(self):
        """Test local summarizer fallback"""
        summarizer = LocalSummarizer()
        
        text = "This is a test document. It contains multiple sentences. Some are more important than others."
        summary = summarizer.summarize(text)
        
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        self.assertLessEqual(len(summary), len(text))
    
    def test_http_config(self):
        """Test HTTP configuration"""
        config = HTTPConfig(
            base_url="http://localhost:8080",
            model="test-model",
            mode="llama-cpp"
        )
        
        self.assertEqual(config.base_url, "http://localhost:8080")
        self.assertEqual(config.model, "test-model")
        self.assertEqual(config.mode, "llama-cpp")
    
    def test_orchestrator_settings(self):
        """Test orchestrator settings"""
        settings = OrchestratorSettings(
            temperature=0.8,
            max_tokens=256,
            style="detailed"
        )
        
        self.assertEqual(settings.temperature, 0.8)
        self.assertEqual(settings.max_tokens, 256)
        self.assertEqual(settings.style, "detailed")

class TestNeuroSymbolicEngine(unittest.TestCase):
    """Test neuro-symbolic engine components"""
    
    def test_entropy_analyzer(self):
        """Test entropy analysis"""
        analyzer = EntropyAnalyzer()
        
        # Test with different types of data
        low_entropy = "aaaaaaaaaa"
        high_entropy = "abcdefghij"
        
        low_score = analyzer.measure(low_entropy)
        high_score = analyzer.measure(high_entropy)
        
        self.assertGreater(high_score, low_score)
        self.assertGreaterEqual(low_score, 0.0)
    
    def test_dianne_reflector(self):
        """Test reflective analysis"""
        reflector = DianneReflector()
        
        text = "This is a test with some patterns and structure."
        result = reflector.reflect(text)
        
        self.assertIn('insight', result)
        self.assertIn('patterns', result)
        self.assertIn('symbolic_depth', result)
        self.assertIsInstance(result['patterns'], list)
    
    def test_matrix_transformer(self):
        """Test matrix transformation"""
        transformer = MatrixTransformer()
        
        data = "Test data for matrix analysis"
        result = transformer.project(data)
        
        self.assertIn('projected_rank', result)
        self.assertIn('structure', result)
        self.assertIn('eigenvalues', result)
        self.assertIsInstance(result['eigenvalues'], list)
    
    def test_mirror_cast_engine(self):
        """Test complete mirror cast engine"""
        engine = MirrorCastEngine()
        
        data = "Test input for comprehensive analysis"
        result = engine.cast(data)
        
        # Check all expected components
        expected_keys = [
            'entropy', 'reflection', 'matrix', 'symbolic', 'chunks',
            'endpoints', 'semantic', 'love', 'fractal', 'timestamp'
        ]
        
        for key in expected_keys:
            self.assertIn(key, result)
    
    def test_adaptive_link_planner(self):
        """Test adaptive link planner"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_db.json"
            planner = AdaptiveLinkPlanner(str(db_path))
            
            # Create mock analysis
            analysis = {
                "entropy": 2.5,
                "endpoints": {"metadata": {"complexity": 0.7}},
                "semantic": {"analysis": 0.3, "synthesis": 0.2},
                "love": {"harmony_index": 0.8},
                "fractal": {"fractal_dimension": 1.5}
            }
            
            config, explanation = planner.plan("test text", analysis)
            
            self.assertIsInstance(config, dict)
            self.assertIn("modulation", config)
            self.assertIsInstance(explanation, str)

class TestSignalProcessing(unittest.TestCase):
    """Test signal processing components"""
    
    def test_bit_conversion(self):
        """Test bit conversion utilities"""
        original = b"Hello, World!"
        bits = to_bits(original)
        recovered = from_bits(bits)
        
        self.assertEqual(original, recovered)
        self.assertEqual(len(bits), len(original) * 8)
    
    def test_hamming_encoding(self):
        """Test Hamming (7,4) encoding/decoding"""
        data_bits = [1, 0, 1, 1, 0, 0, 1, 0]  # 2 4-bit blocks
        
        encoded = hamming74_encode(data_bits)
        decoded, errors = hamming74_decode(encoded)
        
        self.assertEqual(len(encoded), 14)  # 2 * 7 bits
        self.assertEqual(decoded, data_bits)
        self.assertEqual(errors, 0)
        
        # Test error correction
        encoded[0] ^= 1  # introduce single bit error
        decoded_corrected, errors_corrected = hamming74_decode(encoded)
        
        self.assertEqual(decoded_corrected, data_bits)
        self.assertEqual(errors_corrected, 1)
    
    def test_modulation_schemes(self):
        """Test various modulation schemes"""
        bits = [1, 0, 1, 1, 0, 0, 1, 0]
        config = ModConfig(sample_rate=8000, symbol_rate=1000)
        
        # Test BFSK
        bfsk_signal = Modulators.bfsk(bits, config)
        self.assertGreater(len(bfsk_signal), 0)
        self.assertEqual(bfsk_signal.dtype, np.float32)
        
        # Test BPSK
        bpsk_audio, bpsk_iq = Modulators.bpsk(bits, config)
        self.assertGreater(len(bpsk_audio), 0)
        self.assertGreater(len(bpsk_iq), 0)
        self.assertEqual(bpsk_audio.dtype, np.float32)
        self.assertEqual(bpsk_iq.dtype, np.complex64)
        
        # Test QPSK
        qpsk_audio, qpsk_iq = Modulators.qpsk(bits, config)
        self.assertGreater(len(qpsk_audio), 0)
        self.assertGreater(len(qpsk_iq), 0)
    
    def test_encoding_decoding_pipeline(self):
        """Test complete encoding/decoding pipeline"""
        text = "Test message for encoding/decoding"
        
        fcfg = FrameConfig()
        sec = SecurityConfig(watermark="test_watermark")
        fec_scheme = FEC.HAMMING74
        
        # Encode
        bits = encode_text(text, fcfg, sec, fec_scheme)
        self.assertGreater(len(bits), 0)
        
        # Decode
        decoded_text, info = decode_bits(bits, fcfg, sec, fec_scheme)
        
        self.assertEqual(decoded_text, text)
        self.assertIn('errors_corrected', info)
        self.assertIn('watermark_ok', info)
        self.assertTrue(info['watermark_ok'])

class TestEnhancedWaveCaster(unittest.TestCase):
    """Test main integration system"""
    
    def setUp(self):
        self.config = create_default_config()
        # Remove LLM config for testing (avoid network calls)
        self.config.pop('llm', None)
        
    def test_config_creation(self):
        """Test configuration creation"""
        config = create_default_config()
        
        self.assertIn('modulation', config)
        self.assertIn('framing', config)
        self.assertIn('security', config)
        self.assertIn('llm', config)
    
    def test_wavecaster_initialization(self):
        """Test WaveCaster initialization"""
        wavecaster = EnhancedWaveCaster(self.config)
        
        self.assertIsNotNone(wavecaster.mirror_engine)
        self.assertIsNotNone(wavecaster.adaptive_planner)
    
    def test_direct_casting(self):
        """Test direct text casting"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wavecaster = EnhancedWaveCaster(self.config)
            
            result = wavecaster.cast_text_direct(
                text="Test message",
                scheme=ModulationScheme.QPSK,
                output_dir=Path(tmpdir),
                use_adaptive=True,
                want_wav=True,
                want_iq=False
            )
            
            self.assertIn('text', result)
            self.assertIn('analysis', result)
            self.assertIn('config', result)
            self.assertIn('paths', result)
            
            # Check that files were created
            if result['paths']['wav']:
                self.assertTrue(Path(result['paths']['wav']).exists())
    
    def test_adaptive_learning(self):
        """Test adaptive learning system"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self.config.copy()
            config['db_path'] = str(Path(tmpdir) / "test_db.json")
            
            wavecaster = EnhancedWaveCaster(config)
            
            result = wavecaster.learn_adaptive(
                texts=["Test message 1", "Test message 2"],
                episodes=5
            )
            
            self.assertIn('episodes', result)
            self.assertIn('success_rate', result)
            self.assertEqual(len(result['episodes']), 5)
            self.assertIsInstance(result['success_rate'], float)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_end_to_end_processing(self):
        """Test complete end-to-end processing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test configuration
            config = create_default_config()
            config.pop('llm', None)  # Remove LLM to avoid network calls
            
            wavecaster = EnhancedWaveCaster(config)
            
            # Test text
            test_text = "This is a comprehensive test of the Enhanced WaveCaster system! 🚀"
            
            # Process with different modulation schemes
            schemes_to_test = [
                ModulationScheme.BFSK,
                ModulationScheme.QPSK,
                ModulationScheme.QAM16
            ]
            
            results = []
            
            for scheme in schemes_to_test:
                result = wavecaster.cast_text_direct(
                    text=test_text,
                    scheme=scheme,
                    output_dir=Path(tmpdir) / scheme.name.lower(),
                    use_adaptive=True,
                    want_wav=True,
                    want_iq=True
                )
                
                results.append(result)
                
                # Verify structure
                self.assertIn('analysis', result)
                self.assertIn('config', result)
                self.assertIn('paths', result)
                
                # Verify files exist
                if result['paths']['wav']:
                    self.assertTrue(Path(result['paths']['wav']).exists())
                if result['paths']['iq']:
                    self.assertTrue(Path(result['paths']['iq']).exists())
                if result['paths']['meta']:
                    self.assertTrue(Path(result['paths']['meta']).exists())
            
            # Verify all schemes produced different configurations
            configs = [r['config'] for r in results]
            self.assertEqual(len(set(str(c) for c in configs)), len(configs))

def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestTAULSTransformer,
        TestDualLLMOrchestrator,
        TestNeuroSymbolicEngine,
        TestSignalProcessing,
        TestEnhancedWaveCaster,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)