# Atlas report — kgirl and its neighbours

_Generated 2026-09-28 18:23 UTC by `python -m kgirl.harness report`. 33 repositories, 2806 files, 34201 symbols. Structure only: signatures, docstrings, imports, clone hashes — no file bodies._

Edges: **import** = resolved import (cross-repo when module names match); **clone** = identical normalized function/class body in both repos (same defect, no runtime link); **shared names** = same non-generic top-level class/function name defined in both.

## Coupling into `kgirl` (blast radius across repos)

| repo | kgirl→repo imports | repo→kgirl imports | cloned symbols | shared names | kgirl files exposed | score |
|---|---:|---:|---:|---:|---:|---:|
| numbskull | 1 | 0 | 884 | 186 | 2 | 1820 |
| neurotronic_phase_caster | 0 | 0 | 413 | 91 | 0 | 849 |
| TEmp-oral_vectraxice | 0 | 0 | 101 | 124 | 0 | 233 |
| orwells-egg | 0 | 4 | 92 | 76 | 0 | 215 |
| Fractal_cascade_simulation | 2 | 0 | 54 | 24 | 3 | 123 |
| al-ULS | 0 | 0 | 43 | 14 | 0 | 90 |
| Eopiez | 0 | 0 | 37 | 37 | 0 | 83 |
| carryon | 0 | 4 | 10 | 114 | 0 | 60 |
| NuRea_sim | 1 | 0 | 22 | 41 | 3 | 60 |
| Implementing_julia_bridge_al-ULS | 0 | 0 | 19 | 20 | 0 | 43 |
| shout | 0 | 0 | 17 | 12 | 0 | 37 |
| bigLIMp | 0 | 0 | 15 | 14 | 0 | 34 |
| 9xdSq-LIMPS-FemTO-R1C | 4 | 0 | 4 | 7 | 6 | 28 |
| engapp | 0 | 0 | 9 | 19 | 0 | 23 |
| Consciousness_as_Topological_Holography | 0 | 0 | 5 | 19 | 0 | 15 |

- **numbskull** — **hub imports**: `scripts/workflows/complete_unified_platform.py -> holographic_similarity_engine.py`; **clones**: `ALULS.batch_eval_symbolic_calls: kgirl:src/chaos_llm/services/al_uls.py == numbskull:src/chaos_llm/services/al_uls.py`, `ALULS.batch_eval_symbolic_calls: kgirl:src/kgirl/llm/al_uls.py == numbskull:src/chaos_llm/services/al_uls.py`, `ALULS.eval_symbolic_call_async: kgirl:src/chaos_llm/services/al_uls.py == numbskull:src/chaos_llm/services/al_uls.py`, `ALULS.eval_symbolic_call_async: kgirl:src/kgirl/llm/al_uls.py == numbskull:src/chaos_llm/services/al_uls.py`; **shared names**: `ALULSClient`, `ALULSWSClient`, `AdaptiveLinkPlanner`, `AppState`
- **neurotronic_phase_caster** — **clones**: `AdaptiveResonanceController.__init__: kgirl:(QNPCE) v3.0.py == neurotronic_phase_caster:Bloom.py`, `AdaptiveResonanceController.__init__: kgirl:(QNPCE) v3.0.py == neurotronic_phase_caster:QNPCE_v3.0.py`, `AdaptiveResonanceController.__init__: kgirl:QBCROM.py == neurotronic_phase_caster:QUANTUM BIO-COHERENCE RESONATOR: THE OCTITRICE MANIFESTATION.py`, `AdaptiveResonanceController.__init__: kgirl:QBCROM.py == neurotronic_phase_caster:src/quantum_bio_coherence_resonator.py`; **shared names**: `AdaptiveResonanceController`, `AdvancedEmergentMemoryPatterns`, `AdvancedFractalEncoder`, `BioFractalConfig`
- **TEmp-oral_vectraxice** — **clones**: `EntropyRegulationModule.__init__: kgirl:src/kgirl/llm/ta_uls_llm.py == TEmp-oral_vectraxice:ta_uls_llm.py`, `EntropyRegulationModule.__init__: kgirl:src/kgirl/llm/tau_uls_wavecaster_enhanced.py == TEmp-oral_vectraxice:ta_uls_llm.py`, `EntropyRegulationModule.__init__: kgirl:src/kgirl/llm/tauls_model.py == TEmp-oral_vectraxice:ta_uls_llm.py`, `EntropyRegulationModule.__init__: kgirl:src/kgirl/llm/tauls_transformer.py == TEmp-oral_vectraxice:ta_uls_llm.py`; **shared names**: `AdapterDetail`, `AdaptersList`, `AdaptiveLinkPlanner`, `Backup`
- **orwells-egg** — **imports hub**: `neutronics.py -> src/chaos_llm/services/al_uls_client.py`, `neutronics.py -> src/chaos_llm/services/al_uls_ws_client.py`; **clones**: `BaseLLM: kgirl:src/kgirl/llm/dual_llm_orchestrator.py == orwells-egg:DLmWaveCaster.py`, `BaseLLM: kgirl:src/kgirl/llm/tau_uls_wavecaster_enhanced.py == orwells-egg:DLmWaveCaster.py`, `BasicModulator.to_bits: kgirl:scripts/demo/demo_basic.py == orwells-egg:DLmWaveCaster.py`, `DualLLMOrchestrator.__init__: kgirl:src/kgirl/llm/tau_uls_wavecaster_enhanced.py == orwells-egg:DLmWaveCaster.py`; **shared names**: `ALULSClient`, `ALULSWSClient`, `BaseLLM`, `BatchSymbolicRequest`
- **Fractal_cascade_simulation** — **hub imports**: `scripts/workflows/complete_integration_runner.py -> matrix_orchestrator.py`, `src/kgirl/llm/unified_quantum_llm_system.py -> matrix_orchestrator.py`; **clones**: `AgentHints: kgirl:src/kgirl/core/UCs.py == Fractal_cascade_simulation:unified_coherence_system.py`, `AuditState: kgirl:src/kgirl/core/UCs.py == Fractal_cascade_simulation:unified_coherence_system.py`, `CR2BC.__init__: kgirl:src/kgirl/core/UCs.py == Fractal_cascade_simulation:unified_coherence_system.py`, `CR2BC._apply_efl_coend: kgirl:src/kgirl/core/UCs.py == Fractal_cascade_simulation:unified_coherence_system.py`; **shared names**: `AgentHints`, `AuditState`, `Backend`, `BridgeState`
- **al-ULS** — **clones**: `JuliaClient.__init__: kgirl:src/kgirl/neural/sweet integrated_training_system.py == al-ULS:al-ULs.Py`, `JuliaClient._make_request: kgirl:src/kgirl/neural/sweet integrated_training_system.py == al-ULS:al-ULs.Py`, `JuliaClient.analyze_polynomials: kgirl:src/kgirl/neural/sweet integrated_training_system.py == al-ULS:al-ULs.Py`, `JuliaClient.create_polynomials: kgirl:src/kgirl/neural/sweet integrated_training_system.py == al-ULS:al-ULs.Py`; **shared names**: `JuliaClient`, `JuliaServerManager`, `KFPLayer`, `SimpleTA_ULS`
- **Eopiez** — **clones**: `ALULSClient.__init__: kgirl:src/chaos_llm/services/al_uls_client.py == Eopiez:al-uls-evolution/services/al_uls_client.py`, `ALULSClient.__init__: kgirl:src/kgirl/llm/al_uls_client.py == Eopiez:al-uls-evolution/services/al_uls_client.py`, `ALULSClient.batch_eval: kgirl:src/chaos_llm/services/al_uls_client.py == Eopiez:al-uls-evolution/services/al_uls_client.py`, `ALULSClient.batch_eval: kgirl:src/kgirl/llm/al_uls_client.py == Eopiez:al-uls-evolution/services/al_uls_client.py`; **shared names**: `ALULSClient`, `AppState`, `EmotionalArc`, `EntanglementLink`
- **carryon** — **imports hub**: `sydv.py -> src/chaos_llm/services/al_uls_client.py`, `sydv.py -> src/chaos_llm/services/al_uls_ws_client.py`; **clones**: `Persona: kgirl:src/kgirl/utils/soulpack.py == carryon:server/app/core/soulpack.py`, `Pointers: kgirl:src/kgirl/utils/soulpack.py == carryon:server/app/core/soulpack.py`, `PrimeReq: kgirl:src/kgirl/utils/prime.py == carryon:server/app/routers/prime.py`, `Settings: kgirl:src/kgirl/utils/config.py == carryon:server/app/config.py`; **shared names**: `ALULSClient`, `ALULSWSClient`, `AdaptiveLinkPlanner`, `Backup`

## Highest-blast modules in `kgirl`

### `advanced_embedding_pipeline/__init__.py` — 47 dependents (23 direct, 0 clone sites)
- d1 import `kgirl:scripts/workflows/master_data_flow_orchestrator.py` — L50: advanced_embedding_pipeline [HybridEmbeddingPipeline,HybridConfig]
- d1 import `kgirl:scripts/workflows/complete_integration_orchestrator.py` — L46: advanced_embedding_pipeline [HybridEmbeddingPipeline,HybridConfig]
- d1 import `kgirl:src/kgirl/core/limp_module_manager.py` — L186: advanced_embedding_pipeline [HybridEmbeddingPipeline,HybridConfig]
- d1 import `kgirl:scripts/demo/verify_all_components.py` — L46: advanced_embedding_pipeline [HybridEmbeddingPipeline,HybridConfig]
- d1 import `kgirl:src/kgirl/cognitive/recursive_cognitive_knowledge.py` — L36: advanced_embedding_pipeline [HybridEmbeddingPipeline,HybridConfig]
- d1 import `kgirl:src/kgirl/quantum/quantum_knowledge_database.py` — L57: advanced_embedding_pipeline [HybridEmbeddingPipeline,HybridConfig]
- d1 import `kgirl:scripts/demo/benchmark_full_stack.py` — L38: advanced_embedding_pipeline [HybridEmbeddingPipeline,HybridConfig,SemanticConfig,MathematicalConfig,FractalConfig]
- d1 import `kgirl:scripts/demo/benchmark_integration.py` — L40: advanced_embedding_pipeline [HybridEmbeddingPipeline,HybridConfig,SemanticConfig,MathematicalConfig,FractalConfig]

### `src/kgirl/quantum/holographic_memory_system.py` — 34 dependents (12 direct, 0 clone sites)
- d1 import `kgirl:scripts/demo/demo_integrated_system.py` — L34: holographic_memory_system [EnhancedCognitiveMemoryOrchestrator,demo_enhanced_holographic_memory]
- d1 import `kgirl:src/kgirl/core/limps_holographic_orchestrator.py` — L28: holographic_memory_system [EnhancedCognitiveMemoryOrchestrator]
- d1 import `kgirl:src/kgirl/quantum/quantum_knowledge_database.py` — L31: holographic_memory_system [HolographicAssociativeMemory]
- d1 import `kgirl:scripts/demo/verify_all_components.py` — L72: holographic_memory_system [HolographicMemorySystem]
- d1 import `kgirl:scripts/workflows/complete_integration_orchestrator.py` — L50: holographic_memory_system [HolographicMemorySystem]
- d1 import `kgirl:src/kgirl/cognitive/advanced_cognitive_enhancements.py` — L24: holographic_memory_system [EnhancedCognitiveMemoryOrchestrator,HolographicAssociativeMemory,FractalMemoryEncoder,QuantumHolographicStorage]
- d1 import `kgirl:src/kgirl/cognitive/cognitive_integration_bridge.py` — L24: holographic_memory_system [EnhancedCognitiveMemoryOrchestrator,HolographicAssociativeMemory,FractalMemoryEncoder,QuantumHolographicStorage,EmergentMemoryPatterns]
- d1 import `kgirl:src/kgirl/cognitive/recursive_cognitive_knowledge.py` — L42: holographic_memory_system [HolographicMemorySystem]

### `src/kgirl/cognitive/neuro_symbolic_numbskull_adapter.py` — 10 dependents (9 direct, 0 clone sites)
- d1 import `kgirl:scripts/demo/master_playground.py` — L47: neuro_symbolic_numbskull_adapter [NeuroSymbolicNumbskullAdapter]
- d1 import `kgirl:scripts/demo/playground.py` — L28: neuro_symbolic_numbskull_adapter [NeuroSymbolicNumbskullAdapter]
- d1 import `kgirl:scripts/demo/simple_integrated_wavecaster_demo.py` — L33: neuro_symbolic_numbskull_adapter [NeuroSymbolicNumbskullAdapter]
- d1 import `kgirl:scripts/workflows/adapter_integration_demo.py` — L33: neuro_symbolic_numbskull_adapter [NeuroSymbolicNumbskullAdapter]
- d1 import `kgirl:scripts/workflows/complete_adapter_suite_demo.py` — L38: neuro_symbolic_numbskull_adapter [NeuroSymbolicNumbskullAdapter]
- d1 import `kgirl:scripts/workflows/complete_integration_orchestrator.py` — L44: neuro_symbolic_numbskull_adapter [NeuroSymbolicNumbskullAdapter]
- d1 import `kgirl:scripts/workflows/integrated_wavecaster_runner.py` — L42: neuro_symbolic_numbskull_adapter [NeuroSymbolicNumbskullAdapter]
- d1 import `kgirl:src/kgirl/llm/aipyapp_playground.py` — L37: neuro_symbolic_numbskull_adapter [NeuroSymbolicNumbskullAdapter]

### `src/kgirl/llm/signal_processing_numbskull_adapter.py` — 9 dependents (8 direct, 0 clone sites)
- d1 import `kgirl:scripts/demo/master_playground.py` — L48: signal_processing_numbskull_adapter [SignalProcessingNumbskullAdapter]
- d1 import `kgirl:scripts/demo/playground.py` — L29: signal_processing_numbskull_adapter [SignalProcessingNumbskullAdapter]
- d1 import `kgirl:scripts/demo/simple_integrated_wavecaster_demo.py` — L34: signal_processing_numbskull_adapter [SignalProcessingNumbskullAdapter]
- d1 import `kgirl:scripts/workflows/adapter_integration_demo.py` — L34: signal_processing_numbskull_adapter [SignalProcessingNumbskullAdapter]
- d1 import `kgirl:scripts/workflows/complete_adapter_suite_demo.py` — L39: signal_processing_numbskull_adapter [SignalProcessingNumbskullAdapter]
- d1 import `kgirl:scripts/workflows/complete_integration_orchestrator.py` — L45: signal_processing_numbskull_adapter [SignalProcessingNumbskullAdapter]
- d1 import `kgirl:scripts/workflows/integrated_wavecaster_runner.py` — L43: signal_processing_numbskull_adapter [SignalProcessingNumbskullAdapter]
- d1 import `kgirl:src/kgirl/utils/play.py` — L7: signal_processing_numbskull_adapter [SignalProcessingNumbskullAdapter]

### `src/kgirl/llm/numbskull_dual_orchestrator.py` — 19 dependents (8 direct, 0 clone sites)
- d1 import `kgirl:scripts/workflows/integrated_wavecaster_runner.py` — L41: numbskull_dual_orchestrator [create_numbskull_orchestrator]
- d1 import `kgirl:scripts/demo/simple_integrated_wavecaster_demo.py` — L32: numbskull_dual_orchestrator [create_numbskull_orchestrator]
- d1 import `kgirl:scripts/demo/benchmark_full_stack.py` — L46: numbskull_dual_orchestrator [create_numbskull_orchestrator,NUMBSKULL_AVAILABLE]
- d1 import `kgirl:scripts/demo/benchmark_integration.py` — L48: numbskull_dual_orchestrator [create_numbskull_orchestrator,NUMBSKULL_AVAILABLE]
- d1 import `kgirl:scripts/demo/playground.py` — L27: numbskull_dual_orchestrator [create_numbskull_orchestrator]
- d1 import `kgirl:scripts/workflows/run_integrated_workflow.py` — L31: numbskull_dual_orchestrator [create_numbskull_orchestrator,NUMBSKULL_AVAILABLE]
- d1 import `kgirl:src/kgirl/cognitive/unified_cognitive_orchestrator.py` — L48: numbskull_dual_orchestrator [create_numbskull_orchestrator,NumbskullDualOrchestrator]
- d1 import `kgirl:src/kgirl/llm/enable_aluls_and_qwen.py` — L27: numbskull_dual_orchestrator [create_numbskull_orchestrator]

### `src/kgirl/core/signal_processing.py` — 26 dependents (8 direct, 8 clone sites)
- d1 import `kgirl:scripts/workflows/integrated_wavecaster_runner.py` — L46: signal_processing
- d1 import `kgirl:scripts/demo/simple_integrated_wavecaster_demo.py` — L37: signal_processing
- d1 import `kgirl:scripts/demo/test_system.py` — L33: signal_processing [ModulationScheme,FEC,ModConfig,FrameConfig,SecurityConfig,hamming74_encode,hamming74_decode,to_bits,from_bits,Modulators,encode_text,decode_bits]
- d1 import `kgirl:src/kgirl/cognitive/evolutionary_communicator.py` — L28: signal_processing
- d1 import `kgirl:src/kgirl/llm/evolutionary_numbskull_adapter.py` — L36: signal_processing
- d1 import `kgirl:src/kgirl/llm/signal_processing_numbskull_adapter.py` — L35: signal_processing
- d1 import `kgirl:src/kgirl/neural/bi-inrefernce.py` — L36: signal_processing [ModulationScheme,FEC,ModConfig,FrameConfig,SecurityConfig,full_process_and_save,demo_signal_processing,play_audio]
- d1 import `kgirl:src/kgirl/neural/enhanced_wavecaster.py` — L36: signal_processing [ModulationScheme,FEC,ModConfig,FrameConfig,SecurityConfig,full_process_and_save,demo_signal_processing,play_audio]

## Repository cards

### 9xdSq-LIMPS-FemTO-R1C  `f7df039391bf`
- **languages**: julia 22f/3125loc, python 17f/1819loc, markdown 5f/1232loc, shell 1f/110loc
- **readme**: A comprehensive, integrated system combining polynomial operations, matrix processing, entropy analysis, and AI model integration.
- **entrypoints**: `main.py`, `tests/test_integration.py`, `interfaces/julia_client/julia_client.py`, `matrix_ops/processors/matrix_processor.py`
- **hub modules**: `polynomial_system/core/DynamicPolynomials.jl (<-9)`, `matrix_ops/processors/matrix_processor.py (<-4)`, `interfaces/julia_client/julia_client.py (<-3)`, `limps_core/python/limps_workflow.py (<-3)`, `entropy_analysis/engines/entropy_engine.py (<-2)`, `limps_core/julia/api.jl (<-1)`
- **key types**: `EntropyEngine — entropy_analysis/engines/entropy_engine.py:85`, `Token — entropy_analysis/engines/entropy_engine.py:5`, `EntropyNode — entropy_analysis/engines/entropy_engine.py:35`, `Monomial — polynomial_system/core/mono.jl:8`, `Polynomial — polynomial_system/core/poly.jl:7`, `JuliaClient — interfaces/julia_client/julia_client.py:15`
- **external deps**: `torch`, `numpy`, `JSON`, `Statistics`, `MultivariatePolynomials`, `LinearAlgebra`, `HTTP`, `Sockets`

### Badnono  `857b45749d30`
- **languages**: markdown 1f/101loc
- **readme**: A tiny offline top-down action game in one HTML file. 8-bit look, Zelda-ish overhead view, but with height: jumping, flying, dashing and gliding.

### Consciousness_as_Topological_Holography  `23db1c0f8372`
- **languages**: python 12f/5413loc, markdown 2f/343loc
- **readme**: A runnable implementation of consciousness as topological holography in (2+1)D spacetime, where observer states are Cardy boundaries in a topological quantum field theory.
- **entrypoints**: `cr2bc.py`, `run_demos.py`, `gradio_app.py`, `efl_cr2bc_demo.py`, `qincrs_guardian.py`, `quantum_coherence.py`, `unified_framework.py`, `qincrs_cr2bc_bridge.py`
- **hub modules**: `cr2bc.py (<-4)`, `topological_consciousness.py (<-3)`, `efl_mem.py (<-2)`, `quantum_coherence.py (<-2)`, `qincrs_guardian.py (<-1)`
- **key types**: `CoherenceState — Lifesaver.py:639`, `ConsciousnessState — quantum_coherence.py:68`, `FrequencyBand — cr2bc.py:28`, `AgentHints — cr2bc.py:72`, `CR2BC — cr2bc.py:144`, `CR2BCConfig — cr2bc.py:109`
- **external deps**: `numpy`, `matplotlib`, `types`, `gradio`
- **warning**: 2 python file(s) fail to parse

### Eopiez  `ff7eceba1304`
- **languages**: julia 36f/11946loc, markdown 31f/7638loc, python 41f/7226loc, shell 9f/1404loc
- **readme**: > A hybrid AI/ML symbolic computation platform combining neural networks, symbolic reasoning, and quantum-inspired algorithms for advanced pattern detection and memory processing.
- **entrypoints**: `api.py`, `api_gateway.py`, `limps-aalc/demo.py`, `src/qvnm_server.jl`, `refactored/limps_env.py`, `examples/qvnm_example.py`, `refactored/limps_client.py`, `al-uls-evolution/api/main.py`
- **hub modules**: `src/MessageVectorizer.jl (<-12)`, `src/motif_detection/motif_server.jl (<-5)`, `src/motif_detection/motifs.jl (<-5)`, `src/motif_detection/parser.jl (<-5)`, `src/limps/symbolic_memory.jl (<-4)`, `limps-aalc/services/admin-api/coach.py (<-3)`
- **key types**: `SecurityConfig — limps_env.jl:149`, `MotifToken — src/MessageVectorizer.jl:19`, `MotifToken — src/Types.jl:38`, `ALULSClient — al-uls-evolution/services/al_uls_client.py:52`, `ALULSClient — limps-aalc/services/al-uls-client/aluls_client.py:3`, `MessageVectorizer — src/MessageVectorizer.jl:44`
- **external deps**: `torch`, `LinearAlgebra`, `Statistics`, `numpy`, `JSON3`, `fastapi`, `Random`, `Pkg`, `HTTP`

### Fractal_cascade_simulation  `b1021b6ebf1a`
- **languages**: python 23f/7266loc, markdown 4f/1176loc, javascript 1f/176loc
- **readme**: Hierarchical Reasoning Model with advanced mathematical optimization via Matrix Orchestrator, enhanced with the Unified Coherence System for safety, coherence tracking, and resilient optimization.
- **entrypoints**: `run_opt.py`, `set_env.py`, `evaluate.py`, `pretrain.py`, `test_npbs.py`, `test_adapter.py`, `setup_environment.py`, `matrix_integration.py`
- **hub modules**: `matrix_orchestrator.py (<-6)`, `dataset/common.py (<-4)`, `matrix_integration.py (<-2)`, `neuro_phasonic_bridge_v2.py (<-2)`, `unified_coherence_system.py (<-2)`, `coherence_matrix_integration.py (<-1)`
- **key types**: `CoherenceState — unified_coherence_system.py:41`, `Settings — Enhanced _Matrix_Orchestrator.py:44`, `Settings — matrix_orchestrator.py:44`, `MotifToken — neuro_phasonic_bridge_v2.py:83`, `BridgeState — neuro_phasonic_bridge_v2.py:93`, `NeuroPhasonicBridge — neuro_phasonic_bridge_v2.py:123`
- **external deps**: `torch`, `numpy`, `pydantic`, `models`, `scipy`

### H_ealer  `295f4bd11f79`
- **languages**: markdown 1f/289loc, python 15f/198loc
- **readme**: **Harm Evaluation, Adversarial Learning, Evidence & Repair**
- **key types**: `ModelRequest — ModRequest.py:1`, `ModelResponse — ModResposne.py:10`, `ExperimentEvent — domain/events.py:1`, `ExperimentStore — ExpStore.py:1`, `ScoreResult — ScoResults.py:1`, `TerminationInfo — ModResposne.py:1`
- **external deps**: `textwrap`
- **warning**: 2 python file(s) fail to parse

### Implementing_julia_bridge_al-ULS  `d0c6029d1f0a`
- **languages**: python 8f/2882loc, markdown 3f/1259loc, julia 1f/142loc
- **readme**: A comprehensive system combining **Categorical Coherence Linting (CCL)** with **Julia-based optimization** and an advanced **WaveCaster signal modulation** engine for dual LLM orchestration.
- **entrypoints**: `ccl.py`, `wavecaster.py`, `mock_al_uls_server.py`, `examples/ccl_analysis.py`, `tests/test_wavecaster.py`, `examples/basic_modulation.py`
- **hub modules**: `wavecaster.py (<-2)`, `ccl.py (<-1)`
- **key types**: `SecurityConfig — wavecaster.py:123`, `DualLLMOrchestrator — wavecaster.py:746`, `FrameConfig — wavecaster.py:116`, `ModulationScheme — wavecaster.py:76`, `HTTPConfig — wavecaster.py:90`, `ContentAnalyzer — wavecaster.py:132`
- **external deps**: `numpy`, `Crypto`, `scipy`, `matplotlib`, `wave`, `ast`, `importlib`

### KoLd__FEEt.ice_skating_  `6154e62eff8e`
- **languages**: typescript 165f/31086loc, markdown 32f/13163loc, javascript 15f/1913loc, gdscript 6f/1832loc, cpp 8f/1478loc, c 3f/921loc, python 3f/504loc, shell 1f/6loc
- **readme**: **A physics-first figure skating simulation.** *The ice remembers every line.*
- **hub modules**: `tools/ice-lab/sim/types.ts (<-126)`, `tools/ice-lab/sim/params.ts (<-112)`, `tools/ice-lab/sim/solver.ts (<-81)`, `tools/ice-lab/sim/jump.ts (<-57)`, `tools/ice-lab/sim/math.ts (<-48)`, `tools/ice-lab/app/pad.ts (<-28)`
- **key types**: `ReplayRecorder — tools/ice-lab/sim/replay.ts:315`, `IceGrid — tools/ice-lab/sim/ice.ts:31`, `Renderer — tools/ice-lab/app/draw.ts:210`, `Panel — tools/ice-lab/app/panel.ts:145`, `ReplayPlayer — tools/ice-lab/sim/replay.ts:429`, `ComboTracker — tools/ice-lab/sim/combo.ts:47`
- **external deps**: `node:assert`, `node:test`, `node:fs`, `node:path`, `res:`, `node:url`, `node:child_process`, `node:os`, `node:crypto`, `limits`

### LiMp  `402128798de0`
- **languages**: typescript 185f/30524loc, python 68f/17691loc, julia 37f/4577loc, markdown 18f/1253loc, shell 7f/863loc, javascript 2f/32loc
- **readme**: - **Entropy Calculation**: Based on SHA256 hash of token values - **Dynamic Branching**: Nodes can create new child nodes based on token state - **Entropy Limits**: Stop processing when entropy reaches certain thresholds - **Memory Trackin…
- **entrypoints**: `run_tests.py`, `backend/api.py`, `demo_example.py`, `julia_client.py`, `example_usage.py`, `matrix_processor.py`, `entropy_engine/cli.py`, `test_entropy_engine.py`
- **hub modules**: `backend/utils/logger.py (<-18)`, `backend/agentpress/tool.py (<-14)`, `backend/agentpress/thread_manager.py (<-12)`, `frontend/src/components/thread/tool-views/types.ts (<-10)`, `frontend/src/components/thread/tool-views/utils.ts (<-10)`, `backend/sandbox/sandbox.py (<-9)`
- **key types**: `EntropyEngine — demo_example.py:86`, `EntropyEngine — entropy_engine.py:85`, `EntropyEngine — entropy_engine/core.py:79`, `Token — demo_example.py:10`, `Token — entropy_engine.py:5`, `Token — entropy_engine/core.py:5`
- **external deps**: `@/components`, `@/lib`, `react`, `next`, `lucide-react`, `torch`, `@/hooks`, `motion`, `dotenv`, `next-themes`

### NuRea_sim  `19ef2c1ec3d5`
- **languages**: python 77f/6886loc, julia 12f/2756loc, markdown 12f/1585loc, javascript 5f/706loc, shell 8f/370loc, typescript 4f/62loc
- **readme**: **Advanced Nuclear Physics Simulation & AI-Powered Analysis Platform**
- **entrypoints**: `run_opt.py`, `set_env.py`, `evaluate.py`, `pretrain.py`, `test_adapter.py`, `setup_environment.py`, `matrix_integration.py`, `matrix_orchestrator.py`
- **hub modules**: `carryon/carryon-mvp-ccl/server/app/db/__init__.py (<-7)`, `carryon/carryon-mvp-20250810-210056/server/app/db.py (<-4)`, `carryon/carryon-mvp-ccl/server/app/retrieval/vector_index.py (<-4)`, `dataset/common.py (<-4)`, `carryon/carryon-mvp-20250810-210056/server/app/config.py (<-3)`, `carryon/carryon-mvp-ccl/server/app/tone/alignment.py (<-3)`
- **key types**: `Settings — Enhanced _Matrix_Orchestrator.py:44`, `Settings — carryon/carryon-mvp-20250810-210056/server/app/config.py:4`, `Settings — carryon/carryon-mvp-ccl/server/app/config/settings.py:4`, `Settings — matrix_orchestrator.py:44`, `EntropyEngine — entropy engine/ent/entropy_engine.py:74`, `Token — entropy engine/ent/entropy_engine.py:5`
- **external deps**: `fastapi`, `JSON3`, `pydantic`, `torch`, `Statistics`, `numpy`, `HTTP`, `sqlmodel`, `Random`, `models`

### Rubies-of_Love  `751440b15d5b`
- **languages**: python 1f/78loc, markdown 1f/56loc
- **readme**: This is a Ruby on Rails implementation of the CarryOn system for portable AI identity.
- **entrypoints**: `test_api.py`

### TEmp-oral_vectraxice  `c70ac75fccdc`
- **languages**: python 22f/2080loc, julia 3f/790loc, typescript 15f/220loc, markdown 4f/149loc, shell 1f/70loc
- **readme**: **Tagline:** Keep your AI *you* across model updates, app wipes, and devices.
- **entrypoints**: `Sfpud.py`, `server.jl`, `ta_uls_llm.py`, `server/app/main.py`, `sweet integrated_training_system.py`, `chaos_rag_single/chaos_rag_single/server.jl`
- **hub modules**: `server/app/config.py (<-4)`, `server/app/db.py (<-3)`, `server/app/core/soulpack.py (<-2)`, `server/app/main.py (<-2)`, `apps/desktop/src/lib/types.ts (<-1)`, `apps/desktop/src/routes/Adapters/Detail.tsx (<-1)`
- **key types**: `SecurityConfig — Sfpud.py:247`, `Settings — server/app/config.py:4`, `DualLLMOrchestrator — Sfpud.py:575`, `FrameConfig — Sfpud.py:125`, `ModulationScheme — Sfpud.py:75`, `ModConfig — Sfpud.py:110`
- **external deps**: `fastapi`, `Statistics`, `torch`, `sqlmodel`, `Random`, `HTTP`, `numpy`, `JSON3`, `Crypto`, `wave`
- **warning**: 1 python file(s) fail to parse

### The_St  `1ce72763b361`
- **languages**: python 96f/13125loc, markdown 7f/1325loc, java 2f/941loc, javascript 6f/721loc, shell 1f/12loc
- **readme**: A local place to explore your own words and digital footprint. Start with personal notes and journals, then compare with browser history or exports from TikTok, YouTube, Instagram, X/Twitter, Spotify, Reddit, Amazon, or your device's own s…
- **entrypoints**: `x.py`, `app.py`, `notes.py`, `usage.py`, `amazon.py`, `mirror.py`, `reddit.py`, `tiktok.py`
- **hub modules**: `remembrance-consent/app/consent/constants.py (<-28)`, `remembrance-consent/app/consent/models.py (<-19)`, `mirror.py (<-18)`, `remembrance-consent/app/db.py (<-17)`, `remembrance-consent/app/consent/audit.py (<-16)`, `remembrance-consent/app/consent/tokens.py (<-16)`
- **key types**: `Response — remembrance-consent/app/consent/schemas.py:25`, `Settings — remembrance-consent/app/config.py:21`, `Request — remembrance-consent/app/consent/schemas.py:21`, `AuditEventType — remembrance-consent/app/consent/constants.py:107`, `ConsentAction — remembrance-consent/app/consent/constants.py:70`, `ApiError — remembrance-consent/app/errors.py:11`
- **external deps**: `sqlalchemy`, `numpy`, `fastapi`, `pytest`, `unittest`

### VIbe_coder_9xk1ll  `b8908f41a88f`
- **languages**: python 102f/32189loc, markdown 72f/15453loc, javascript 9f/2056loc, java 1f/248loc
- **readme**: A Python puzzle game that scores your code on three axes — **is it right, how fast did you write it, and how much work does it actually do** — and adapts its challenges to how you already write code.
- **entrypoints**: `tests/test_ui.py`, `vibecoder/cli.py`, `tests/test_keys.py`, `tests/test_term.py`, `tools/web/build.py`, `tools/web/serve.py`, `tools/web/smoke.py`, `tests/test_daily.py`
- **hub modules**: `vibecoder/models.py (<-49)`, `vibecoder/levels/__init__.py (<-19)`, `vibecoder/runner.py (<-16)`, `vibecoder/ui.py (<-15)`, `vibecoder/__init__.py (<-13)`, `vibecoder/mastery.py (<-10)`
- **key types**: `Session — vibecoder/session.py:74`, `TestCase — vibecoder/models.py:66`, `Level — vibecoder/models.py:89`, `Source — vibecoder/models.py:23`, `Renderer — vibecoder/ui.py:235`, `RunResult — vibecoder/models.py:169`
- **external deps**: `unittest`, `android`

### al-ULS  `4437ecdb7f1a`
- **languages**: python 2f/817loc, julia 2f/558loc, markdown 2f/469loc, shell 1f/86loc
- **readme**: A high-performance Julia-based microservice providing advanced matrix optimization, stability analysis, and entropy regularization inspired by **TA ULS (Topology-Aware Uncertainty Learning Systems)**. Designed for integration with Python w…
- **entrypoints**: `al-ULs.Py`, `test_ta_uls.py`
- **key types**: `TAULSControlUnit — al-ULs.Py:69`, `KFPLayer — al-ULs.Py:50`, `JuliaClient — al-ULs.Py:22`, `StabilityAwareLoss — al-ULs.Py:185`, `TAULSTrainer — al-ULs.Py:325`, `TAULSTrainingConfig — al-ULs.Py:118`
- **external deps**: `torch`, `numpy`, `requests`, `importlib`, `MultivariatePolynomials`, `LinearAlgebra`, `JSON`, `Random`, `HTTP`, `Statistics`

### astro_caster  `4ecf721720b6`
- **languages**: typescript 201f/36863loc, python 119f/29225loc, markdown 54f/18567loc, javascript 6f/2599loc, shell 10f/1169loc, java 5f/317loc
- **readme**: <p align="center"> <img src="docs/screenshots/01-threshold-first-visit.png" alt="A first visit: the live sky, computed now — no account, nothing to dismiss" width="90%"> </p>
- **entrypoints**: `backend/main.py`, `backend/tools/dev.py`, `backend/verify_ai.py`, `backend/evals/runner.py`, `backend/tools/backup.py`, `backend/tools/unlock.py`, `backend/delivery_audit.py`, `backend/tests/test_chart.py`
- **hub modules**: `frontend/e2e/helpers.ts (<-42)`, `backend/models.py (<-39)`, `backend/ephemeris.py (<-35)`, `backend/entitlements.py (<-31)`, `backend/main.py (<-28)`, `frontend/src/store/useStore.ts (<-27)`
- **key types**: `ChartRequest — backend/models.py:24`, `ApiError — frontend/src/api/client.ts:71`, `Case — backend/evals/checks.py:66`, `TarotReadingRequest — backend/tarot_models.py:92`, `ChartResponse — backend/models.py:166`, `ChartValidationError — packages/astra-core/src/resonarium.ts:67`
- **external deps**: `react`, `node:assert`, `node:test`, `fastapi`, `pytest`, `node:path`, `node:fs`

### beatmI  `7092ad436337`
- **languages**: python 3f/1547loc, markdown 1f/169loc
- **readme**: A beat instrument for producers who can program drums fine but keep writing melodies that wander and rhythms that don't lock together.
- **entrypoints**: `twin/analyze.py`, `twin/braille.py`, `twin/test_analyze.py`
- **hub modules**: `twin/analyze.py (<-1)`, `twin/braille.py (<-1)`
- **external deps**: `numpy`, `scipy`

### bigLIMp  `c7e7c321d3af`
- **languages**: python 18f/2592loc, markdown 8f/2235loc, julia 4f/371loc, javascript 4f/178loc, typescript 6f/120loc, shell 1f/15loc
- **entrypoints**: `qincrs.py`, `tritto.py`, `Qwen_nstcp.py`, `lattice-being/libs/ts-sdk/index.ts`, `lattice-being/services/exo-lattice/index.ts`, `lattice-being/services/emotion-layer/main.py`, `lattice-being/services/quantum-sim/server.jl`, `lattice-being/services/time-crystals/main.py`
- **hub modules**: `lattice-being/apps/control-panel/src/App.tsx (<-1)`, `lattice-being/services/exo-lattice/verify.ts (<-1)`
- **key types**: `BiometricStream — Qwen_nstcp.py:28`, `CoherenceState — Qwen_nstcp.py:35`, `LearningPhase — Qwen_nstcp.py:43`, `BiometricSignature — Qwen_nstcp.py:53`, `ConsciousnessState — Qwen_nstcp.py:76`, `FrequencyBand — qincrs.py:32`
- **external deps**: `fastapi`, `opentelemetry`, `pydantic`, `numpy`, `scipy`, `uvicorn`, `react`, `httpx`, `HTTP`, `qiskit`
- **warning**: 3 python file(s) fail to parse

### carryon  `cc90487ae105`
- **languages**: python 24f/5157loc, typescript 25f/105loc, markdown 1f/15loc
- **entrypoints**: `sydv.py`, `server/app/main.py`
- **hub modules**: `server/app/db.py (<-4)`, `server/app/config.py (<-3)`, `server/app/main.py (<-2)`, `apps/desktop/src/App.tsx (<-1)`, `server/app/core/soulpack.py (<-1)`, `server/app/retrieval/graph_store.py (<-1)`
- **key types**: `SecurityConfig — sydv.py:2684`, `SecurityConfig — sydv.py:4116`, `Settings — server/app/config.py:4`, `DualLLMOrchestrator — sydv.py:3012`, `DualLLMOrchestrator — sydv.py:4444`, `FrameConfig — sydv.py:2562`
- **external deps**: `fastapi`, `Crypto`, `wave`, `numpy`, `pydantic`, `scipy`, `matplotlib`, `soundfile`, `sqlmodel`, `torch`
- **warning**: 1 python file(s) fail to parse

### coincidense_app  `326565a92e16`
- **languages**: python 15f/4591loc, markdown 9f/1460loc
- **readme**: [![tests](https://github.com/9x25dillon/coincidense_app/actions/workflows/tests.yml/badge.svg)](https://github.com/9x25dillon/coincidense_app/actions/workflows/tests.yml) [![license: MIT](https://img.shields.io/badge/license-MIT-0f766e.svg…
- **entrypoints**: `coincidence/cli.py`, `tests/test_coincidence.py`, `examples/make_synthetic.py`
- **hub modules**: `coincidence/boundary.py (<-6)`, `coincidence/grid.py (<-6)`, `coincidence/layers.py (<-5)`, `coincidence/analysis.py (<-4)`, `coincidence/loading.py (<-3)`, `coincidence/nulls.py (<-3)`
- **key types**: `Layer — coincidence/layers.py:21`, `Grid — coincidence/grid.py:21`, `Boundary — coincidence/boundary.py:33`, `BoundaryError — coincidence/boundary.py:28`, `LoadError — coincidence/loading.py:32`, `Style — coincidence/console.py:28`
- **external deps**: `csv`, `textwrap`, `html`

### engapp  `d74cdae378fc`
- **languages**: javascript 15f/3495loc, python 2f/1582loc, markdown 1f/286loc
- **readme**: An intelligent writing assistant with live focus tracking, synonym suggestions, weak word detection, and advanced text analysis.
- **entrypoints**: `Qwen_nstcp.py`, `holographic memory.py`
- **hub modules**: `src/config/settings.js (<-7)`, `src/utils/textProcessing.js (<-4)`, `src/utils/posDetection.js (<-3)`, `src/app.js (<-1)`, `src/modules/EditorController.js (<-1)`, `src/modules/UIManager.js (<-1)`
- **key types**: `BiometricStream — Qwen_nstcp.py:28`, `CoherenceState — Qwen_nstcp.py:35`, `LearningPhase — Qwen_nstcp.py:43`, `BiometricSignature — Qwen_nstcp.py:53`, `Settings — src/config/settings.js:205`, `ConsciousnessState — Qwen_nstcp.py:76`
- **external deps**: `scipy`, `vitest`, `numpy`, `torch`, `matplotlib`, `vite`

### enjoypy  `f0a15a81572f`
- **languages**: markdown 137f/10921loc, python 9f/2252loc
- **readme**: Integrate the DeepSeek API into popular softwares. Access [DeepSeek Open Platform](https://platform.deepseek.com/) to get an API key.
- **entrypoints**: `main.py`, `test_rag.py`, `entropy_cli.py`, `example_usage.py`, `test_entropy_system.py`
- **hub modules**: `julia_client.py (<-4)`, `utils.py (<-3)`, `core.py (<-2)`, `models.py (<-2)`, `entropy_cli.py (<-1)`
- **key types**: `EntropyEngine — core.py:199`, `Token — core.py:13`, `EntropyNode — core.py:133`, `JuliaClient — julia_client.py:7`, `ChunkMetadata — main.py:39`, `ChunkedResponse — main.py:45`
- **external deps**: `requests`, `fastapi`, `pydantic`, `openai`
- **warning**: 1 python file(s) fail to parse

### kgirl  `00232d485e04`
- **languages**: python 261f/120733loc, markdown 91f/40644loc, julia 8f/1945loc, shell 11f/564loc, typescript 13f/187loc
- **readme**: [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/) [![Julia](https://img.shields.io/badge/Julia-1.9%2B-purple)](https://j…
- **entrypoints**: `dec6`, `quat5`, `revsu6`, `AOv6.py`, `AqmpU.py`, `Auric.py`, `qrnsx.py`, `QBCROM.py`
- **hub modules**: `advanced_embedding_pipeline/__init__.py (<-23)`, `src/kgirl/quantum/holographic_memory_system.py (<-12)`, `src/kgirl/harness/util.py (<-10)`, `src/kgirl/cognitive/neuro_symbolic_numbskull_adapter.py (<-9)`, `src/kgirl/core/signal_processing.py (<-8)`, `src/kgirl/llm/numbskull_dual_orchestrator.py (<-8)`
- **key types**: `Linear — src/kgirl/core/model.py:166`, `Linear — src/kgirl/neural/yarn_transformer.py:79`, `BiometricStream — (QNPCE) v3.0.py:285`, `BiometricStream — AOv6.py:175`, `BiometricStream — AqmpU.py:381`, `BiometricStream — AqmpU.py:1462`
- **external deps**: `numpy`, `scipy`, `torch`
- **warning**: 19 python file(s) fail to parse

### myAssistant  `12ff1fe2fed8`
- **languages**: python 8f/405loc, markdown 1f/42loc
- **readme**: myAssistant
- **entrypoints**: `demo.py`
- **hub modules**: `app/config.py (<-5)`, `app/history.py (<-1)`, `app/ollama_client.py (<-1)`, `app/template.py (<-1)`, `app/vector_store.py (<-1)`
- **external deps**: `sqlalchemy`, `fastapi`, `torch`, `dotenv`, `requests`, `jinja2`, `pickle`, `faiss`

### neurotronic_phase_caster  `be29d2c23e8a`
- **languages**: python 40f/29321loc, markdown 11f/4370loc
- **readme**: Neurotronic Phase Caster
- **entrypoints**: `=.py`, `V3.py`, `N=$.py`, `nuty.py`, `Bloom.py`, `LTAUR.py`, `freeq.py`, `mattrx.py`
- **hub modules**: `src/QABCr.py (<-7)`, `src/yhwh_soliton_field_physics.py (<-7)`, `src/nscts_coherence_trainer.py (<-6)`, `src/integration/eopiez_vectorizer.py (<-2)`, `src/thz_bio_evolutionary_engine_advanced.py (<-2)`, `src/thz_coherence_wearable_spec.py (<-2)`
- **key types**: `BiometricStream — Bloom.py:285`, `BiometricStream — QNPCE_v3.0.py:285`, `BiometricStream — QUANTUM BIO-COHERENCE RESONATOR: THE OCTITRICE MANIFESTATION.py:772`, `BiometricStream — Unified_Coherence_Engine.py:222`, `BiometricStream — mattrx.py:405`, `BiometricStream — src/nscts_coherence_trainer.py:29`
- **external deps**: `scipy`, `numpy`, `matplotlib`
- **warning**: 7 python file(s) fail to parse

### new_repository_30  `0ec5a7b64c41`
- **languages**: typescript 89f/29760loc, markdown 19f/4288loc, javascript 10f/1532loc, python 2f/636loc, java 1f/200loc, shell 4f/191loc
- **readme**: **Symmetry-first design tools for acoustic cell manipulation** — BAW in closed microfluidic channels, SAW on open piezoelectric substrate.
- **entrypoints**: `test/python/test_biosentinel.py`, `photometabolic_biosentinel_v3_unified.py`
- **hub modules**: `photometabolic_biosentinel_v3_unified.py (<-1)`
- **key types**: `Engine — app/resonarium/photometabolic.js:74`, `ChartValidationError — app/resonarium/natal_seed.js:67`, `Bridge — packaging/android/java/app/sonicdrifter/MainActivity.java:160`, `FibonacciEngine — photometabolic_biosentinel_v3_unified.py:66`, `Game — app/drifter.ts:392`, `IsodoseMapper — photometabolic_biosentinel_v3_unified.py:104`
- **external deps**: `node:assert`, `node:test`, `android`, `node:fs`, `java`, `node:path`, `numpy`, `node:url`, `node:os`

### numbskull  `b9ec4d2a65f7`
- **languages**: python 97f/26884loc, markdown 12f/3316loc, julia 1f/131loc
- **readme**: A comprehensive system integrating multiple advanced AI architectures including: - **Enhanced Dual LLM WaveCaster with TA ULS Integration** - Intelligent waveform generation and signal processing - **Emergent Cognitive Network** - Quantum-…
- **entrypoints**: `demo_basic.py`, `test_system.py`, `signal_processing.py`, `tauls_transformer.py`, `enhanced_wavecaster.py`, `test_enhanced_system.py`, `demo_emergent_network.py`, `dual_llm_orchestrator.py`
- **hub modules**: `advanced_embedding_pipeline/fractal_cascade_embedder.py (<-7)`, `advanced_embedding_pipeline/mathematical_embedder.py (<-7)`, `advanced_embedding_pipeline/semantic_embedder.py (<-7)`, `packages/wavecaster/src/wavecaster/fec/base.py (<-6)`, `packages/wavecaster/src/wavecaster/phy.py (<-6)`, `advanced_embedding_pipeline/hybrid_pipeline.py (<-5)`
- **key types**: `HybridConfig — advanced_embedding_pipeline/hybrid_pipeline.py:25`, `HybridEmbeddingPipeline — advanced_embedding_pipeline/hybrid_pipeline.py:50`, `SecurityConfig — signal_processing.py:100`, `SecurityConfig — tau_uls_wavecaster_enhanced.py:506`, `DualLLMOrchestrator — dual_llm_orchestrator.py:233`, `DualLLMOrchestrator — tau_uls_wavecaster_enhanced.py:988`
- **external deps**: `numpy`, `scipy`, `torch`, `matplotlib`
- **warning**: 4 python file(s) fail to parse

### orwells-egg  `306b0c0c5093`
- **languages**: python 32f/11334loc, julia 6f/2275loc, markdown 4f/1150loc, shell 1f/7loc
- **readme**: Chaos RAG/SQL + ML2 minimal scaffold
- **entrypoints**: `app.py`, `src.py`, `kgirl.py`, `cli/main.py`, `cli/repl.py`, `joeshoeaye.py`, `neutronics.py`, `orchestrator.py`
- **hub modules**: `cli/client.py (<-5)`, `cli/config.py (<-5)`, `coach.py (<-2)`, `db.py (<-2)`, `ds_adapter.py (<-2)`, `rfv.py (<-2)`
- **key types**: `SecurityConfig — DLmWaveCaster.py:247`, `DualLLMOrchestrator — DLmWaveCaster.py:575`, `FrameConfig — DLmWaveCaster.py:125`, `ModulationScheme — DLmWaveCaster.py:75`, `ModConfig — DLmWaveCaster.py:110`, `TAULSControlUnit — neutronics.py:2816`
- **external deps**: `rich`, `LinearAlgebra`, `Statistics`, `torch`, `numpy`, `Random`, `sqlalchemy`
- **warning**: 2 python file(s) fail to parse

### patern-coding  `6590849d6752`
- **languages**: python 16f/1682loc, markdown 8f/920loc, julia 3f/472loc, shell 2f/8loc
- **readme**: The new development path is **`auric/`**: a small language model trained from random weights, a persistent document/code index, explicit project memory, and a VibeCoder bridge for isolated coding exercises. Existing research projects below…
- **entrypoints**: `auric/cli.py`, `tests/test_auric.py`
- **hub modules**: `auric/tokenizer.py (<-7)`, `auric/training.py (<-6)`, `auric/model.py (<-5)`, `auric/workspace.py (<-4)`, `auric/data.py (<-3)`, `auric/memory.py (<-3)`
- **key types**: `MotifToken — MessageVectorizer/src/MessageVectorizer.jl:26`, `MessageVectorizer — MessageVectorizer/src/MessageVectorizer.jl:65`, `ByteTokenizer — auric/tokenizer.py:4`, `LanguageModel — auric/model.py:73`, `MessageState — MessageVectorizer/src/MessageVectorizer.jl:45`, `TrainConfig — auric/training.py:20`
- **external deps**: `torch`, `unittest`, `importlib`, `sqlite3`, `numpy`, `urllib`

### pogo-showdown  `f4067a765e9b`
- **languages**: typescript 63f/13177loc, javascript 8f/1682loc, shell 3f/291loc, markdown 4f/258loc, java 3f/49loc
- **readme**: History's icons are highschoolers with a pogo stick, and they're stuck in one cursed open world. Dig and build like Terraria, fight Chakan-style through four elemental realms to the Forever Gate, and do everything else there too: duel pog …
- **hub modules**: `src/game/realm/worldGen.ts (<-14)`, `src/game/realm/items.ts (<-12)`, `src/game/realm/tiles.ts (<-12)`, `src/game/systems/gamepad.ts (<-11)`, `src/game/data/characters.ts (<-10)`, `src/game/data/pogs.ts (<-9)`
- **key types**: `RealmPanel — src/game/realm/RealmPanel.ts:40`, `MusicManager — src/game/systems/music.ts:33`, `RealmArena — src/game/realm/RealmArena.ts:49`, `RealmDash — src/game/realm/RealmDash.ts:36`, `RealmDuel — src/game/realm/RealmDuel.ts:52`, `RealmExpedition — src/game/realm/RealmExpedition.ts:27`
- **external deps**: `phaser`, `node:assert`, `org`, `androidx`, `@capacitor/cli`, `vite`, `android`, `com`, `node:fs`

### shout  `158435d35066`
- **languages**: markdown 4f/6646loc, python 13f/4808loc, julia 3f/965loc, shell 1f/15loc
- **entrypoints**: `tests/run.py`, `python/ccl.py`, `patch_apply.py`, `dianne/python/api.py`, `dianne/julia/server.jl`, `python/tauls_trainer.py`, `python/ccl_julia_client.py`, `python/mock_al_uls_server.py`
- **hub modules**: `python/kgirl_bridge.py (<-1)`, `python/orwells_bridge.py (<-1)`
- **key types**: `TAULSControlUnit — python/tauls_trainer.py:72`, `KFPLayer — python/tauls_trainer.py:43`, `JuliaClient — python/tauls_trainer.py:18`, `CodeProj — dianne/julia/server.jl:220`, `StabilityAwareLoss — python/tauls_trainer.py:166`, `TAULSTrainer — python/tauls_trainer.py:250`
- **external deps**: `fastapi`, `numpy`, `LinearAlgebra`, `Statistics`, `torch`, `JSON3`, `Random`, `requests`
- **warning**: 1 python file(s) fail to parse

### soundy_thingy  `8e3e64ac02aa`
- **languages**: python 5f/2002loc, javascript 2f/574loc, markdown 3f/313loc
- **readme**: A collection of self-contained, browser-based audio instruments and tools focused on generative sound design, resonance, and ambient synthesis.
- **entrypoints**: `resonarium_cli_skeleton.py`, `resonarium_cli_engineering.py`, `resonarium/tests/test_biosentinel.py`, `resonarium/resonarium_biosentinel_cli.py`
- **hub modules**: `resonarium/natal_seed.py (<-2)`
- **key types**: `ChartValidationError — resonarium/natal_seed.js:62`, `ChartValidationError — resonarium/natal_seed.py:74`, `ResonariumCLI — resonarium_cli_engineering.py:76`, `ResonariumCLI — resonarium_cli_skeleton.py:71`, `Single — resonarium_cli_engineering.py:64`, `Single — resonarium_cli_skeleton.py:65`
- **external deps**: `rich`, `shlex`, `fs`, `path`

### substrate-comm  `6fbac6f327f5`
- **languages**: python 10f/1905loc, markdown 4f/460loc
- **readme**: > **Nature builds the forms energy can take.**
- **entrypoints**: `run_all.py`, `substrate/physics.py`, `substrate/symbols.py`, `substrate/bootstrap.py`, `substrate/framework.py`, `substrate/hierarchy.py`, `substrate/materials.py`, `substrate/renderers.py`
- **hub modules**: `substrate/symbols.py (<-7)`, `substrate/physics.py (<-3)`, `substrate/renderers.py (<-3)`, `substrate/__init__.py (<-1)`, `substrate/bootstrap.py (<-1)`, `substrate/framework.py (<-1)`
- **key types**: `Renderer — substrate/renderers.py:81`, `LayerParams — substrate/physics.py:28`, `SubstitutionSystem — substrate/symbols.py:35`, `BootstrapCode — substrate/bootstrap.py:38`, `AbsoluteLevel — substrate/renderers.py:92`, `AcousticDuration — substrate/renderers.py:113`
- **external deps**: `numpy`, `matplotlib`, `sympy`

