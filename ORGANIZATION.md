# Repository Organization

This document describes the organization of the kgirl repository after restructuring.

## 📁 Directory Structure

```
kgirl/
├── docs/                      # All documentation
│   ├── guides/               # User guides, quickstarts, setup instructions
│   ├── integration/          # Integration documentation
│   ├── technical/            # Technical reports, benchmarks, research findings
│   └── api/                  # API documentation
│
├── src/                       # Source code
│   ├── kgirl/                # Main Python package
│   │   ├── __init__.py       # Package initialization
│   │   ├── main.py           # Main entry point
│   │   ├── setup.py          # Setup configuration
│   │   ├── core/             # Core platform components
│   │   │   ├── ASPM_system.py
│   │   │   ├── Bloop.py, Brio.py, INST.py
│   │   │   ├── CoCo_0rg.py
│   │   │   ├── UCs.py, Sfpud.py, sydv.py, yyybet.py
│   │   │   ├── bridge_newthought_crd.py
│   │   │   ├── chaos_llm_integration.py
│   │   │   ├── distributed_knowledge_base.py
│   │   │   ├── entropy_engine.py
│   │   │   ├── infractence.py
│   │   │   ├── kernel.py, model.py
│   │   │   ├── limp_module_manager.py
│   │   │   ├── limp_numbskull_integration_map.py
│   │   │   ├── limps_eopiez_adapter.py
│   │   │   ├── limps_holographic_orchestrator.py
│   │   │   ├── logic_plague.py
│   │   │   ├── loom_bridge.py
│   │   │   ├── matrix_processor.py
│   │   │   ├── motif_engine.py
│   │   │   └── signal_processing.py
│   │   │
│   │   ├── quantum/          # Quantum-inspired modules
│   │   │   ├── holographic_memory_system.py
│   │   │   ├── lattice.py
│   │   │   ├── quantum_cognitive_processor.py
│   │   │   ├── quantum_holographic_knowledge_synthesis.py
│   │   │   ├── quantum_knowledge_database.py
│   │   │   ├── quantum_knowledge_processing.py
│   │   │   ├── quantum_limps_integration.py
│   │   │   ├── quantum_llm_interface.py
│   │   │   ├── yhwh_abcr_integration.py
│   │   │   ├── yhwh_demo_interactive.py
│   │   │   └── yhwh_soliton_field_physics.py
│   │   │
│   │   ├── cognitive/        # Cognitive processing
│   │   │   ├── CoCo_0rg.py
│   │   │   ├── advanced_cognitive_enhancements.py
│   │   │   ├── cognitive_integration_bridge.py
│   │   │   ├── emergent_cognitive_network.py
│   │   │   ├── evolutionary_communicator.py
│   │   │   ├── NEWTHOUGHT_WORKFLOW_ALGORITHM.py
│   │   │   ├── narrative_agent.py
│   │   │   ├── neuro_symbolic_engine.py
│   │   │   ├── neurosymbiotic_coherence_training.py
│   │   │   ├── recursive_ai_core.py
│   │   │   ├── recursive_ai_system.py
│   │   │   ├── recursive_cognitive_knowledge.py
│   │   │   ├── recursive_cognitive_system.py
│   │   │   └── unified_cognitive_orchestrator.py
│   │   │
│   │   ├── llm/              # LLM adapters and interfaces
│   │   │   ├── aipyapp_playground.py
│   │   │   ├── al_uls.py
│   │   │   ├── al_uls_client.py
│   │   │   ├── al_uls_ws_client.py
│   │   │   ├── aluls_numbskull_adapter.py
│   │   │   ├── cognitive_organism_numbskull_adapter.py
│   │   │   ├── coco_integrated_playground.py
│   │   │   ├── dual_llm_orchestrator.py
│   │   │   ├── emergent_network_numbskull_adapter.py
│   │   │   ├── enable_aluls_and_qwen.py
│   │   │   ├── evolutionary_numbskull_adapter.py
│   │   │   ├── llm_adapters.py
│   │   │   ├── llm_eval.py
│   │   │   ├── llm_orchestrator.py
│   │   │   ├── llm_training_adapter.py
│   │   │   ├── narrative_numbskull_adapter.py
│   │   │   ├── neuro_symbolic_numbskull_adapter.py
│   │   │   ├── newthought_hf_integration.py
│   │   │   ├── numbskull_dual_orchestrator.py
│   │   │   ├── play_aluls_qwen.py
│   │   │   ├── pytorch_components_numbskull_adapter.py
│   │   │   ├── signal_processing_numbskull_adapter.py
│   │   │   ├── ta_uls_llm.py
│   │   │   ├── tau_uls_wavecaster_enhanced.py
│   │   │   ├── tauls_model.py
│   │   │   ├── tauls_transformer.py
│   │   │   └── unified_quantum_llm_system.py
│   │   │
│   │   ├── embeddings/       # Embedding pipelines
│   │   │   └── fractal_resonance.py
│   │   │
│   │   ├── neural/           # Neural network components
│   │   │   ├── bciloop.py
│   │   │   ├── bi-inrefernce.py
│   │   │   ├── convert.py
│   │   │   ├── dianne_polyserve.py
│   │   │   ├── enhanced_graph_store.py
│   │   │   ├── enhanced_vector_index.py
│   │   │   ├── enhanced_wavecaster.py
│   │   │   ├── fp8_cast_bf16.py
│   │   │   ├── matrix_processor_adapter.py
│   │   │   ├── sweet integrated_training_system.py
│   │   │   ├── tdcs_enhanced_recovery.py
│   │   │   ├── thz_coherence_wearable_spec.py
│   │   │   ├── unitary_mixer.py
│   │   │   └── yarn_transformer.py
│   │   │
│   │   ├── api/              # API servers
│   │   │   ├── api.py
│   │   │   ├── bloom_backend.py
│   │   │   └── integrated_api_server.py
│   │   │
│   │   └── utils/            # Utility modules
│   │       ├── UNIFIED_COHERENCE_INTEGRATION_ALGORITHM.py
│   │       ├── config.py
│   │       ├── crd.py
│   │       ├── crypto.py
│   │       ├── db.py
│   │       ├── domain_mapping.py
│   │       ├── find_wallet_artifacts.py
│   │       ├── graph_store.py
│   │       ├── health.py
│   │       ├── memories.py
│   │       ├── memory_event.py
│   │       ├── phrain.py
│   │       ├── play.py
│   │       ├── prime.py
│   │       ├── qgi.py
│   │       ├── ranker.py
│   │       ├── retrieval.py
│   │       ├── soulpack.py
│   │       ├── soulpack_meta.py
│   │       ├── soulpacks.py
│   │       ├── stub_modules.py
│   │       ├── suggestions.py
│   │       ├── tools.py
│   │       └── vector_index.py
│   │
│   └── julia/                # Julia source code
│       ├── Server.jl
│       ├── julia_server_script.jl
│       ├── mqt.jl
│       ├── quantum_memory.jl
│       ├── quantum_neural_demo.jl
│       ├── server.jl
│       ├── setup_limps_service.jl
│       └── vibrational_lattice.jl
│
├── scripts/                   # Executable scripts
│   ├── setup/                # Setup and installation scripts
│   │   ├── Dockerfile
│   │   ├── INSTALL_ALL_SERVICES.sh
│   │   ├── Makefile
│   │   ├── OLLAMA_SETUP_GUIDE.sh
│   │   ├── SIMPLE_COPY_PASTE.fish
│   │   ├── START_NOW.sh
│   │   ├── activate, activate.csh, Activate.ps1
│   │   ├── install_fluidsynth_with_soundfonts_osx.sh
│   │   ├── ram_monitor.sh
│   │   ├── run.sh
│   │   ├── start_all_services.sh
│   │   ├── start_lfm2.sh
│   │   ├── start_limps.sh
│   │   └── start_qwen.sh
│   │
│   ├── demo/                 # Demo and testing scripts
│   │   ├── adapter_integration_demo.py
│   │   ├── benchmark_full_stack.py
│   │   ├── benchmark_integration.py
│   │   ├── complete_adapter_suite_demo.py
│   │   ├── demo.py
│   │   ├── demo_adapter.py
│   │   ├── demo_basic.py
│   │   ├── demo_consensus.py
│   │   ├── demo_emergent_system.py
│   │   ├── demo_integrated_system.py
│   │   ├── full_system_demo.py
│   │   ├── master_playground.py
│   │   ├── playground.py
│   │   ├── quantum_knowledge_demo.py
│   │   ├── quantum_limps_demo.py
│   │   ├── quick_demo.py
│   │   ├── recursive_playground.py
│   │   ├── run_demo.py
│   │   ├── simple_integrated_wavecaster_demo.py
│   │   ├── test_emergent_system.py
│   │   ├── test_enhanced_system.py
│   │   ├── test_local_llm.py
│   │   ├── test_newthought.py
│   │   ├── test_newthought_standalone.py
│   │   ├── test_system.py
│   │   ├── verify_all_components.py
│   │   └── verify_integration.py
│   │
│   └── workflows/            # Workflow orchestrators
│       ├── complete_integration_orchestrator.py
│       ├── complete_integration_runner.py
│       ├── complete_system_integration.py
│       ├── complete_unified_platform.py
│       ├── generate.py
│       ├── generate_graphical_abstract.py
│       ├── integrated_wavecaster_runner.py
│       ├── integration_health_check.py
│       ├── master_data_flow_orchestrator.py
│       ├── research_simulation.py
│       ├── run_integrated_workflow.py
│       ├── upload_newthought.fish
│       └── upload_to_hf.py
│
├── tests/                     # Test files
│   ├── smoke_adapters.py
│   ├── test_llm_orchestrator.py
│   └── test_tauls_evaluator.py
│
├── configs/                   # Configuration files
│   ├── Project.toml
│   ├── config_16B.json
│   ├── config_236B.json
│   ├── config_671B.json
│   ├── config_lfm2.json
│   ├── config_v3.1.json
│   ├── docker-compose.yml
│   ├── portable.yml
│   ├── pyvenv.cfg
│   ├── requirements-dev.txt
│   ├── requirements-extra.txt
│   ├── requirements.txt
│   └── requirements.txt.backup
│
├── examples/                  # Example usage
│   ├── README.md
│   ├── ask_client.py
│   ├── chaos_rag_client.py
│   └── rerank_client.py
│
├── models/                    # Model definitions
│   └── newthought_model/
│       ├── README.md
│       ├── USAGE_EXAMPLES.md
│       ├── config.json
│       └── newthought.py
│
├── research/                  # Research papers and LaTeX
│   ├── ADD_TO_PAPER_cosmology_citation.tex
│   ├── ALGORythm.tex
│   ├── Algorithm.TX
│   ├── CN118374327A.pdf
│   ├── Cognitive_Renewal_Dynamics_FINAL.tex
│   ├── NSF 25-509_ Emerging Mathematics in Biology (eMB) _ NSF - National Science Foundation.PDF
│   ├── Palgorithms.tex
│   ├── eGoG_OMEGAPROTOCOL_2025-11-04_145346.pdf
│   └── newfile.TX
│
├── data/                      # Data files and databases
│   ├── 22e94c54cbf7934afd684754b7b84513f04f1d
│   ├── 9x25dillon_LiMp_ luck
│   ├── CodeChunks.db
│   ├── bc-c5221a6f-1fa6-4e1d-9227-515f76569ff6-e270
│   ├── benchmark_full_stack_results.json
│   ├── benchmark_results.json
│   ├── carryon.db
│   ├── carryon.zip
│   ├── chaos_rag_single2.zip
│   ├── demo_results.json
│   ├── integration_map.json
│   ├── limp_module_status.json
│   ├── memory_event.schema.json
│   ├── soulpack.schema.json
│   ├── suo
│   └── yhwh_soliton_evolution.png
│
├── frontend/                  # Frontend TSX/React components
│   ├── App.tsx
│   ├── Backup.tsx
│   ├── ConsentPrivacy.tsx
│   ├── Dashboard.tsx
│   ├── Detail.tsx
│   ├── ImportSources.tsx
│   ├── List.tsx
│   ├── PersonaBasics.tsx
│   ├── PrimerPreview.tsx
│   ├── ReviewPin.tsx
│   ├── Start.tsx
│   ├── Studio.tsx
│   └── Timeline.tsx
│
├── advanced_embedding_pipeline/ # Advanced embedding pipeline (existing)
│   ├── INTEGRATION_SUMMARY.md
│   ├── README.md
│   ├── __init__.py
│   ├── demo.py
│   ├── fractal_cascade_embedder.py
│   ├── hybrid_pipeline.py
│   ├── integration_test.py
│   ├── mathematical_embedder.py
│   ├── optimizer.py
│   ├── requirements.txt
│   ├── semantic_embedder.py
│   ├── setup.py
│   └── simple_test.py
│
├── outputs/                   # Output files
│   └── polyserve_demo.wav
│
├── .github/workflows/         # GitHub Actions workflows
│   ├── publish.yml
│   └── stale.yml
│
├── LICENSE                    # License files
├── LICENSE-CODE
├── README.md                  # Main README
└── ORGANIZATION.md           # This file

```

## 📋 Quick Reference

### Find Documentation
- **Getting Started**: `docs/guides/QUICKSTART.md`
- **Local LLM Setup**: `docs/guides/LOCAL_LLM_SETUP.md`
- **API Reference**: `docs/api/API.md`
- **Integration Guides**: `docs/integration/`
- **Technical Reports**: `docs/technical/`

### Find Source Code
- **Core Platform**: `src/kgirl/core/`
- **Quantum Modules**: `src/kgirl/quantum/`
- **Cognitive Systems**: `src/kgirl/cognitive/`
- **LLM Adapters**: `src/kgirl/llm/`
- **Neural Networks**: `src/kgirl/neural/`
- **API Servers**: `src/kgirl/api/`
- **Julia Code**: `src/julia/`

### Run Demos
- **All Demos**: `scripts/demo/`
- **Quick Demo**: `scripts/demo/quick_demo.py`
- **Full System**: `scripts/demo/full_system_demo.py`

### Setup & Installation
- **Setup Scripts**: `scripts/setup/`
- **Dependencies**: `configs/requirements.txt`
- **Docker**: `scripts/setup/Dockerfile`

## 🔄 Migration Notes

### Import Path Changes

After reorganization, Python imports need to be updated:

**Old:**
```python
from cognitive_integration_bridge import CognitiveBridge
from quantum_knowledge_processing import QuantumKnowledge
```

**New:**
```python
from kgirl.cognitive.cognitive_integration_bridge import CognitiveBridge
from kgirl.quantum.quantum_knowledge_processing import QuantumKnowledge
```

### Running Scripts

**Old:**
```bash
python demo_integrated_system.py
```

**New:**
```bash
python scripts/demo/demo_integrated_system.py
# OR from project root:
python -m scripts.demo.demo_integrated_system
```

## 📦 Package Structure

The `src/kgirl/` directory is now a proper Python package with:
- `__init__.py` in all subdirectories
- Clear module organization by functionality
- Consistent naming conventions

## 🎯 Benefits

1. **Clarity**: Easy to find files by category
2. **Scalability**: Room for growth in each category
3. **Maintainability**: Clear separation of concerns
4. **Best Practices**: Follows Python package conventions
5. **Documentation**: All docs in one place
6. **Testing**: Dedicated test directory
7. **Configuration**: Centralized config management

## 📝 Next Steps

1. Update import statements in Python files
2. Update script shebangs and paths
3. Test all functionality
4. Update CI/CD pipelines if needed
5. Update documentation with new paths

---

For questions or issues with the new structure, please refer to the main [README.md](README.md) or open an issue.
