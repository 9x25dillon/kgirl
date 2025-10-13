# Service Startup Guide for Full Stack Benchmarking

This guide shows you how to start all services needed for comprehensive benchmarking of the Numbskull + LFM2-8B-A1B integration.

## Services Overview

| Service | Port | Purpose | Required |
|---------|------|---------|----------|
| **LFM2-8B-A1B** | 8080 | Local LLM inference | ✅ Yes |
| **Eopiez** | 8001 | Semantic embeddings | 🔶 Optional |
| **LIMPS** | 8000 | Mathematical embeddings | 🔶 Optional |
| **Fractal** | N/A | Local (no service needed) | ✅ Always available |

## Quick Start: All Services

### Terminal 1: LFM2-8B-A1B (Required for LLM benchmarks)

```bash
# Option A: llama.cpp server (recommended)
llama-server \
  --model /path/to/LFM2-8B-A1B.gguf \
  --port 8080 \
  --ctx-size 8192 \
  --n-gpu-layers 35 \
  --threads 8

# Option B: text-generation-webui
cd /path/to/text-generation-webui
python server.py \
  --model LFM2-8B-A1B \
  --api \
  --port 5000
# Then update config_lfm2.json to use port 5000 and mode "textgen-webui"

# Option C: vLLM (OpenAI-compatible)
vllm serve /path/to/LFM2-8B-A1B \
  --port 8080 \
  --dtype auto
```

### Terminal 2: Eopiez (Optional - for semantic embeddings)

```bash
cd ~/aipyapp/Eopiez
python api.py --port 8001

# Or if in different location:
cd /path/to/Eopiez
python api.py --port 8001
```

### Terminal 3: LIMPS (Optional - for mathematical embeddings)

```bash
cd ~/aipyapp/9xdSq-LIMPS-FemTO-R1C/limps
julia --project=. -e 'using LIMPS; LIMPS.start_limps_server(8000)'

# Or start REPL first:
cd ~/aipyapp/9xdSq-LIMPS-FemTO-R1C/limps
julia --project=.
# Then in Julia REPL:
# using LIMPS
# LIMPS.start_limps_server(8000)
```

### Terminal 4: Run Benchmarks

```bash
cd /home/kill/LiMp

# Check service status first
python benchmark_full_stack.py

# Run with all available services
python benchmark_full_stack.py --all

# Or run specific tests
python benchmark_full_stack.py --with-llm  # LLM integration only
python benchmark_full_stack.py --services-only  # Services only
```

## Benchmark Command Reference

### Basic Benchmarks (No external services)

```bash
cd /home/kill/LiMp

# Quick benchmark (fractal only, ~30 seconds)
python benchmark_integration.py --quick

# Comprehensive benchmark (fractal only, ~2 minutes)
python benchmark_integration.py

# Save to custom file
python benchmark_integration.py --output my_results.json
```

### Full Stack Benchmarks (With services)

```bash
# Check which services are available
python benchmark_full_stack.py

# Test semantic embeddings (requires Eopiez)
python benchmark_full_stack.py --services-only

# Test end-to-end with LFM2 (requires LFM2-8B-A1B)
python benchmark_full_stack.py --with-llm

# Test everything (requires all services)
python benchmark_full_stack.py --all
```

## Service Health Check

Before running benchmarks, verify services are running:

```bash
# Check LFM2-8B-A1B (llama-cpp mode)
curl http://127.0.0.1:8080/health

# Check LFM2-8B-A1B (OpenAI-compatible)
curl http://127.0.0.1:8080/v1/models

# Check Eopiez
curl http://127.0.0.1:8001/health

# Check LIMPS
curl http://127.0.0.1:8000/health

# Or use the verification script
python verify_integration.py
```

## Minimal Setup (Fractal Only)

If you just want to test without external services:

```bash
cd /home/kill/LiMp

# No services needed! Works out of the box
python benchmark_integration.py --quick
```

**Result**: Sub-10ms embeddings with 100% success rate using local fractal embeddings.

## Recommended Setup (LFM2 + Fractal)

For end-to-end LLM testing without external embedding services:

**Terminal 1**: Start LFM2-8B-A1B
```bash
llama-server --model /path/to/LFM2-8B-A1B.gguf --port 8080 --ctx-size 8192
```

**Terminal 2**: Run benchmarks
```bash
cd /home/kill/LiMp
python benchmark_full_stack.py --with-llm
```

**Result**: Full dual LLM orchestration with fractal embeddings.

## Full Setup (All Services)

For comprehensive testing with all embedding types:

**Terminal 1**: LFM2-8B-A1B
```bash
llama-server --model /path/to/LFM2-8B-A1B.gguf --port 8080 --ctx-size 8192
```

**Terminal 2**: Eopiez
```bash
cd ~/aipyapp/Eopiez && python api.py --port 8001
```

**Terminal 3**: LIMPS
```bash
cd ~/aipyapp/9xdSq-LIMPS-FemTO-R1C/limps
julia --project=. -e 'using LIMPS; LIMPS.start_limps_server(8000)'
```

**Terminal 4**: Run benchmarks
```bash
cd /home/kill/LiMp
python benchmark_full_stack.py --all
```

**Result**: Full hybrid system with semantic, mathematical, and fractal embeddings.

## Troubleshooting

### LFM2-8B-A1B won't start

**Issue**: "CUDA out of memory" or similar

**Solution**:
```bash
# Reduce GPU layers or use CPU only
llama-server \
  --model /path/to/LFM2-8B-A1B.gguf \
  --port 8080 \
  --n-gpu-layers 0  # CPU only
  # or
  --n-gpu-layers 20  # Fewer GPU layers
```

### Eopiez not found

**Issue**: Eopiez directory doesn't exist

**Solution**: Fractal embeddings work without Eopiez. Update config to use fractal-only:
```json
{
  "use_semantic": false,
  "use_mathematical": false,
  "use_fractal": true
}
```

### LIMPS not found

**Issue**: LIMPS service not available

**Solution**: System works without LIMPS using local mathematical processing or fractal embeddings.

### Port already in use

**Issue**: "Address already in use"

**Solution**:
```bash
# Find process using port
lsof -i :8080  # Or :8001, :8000

# Kill process
kill -9 <PID>

# Or use different port and update config
```

## Performance Expectations

### With No External Services (Fractal Only)
- **Latency**: 5-10ms per embedding
- **Throughput**: 100-185 samples/second
- **Quality**: Good for general purpose

### With Eopiez (Semantic)
- **Latency**: 50-200ms per embedding (network + model)
- **Throughput**: 5-20 samples/second
- **Quality**: Excellent for semantic understanding

### With LIMPS (Mathematical)
- **Latency**: 100-500ms per expression
- **Throughput**: 2-10 samples/second
- **Quality**: Excellent for mathematical content

### With LFM2-8B-A1B (Full Pipeline)
- **Latency**: 2-5 seconds per query (LLM dominates)
- **Embedding overhead**: <1% of total time
- **Quality**: Production-ready

## Benchmark Result Files

After running benchmarks, you'll find:

- **`benchmark_results.json`** - Quick benchmark results (fractal only)
- **`benchmark_full_stack_results.json`** - Full stack results (all services)
- **`BENCHMARK_ANALYSIS.md`** - Analysis and recommendations

## Next Steps

1. **Start with minimal setup** (fractal only) to verify system works
2. **Add LFM2-8B-A1B** for end-to-end testing
3. **Optionally add Eopiez/LIMPS** for full hybrid embeddings
4. **Run comprehensive benchmarks** with all services
5. **Review results** in generated JSON and markdown files

---

**Quick Command Summary**:

```bash
# 1. Minimal test (no services)
python benchmark_integration.py --quick

# 2. Check service status
python benchmark_full_stack.py

# 3. Full benchmark (with services)
python benchmark_full_stack.py --all

# 4. View results
cat benchmark_full_stack_results.json | python -m json.tool
cat BENCHMARK_ANALYSIS.md
```

**Tip**: Start services in separate terminal tabs/windows for easy management!

