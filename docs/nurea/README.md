# NuRea matrix backend: real operators behind NuRea_sim's `/optimize`

NuRea_sim's orchestrator (`matrix_orchestrator.py`, `JuliaBackend`) calls a matrix service on port 9000. kgirl now provides that service three ways, all with the same formulas:

| Piece | File | Runs here? |
|---|---|---|
| numpy reference and client (falls back to local math when no server answers) | `src/kgirl/nurea_backend.py` | yes |
| stdlib HTTP server, no Julia needed: `python -m kgirl.nurea_backend serve [port]` | same | yes, tested end to end |
| Julia server: `JULIA_PORT=9000 julia src/julia/nurea_backend.jl` (packages HTTP, JSON3) | `src/julia/nurea_backend.jl` | **not executed**: the proxy blocks Julia downloads in this container (L1: written to mirror the reference) |

## What was wrong upstream

The Julia servers shipped with NuRea_sim (`backends/julia/server.jl`, `hrm_julia_backend/server.jl`), and the orchestrator's local fallbacks, had these defects:

| Where | Defect | Here |
|---|---|---|
| `chebyshev_projection` | Multiplied the matrix by 0.95 | Least-squares fit of T₀…T_d per row, then reconstruct; optional coefficient sparsity |
| `rank` | Zeroed small entries one by one (no SVD) | Singular-value soft threshold U·diag(max(σ−τ, 0))·Vᵀ, the prox of τ‖·‖_* |
| Missing `matrix` | Replaced by `randn` data | HTTP 400 |
| `/optimize` reply | Lacked `objective`, `matrix_opt`, `iterations`, `meta`, so the orchestrator's `OptimizeResponse(**data)` could not parse it | Both field sets returned |
| `POST /health`, `/entropy`, `/chebyshev_project`, dispatch `ping` / `entropy` / `optimize` | Not implemented, so the client always fell back | Implemented |
| `entropy_local` | Used `np.histogram(density=True)` values as probabilities (densities, so the entropy could go negative) | p = counts / total |
| `chebyshev_local` | Averaged T_k(z) of each entry on its own; this is not a projection | Least squares in the chebyshev, legendre or monomial basis |

## Protocol

```
GET  /healthz                                     "ok"
GET|POST /health                                  {"ok", "status", "backend", "methods"}
POST /optimize {"matrix","method","params"}       optimized_matrix, rows, cols, diagnostics,
                                                  objective, matrix_opt, iterations, meta
POST /optimize {"matrix","sparsity"}              hrm_julia_backend form (chebyshev_projection)
POST /entropy  {"data"}                           {"shannon","spectral","compression_ratio"}
POST /chebyshev_project {"matrix","degree","basis"}  {"matrix","degree","basis","residual_rms"}
POST /         {"function", "args"}               ping | optimize | entropy | chebyshev_project |
                                                  optimize_matrix | chebyshev_projection | sparsity | rank | poly
```

The methods are:

- `chebyshev_projection`, with params `degree` (default 8), `sparsity` and `basis`.
- `sparsity`, with param `lambda`.
- `rank`, with param `tau`.
- `poly`, an alias for `chebyshev_projection`.

The objective is ½‖X − M‖²_F, plus λ‖X‖₁ for `sparsity` or τ‖X‖_* for `rank`. Every operator is closed form, so `iterations` is always 1.

## Evidence

**L0 tests** (`tests/integration/test_nurea_backend.py`):

- The Chebyshev and Legendre fits equal numpy's independent `chebfit` and `legfit`. A degree-3 polynomial row is reproduced exactly.
- Soft-thresholding and SVT match their closed forms.
- The three request forms (`/optimize`, the hrm form and `/` dispatch) agree.
- NuRea_sim's own `JuliaBackend` client gets answers from every primary endpoint, with no fallbacks:
  - `health_check`
  - `analyze_entropy`
  - `project_chebyshev`
  - `optimize`

**L0 telemetry** on the first 512 NuRea_sim lattice simulations (rows = simulations, 39 rod enrichments each):

| Call | Result |
|---|---|
| `chebyshev_projection` degree 8 | residual RMS 1.089 wt% (5.5 ms) |
| …with sparsity 0.5 (2304 of 4608 coefficients kept) | residual RMS 1.107 wt% |
| `rank` τ = 5 | rank 39 → 39; nuclear norm 1459.7 → 1264.7 (14 ms) |
| `entropy` | Shannon 5.997 bits of a 6-bit maximum; spectral 4.64 bits; compression ratio 512 (full rank) |

**What the telemetry says about the data:**

- The enrichments of the 39 rods behave as independent draws from U(0.7, 5.0) wt%. Uniform on 4.3 wt% has standard deviation 1.241.
- A degree-8 fit absorbs 9 of 39 degrees of freedom of white noise, which predicts a residual of √(1 − 9/39) · 1.241 = 1.089. That matches the measured value.
- Rod order in the file is therefore not a smooth spatial coordinate. Smooth-basis compression of the inputs is the wrong tool for this dataset. A pin-position map would be needed (see `docs/neutronics/README.md`, L1).
