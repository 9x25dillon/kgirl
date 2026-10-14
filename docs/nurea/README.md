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

## LIMPS protocol (LiMp, bigLIMp)

LiMp's `JuliaClient` and `LIMPSJuliaIntegration` (`9x25dillon/LiMp/julia_client.py`) and bigLIMp's revised LIMPS modules use the same `{"function", "args"}` dispatch. The backend now serves their functions too, so the same server handles NuRea, LIMPS and kgirl clients.

| Function | Upstream behaviour | Here |
|---|---|---|
| `optimize_matrix(matrix, "sparsity")` | Zeroed entries below 10 % of the maximum | Soft-threshold (L1 prox). λ defaults to 10 % of the maximum |
| `optimize_matrix(matrix, "rank")` | Always kept half the rank | Without τ: rank chosen by BIC, hard truncation. With τ: singular-value threshold |
| `optimize_matrix(matrix, "structure")` | Row means + column means − mean, with a constant `complexity_reduction = 0.3` | The same additive fit, but reporting a measured R², row and column effect variances, and interaction RMS |
| `optimize_matrix(matrix, "auto")` | Chose a method from fixed thresholds on a complexity score | Fits every candidate (structure, rank-r, Chebyshev degree-d, top-s sparsity) and keeps the lowest BIC = N·ln(RSS/N) + k·ln N |
| `matrix_to_polynomials` / `analyze_matrix_structure` | Symbolic `x[i,j]` strings and fixed-weight scores | Numerical rank, stable rank, condition number, sparsity, additive R², and complexity = normalised spectral entropy ∈ [0, 1] |
| `create_polynomials` / `analyze_polynomials` | Linear forms; `matrices.jl` did not load (`@polyvar` without `DynamicPolynomials`) | The same linear forms. Complexity = mean normalised entropy of \|coefficients\| |
| `analyze_text_structure` | `complexity_score = text_length / 100` (unbounded) | Shannon entropy in nats (`text_entropy`, LiMp's unit) and bits, normalised entropy, Zipf exponent |

Every reply carries `compression_ratio` at the top level, which `optimize_limps_matrix` reads; it is the share of numbers saved. `GET /` answers `test_connection`.

**L0 tests:**

- The additive fit is exact on additive matrices.
- `auto` selects `structure` for an additive matrix, and rank 2 for a rank-2 matrix.
- LiMp's own `JuliaClient` and `LIMPSJuliaIntegration` complete `process_entropy_matrix`, `analyze_natural_language` and `optimize_limps_matrix` against the stdlib server (`tests/integration/test_nurea_backend.py`).
- A static test checks that the `.jl` file serves the same function names. Julia itself still has not been executed here.

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
