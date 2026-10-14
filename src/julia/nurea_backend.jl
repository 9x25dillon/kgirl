#!/usr/bin/env julia
# NuRea matrix backend: the service NuRea_sim's orchestrator calls, with real operators.
#
#   julia --project=@nurea -e 'using Pkg; Pkg.add(["HTTP", "JSON3"])'   # once
#   JULIA_PORT=9000 julia --project=@nurea src/julia/nurea_backend.jl
#
# Replaces NuRea_sim/backends/julia/server.jl and hrm_julia_backend/server.jl. In those servers,
# "chebyshev_projection" multiplied by 0.95, "rank" zeroed small entries without an SVD, a missing
# matrix became randn data, and /optimize replies lacked the fields matrix_orchestrator parses
# (objective, matrix_opt, iterations, meta). Formulas here are term for term those of
# src/kgirl/nurea_backend.py: the numpy reference, also servable without Julia via
# `python -m kgirl.nurea_backend serve`, and the one tested against NuRea's own JuliaBackend client.
#
# Protocols:
#   GET  /healthz                                    → "ok"
#   GET  /health | POST /health                      → {"ok", "status", "backend", "methods"}
#   POST /optimize {"matrix","method","params"}      → {"optimized_matrix","rows","cols","method","params",
#                                                       "diagnostics","objective","matrix_opt","iterations","meta"}
#   POST /optimize {"matrix","sparsity"}             → same, method chebyshev_projection   (hrm_julia_backend)
#   POST /entropy  {"data"}                          → {"shannon","spectral","compression_ratio"}
#   POST /chebyshev_project {"matrix","degree","basis"} → {"matrix","degree","basis","residual_rms"}
#   POST /         {"function": name, "args": [...]} (matrix_orchestrator JuliaBackend._dispatch):
#                  ping | optimize [req] | entropy [{"data"}] | chebyshev_project [matrix, degree, basis]
#                  | optimize_matrix [matrix, method, params] | <method> [matrix, params]
#
# Operators on an m×n matrix M (rows = signals on n equispaced points of [−1, 1]):
#   chebyshev_projection  least squares on φ₀…φ_d per row (basis chebyshev | legendre | monomial,
#                         d = min(degree, n−1), default 8); zero the `sparsity` fraction of smallest
#                         |coefficients| (row-major order, stable sort); reconstruct V·C.  "poly" = alias.
#   sparsity              sign(x)·max(|x| − λ, 0)                   prox of λ‖M‖₁
#   rank                  U·diag(max(σ − τ, 0))·Vᵀ                  prox of τ‖M‖_*
#   objective             ½‖X − M‖²_F (+ λ‖X‖₁ | + τ‖X‖_*);  iterations = 1 (closed form)

using HTTP, JSON3, LinearAlgebra, Statistics

const METHODS = ("chebyshev_projection", "sparsity", "rank")
const BASES = ("chebyshev", "legendre", "monomial")

function to_matrix(rows)::Matrix{Float64}
    rows === nothing && error("matrix is required")
    isempty(rows) && error("matrix must be non-empty")
    rows[1] isa AbstractVector || error("matrix must be a list of rows")   # same rule as the numpy reference
    m, n = length(rows), length(rows[1])
    n == 0 && error("matrix must be non-empty")
    A = Matrix{Float64}(undef, m, n)
    for i in 1:m
        length(rows[i]) == n || error("ragged matrix at row $i")
        for j in 1:n
            A[i, j] = Float64(rows[i][j])
        end
    end
    all(isfinite, A) || error("matrix contains non-finite values")
    A
end

function vandermonde(n::Int, d::Int, basis::AbstractString = "chebyshev")
    basis in BASES || error("unknown basis $basis; one of $(join(BASES, ", "))")
    x = n == 1 ? [-1.0] : collect(range(-1.0, 1.0; length = n))
    V = Matrix{Float64}(undef, n, d + 1)
    V[:, 1] .= 1.0
    d >= 1 && (V[:, 2] .= x)
    for k in 2:d                                           # column k+1 holds φ_k
        if basis == "chebyshev"
            V[:, k + 1] .= 2.0 .* x .* V[:, k] .- V[:, k - 1]
        elseif basis == "legendre"
            V[:, k + 1] .= ((2k - 1) .* x .* V[:, k] .- (k - 1) .* V[:, k - 1]) ./ k
        else
            V[:, k + 1] .= x .* V[:, k]
        end
    end
    V
end

function chebyshev_projection(M::Matrix{Float64}; degree::Int = 8, sparsity::Float64 = 0.0,
                              basis::AbstractString = "chebyshev")
    rows, n = size(M)
    d = max(0, min(degree, n - 1))
    V = vandermonde(n, d, basis)
    C = V \ permutedims(M)                                   # (d+1) × rows, least squares (QR)
    k = sparsity > 0 ? min(round(Int, sparsity * length(C)), length(C)) : 0
    if k > 0
        R = permutedims(C)                                   # rows × (d+1): vec(R) is C in row-major order
        flat = vec(copy(R))
        flat[sortperm(abs.(flat); alg = MergeSort)[1:k]] .= 0.0
        C = permutedims(reshape(flat, size(R)))
    end
    out = permutedims(V * C)
    resid = sqrt(mean((out .- M) .^ 2))
    out, Dict("degree" => d, "basis" => basis, "coefficients" => [C[:, i] for i in 1:rows],
              "coefficients_kept" => length(C) - k, "coefficients_total" => length(C), "residual_rms" => resid)
end

function soft_threshold(M::Matrix{Float64}; lambda::Float64 = 1.0)
    out = sign.(M) .* max.(abs.(M) .- lambda, 0.0)
    out, Dict("lambda" => lambda, "nonzero_fraction" => mean(out .!= 0.0),
              "l1_before" => sum(abs, M), "l1_after" => sum(abs, out))
end

numerical_rank(s, dims) = count(>(maximum(dims) * eps(Float64) * (isempty(s) ? 0.0 : s[1])), s)

function singular_value_threshold(M::Matrix{Float64}; tau::Float64 = 1.0)
    F = svd(M)
    s2 = max.(F.S .- tau, 0.0)
    out = F.U * Diagonal(s2) * F.Vt
    tol = maximum(size(M)) * eps(Float64) * (isempty(F.S) ? 0.0 : F.S[1])
    out, Dict("tau" => tau, "rank_before" => count(>(tol), F.S), "rank_after" => count(>(tol), s2),
              "nuclear_before" => sum(F.S), "nuclear_after" => sum(s2))
end

function entropy_report(data)
    (data === nothing || isempty(data)) && return Dict("shannon" => 0.0, "spectral" => 0.0, "compression_ratio" => 1.0)
    A = data[1] isa AbstractVector ? to_matrix_lenient(data) : reshape(Float64.(collect(data)), 1, :)
    lo, hi = extrema(A)                                      # 64 equal bins on [min, max], like np.histogram
    counts = zeros(Int, 64)
    for a in A
        b = hi > lo ? min(64, 1 + floor(Int, (a - lo) / (hi - lo) * 64)) : 1
        counts[b] += 1
    end
    p = counts[counts .> 0] ./ length(A)
    shannon = -sum(p .* log2.(p))
    s = svdvals(A)
    ps = sum(s) > 0 ? s[s .> 0] ./ sum(s) : Float64[]
    spectral = isempty(ps) ? 0.0 : -sum(ps .* log2.(ps))
    Dict("shannon" => shannon, "spectral" => spectral,
         "compression_ratio" => length(A) / max(numerical_rank(s, size(A)), 1))
end

# entropy_local replaced NaN/±Inf by 0 before measuring; keep that, unlike to_matrix
function to_matrix_lenient(rows)
    A = [Float64(rows[i][j]) for i in eachindex(rows), j in eachindex(rows[1])]
    map(x -> isfinite(x) ? x : 0.0, A)
end

getp(p, key, default) = (p !== nothing && haskey(p, key)) ? p[key] : default

function optimize(rows, method::AbstractString, params)
    M = to_matrix(rows)
    method == "poly" && (method = "chebyshev_projection")
    if method == "chebyshev_projection"
        out, diag = chebyshev_projection(M; degree = round(Int, Float64(getp(params, :degree, 8))),
                                         sparsity = Float64(getp(params, :sparsity, 0.0)),
                                         basis = String(getp(params, :basis, "chebyshev")))
        objective = 0.5 * sum(abs2, out .- M)
    elseif method == "sparsity"
        lambda = Float64(getp(params, :lambda, 1.0))
        out, diag = soft_threshold(M; lambda = lambda)
        objective = 0.5 * sum(abs2, out .- M) + lambda * sum(abs, out)
    elseif method == "rank"
        tau = Float64(getp(params, :tau, 1.0))
        out, diag = singular_value_threshold(M; tau = tau)
        objective = 0.5 * sum(abs2, out .- M) + tau * diag["nuclear_after"]
    else
        error("unknown method $method; one of $(join(METHODS, ", ")), poly")
    end
    rowsout = [out[i, :] for i in 1:size(out, 1)]
    Dict("optimized_matrix" => rowsout, "rows" => size(out, 1), "cols" => size(out, 2), "method" => method,
         "params" => params === nothing ? Dict() : params, "diagnostics" => diag,
         "objective" => objective, "matrix_opt" => rowsout, "iterations" => 1,
         "meta" => merge(Dict{String,Any}("method" => method), diag))
end

function chebyshev_project(rows, degree, basis)
    M = to_matrix(rows)
    out, diag = chebyshev_projection(M; degree = round(Int, Float64(degree)), basis = String(basis))
    Dict("matrix" => [out[i, :] for i in 1:size(out, 1)], "degree" => diag["degree"], "basis" => basis,
         "residual_rms" => diag["residual_rms"])
end

jsonresp(obj; status = 200) = HTTP.Response(status, ["Content-Type" => "application/json"], JSON3.write(obj))
health() = Dict("ok" => true, "status" => "healthy", "backend" => "julia", "methods" => collect(METHODS),
                "julia" => string(VERSION))

function handle(path::AbstractString, body)
    path == "/health" && return health()
    if path == "/optimize"
        if haskey(body, :method) || haskey(body, :params)
            return optimize(get(body, :matrix, nothing), String(get(body, :method, "chebyshev_projection")),
                            get(body, :params, nothing))
        end
        return optimize(get(body, :matrix, nothing), "chebyshev_projection",
                        Dict(:sparsity => get(body, :sparsity, 0.0)))
    elseif path == "/entropy"
        return entropy_report(get(body, :data, nothing))
    elseif path == "/chebyshev_project"
        return chebyshev_project(get(body, :matrix, nothing), get(body, :degree, 3), get(body, :basis, "chebyshev"))
    elseif path == "/"
        name = String(get(body, :function, ""))      # a missing name must be a 400, not the KeyError → 404 path
        name == "ping" && return Dict("ok" => true)
        args = collect(get(body, :args, []))
        isempty(args) && error("args are required")
        if name == "optimize"
            req = args[1]
            return optimize(get(req, :matrix, nothing), String(get(req, :method, "chebyshev_projection")),
                            get(req, :params, nothing))
        elseif name == "entropy"
            return entropy_report(args[1] isa AbstractDict ? get(args[1], :data, nothing) : args[1])
        elseif name == "chebyshev_project"
            return chebyshev_project(args[1], length(args) > 1 ? args[2] : 3,
                                     length(args) > 2 ? args[3] : "chebyshev")
        elseif name == "optimize_matrix"
            return optimize(args[1], length(args) > 1 ? String(args[2]) : "chebyshev_projection",
                            length(args) > 2 ? args[3] : nothing)
        elseif name in METHODS || name == "poly"
            return optimize(args[1], name, length(args) > 1 ? args[2] : nothing)
        end
        error("unknown function $name")
    end
    throw(KeyError(path))
end

function route(req::HTTP.Request)
    path = HTTP.URI(req.target).path
    if HTTP.method(req) == "GET"
        path == "/healthz" && return HTTP.Response(200, ["Content-Type" => "text/plain"], "ok")
        path == "/health" && return jsonresp(health())
        return jsonresp(Dict("error" => "not found"); status = 404)
    elseif HTTP.method(req) == "POST"
        try
            body = isempty(req.body) ? JSON3.read("{}") : JSON3.read(String(req.body))
            return jsonresp(handle(path, body))
        catch e
            e isa KeyError && return jsonresp(Dict("error" => "not found"); status = 404)
            return jsonresp(Dict("error" => sprint(showerror, e)); status = 400)
        end
    end
    jsonresp(Dict("error" => "method not allowed"); status = 405)
end

if abspath(PROGRAM_FILE) == @__FILE__
    host = get(ENV, "JULIA_HOST", "127.0.0.1")
    port = parse(Int, get(ENV, "JULIA_PORT", "9000"))
    @info "kgirl NuRea Julia backend" host port methods = METHODS
    HTTP.serve(route, host, port)
end
