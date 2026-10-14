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

using HTTP, JSON3, LinearAlgebra, Printf, Statistics

const METHODS = ("chebyshev_projection", "sparsity", "rank", "structure", "auto")
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

# ---------------------------------------------------------------- LIMPS functions (LiMp / bigLIMp protocol)

function additive_structure(M::Matrix{Float64})
    mu = mean(M)
    out = mean(M; dims = 2) .+ mean(M; dims = 1) .- mu
    sst = sum(abs2, M .- mu)
    sse = sum(abs2, M .- out)
    out, Dict("grand_mean" => mu, "r2" => sst > 0 ? 1 - sse / sst : 1.0,
              "row_effect_var" => var(vec(mean(M; dims = 2)); corrected = false),
              "col_effect_var" => var(vec(mean(M; dims = 1)); corrected = false),
              "interaction_rms" => sqrt(sse / length(M)), "parameters" => size(M, 1) + size(M, 2) - 1)
end

function bic(M::Matrix{Float64}, X::Matrix{Float64}, k::Integer)
    n = length(M)
    tss = sum(abs2, M .- mean(M))
    tss = tss > 0 ? tss : 1.0
    rss = max(sum(abs2, M .- X), 1e-12 * tss)
    n * log(rss / n) + k * log(n)
end

function auto_select(M::Matrix{Float64})
    rows, cols = size(M)
    cands = Dict{String, Tuple{Float64, Matrix{Float64}, Dict{String, Any}}}()
    Xs, ds = additive_structure(M)
    cands["structure"] = (bic(M, Xs, ds["parameters"]), Xs, Dict{String, Any}("r2" => ds["r2"]))
    F = svd(M)
    top = max(1, min(rows, cols) - 1)
    rs = sort(unique(round.(Int, exp.(range(0, log(top); length = min(24, top))))))
    best = nothing
    for r in rs
        X = F.U[:, 1:r] * Diagonal(F.S[1:r]) * F.Vt[1:r, :]
        b = bic(M, X, r * (rows + cols - r))
        (best === nothing || b < best[1]) && (best = (b, X, Dict{String, Any}("rank" => r)))
    end
    cands["rank"] = best
    best = nothing
    for d in 0:min(cols - 1, 12)
        X, _ = chebyshev_projection(M; degree = d)
        b = bic(M, X, rows * (d + 1))
        (best === nothing || b < best[1]) && (best = (b, X, Dict{String, Any}("degree" => d)))
    end
    cands["chebyshev_projection"] = best
    flat = vec(permutedims(M))                               # row-major, as numpy ravel
    order = sortperm(-abs.(flat); alg = MergeSort)
    best = nothing
    for sz in sort(unique([max(1, round(Int, length(M) * f)) for f in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7)]))
        x = zeros(length(M))
        x[order[1:sz]] .= flat[order[1:sz]]
        X = permutedims(reshape(x, cols, rows))
        b = bic(M, X, 2 * sz)
        (best === nothing || b < best[1]) && (best = (b, X, Dict{String, Any}("kept" => sz)))
    end
    cands["sparsity"] = best
    choice = first(sort(collect(keys(cands)); by = k -> cands[k][1]))
    cands[choice][2], Dict("selected" => choice, "selected_setting" => cands[choice][3],
                           "bic" => Dict(k => v[1] for (k, v) in cands), "settings" => Dict(k => v[3] for (k, v) in cands))
end

function matrix_structure(M::Matrix{Float64})
    rows, cols = size(M)
    s = svdvals(M)
    tol = maximum(size(M)) * eps(Float64) * (isempty(s) ? 0.0 : s[1])
    s2 = s .^ 2
    p = sum(s2) > 0 ? s2[s2 .> 0] ./ sum(s2) : Float64[]
    ent = length(s) > 1 && !isempty(p) ? -sum(p .* log.(p)) / log(length(s)) : 0.0
    mx = maximum(abs, M)
    Dict("shape" => [rows, cols], "matrix_shape" => [rows, cols], "polynomial_terms" => rows * cols,
         "representation" => "hybrid_polynomial", "rank" => count(>(tol), s),
         "stable_rank" => !isempty(s) && s[1] > 0 ? sum(s2) / s2[1] : 0.0,
         "condition_number" => !isempty(s) && s[end] > 0 ? s[1] / s[end] : 1e308,
         "sparsity" => mean(M .== 0.0), "near_zero_fraction" => mean(abs.(M) .< 0.01 * (mx > 0 ? mx : 1.0)),
         "additive_r2" => additive_structure(M)[2]["r2"], "complexity_score" => ent,
         "coeff_matrix" => [M[i, :] for i in 1:rows])
end

fmt(a) = @sprintf("%g", a)                                  # same text as Python's f"{a:g}"

function create_polynomials(data, variables)
    M = to_matrix(data)
    size(M, 2) == length(variables) || error("$(size(M, 2)) columns but $(length(variables)) variables")
    out = Dict{String, Any}()
    for i in 1:size(M, 1)
        nz = [j for j in 1:size(M, 2) if M[i, j] != 0.0]
        terms = ["$(fmt(M[i, j]))*$(variables[j])" for j in nz]
        str = isempty(terms) ? "0" : replace(join(terms, " + "), "+ -" => "- ")
        out["P$i"] = Dict("string" => str, "coeffs" => [M[i, j] for j in nz], "terms" => [String(variables[j]) for j in nz],
                          "degree" => isempty(nz) ? 0 : 1, "term_count" => length(nz))
    end
    out
end

function analyze_polynomials(polys)
    if polys isa AbstractDict && haskey(polys, :coeff_matrix)
        M = to_matrix(polys[:coeff_matrix])
        polys = create_polynomials([M[i, :] for i in 1:size(M, 1)], ["x$j" for j in 1:size(M, 2)])
    end
    vals = [v for v in values(polys) if v isa AbstractDict]
    isempty(vals) && return Dict("total_polynomials" => 0, "complexity_score" => 0.0)
    g(v, k, d) = haskey(v, k) ? v[k] : (haskey(v, Symbol(k)) ? v[Symbol(k)] : d)
    degrees = [Int(g(v, "degree", 0)) for v in vals]
    counts = [Int(g(v, "term_count", length(g(v, "terms", [])))) for v in vals]
    ents = Float64[]
    for v in vals
        c = abs.(Float64.(collect(g(v, "coeffs", Float64[]))))
        if length(c) > 1 && sum(c) > 0
            q = c[c .> 0] ./ sum(c)
            push!(ents, -sum(q .* log.(q)) / log(length(c)))
        else
            push!(ents, 0.0)
        end
    end
    Dict("total_polynomials" => length(vals), "average_degree" => mean(degrees), "max_degree" => maximum(degrees),
         "average_terms" => mean(counts), "max_terms" => maximum(counts), "complexity_score" => mean(ents))
end

function analyze_text_structure(text::AbstractString; top::Int = 50)
    words = String.(split(text))
    n = length(words)
    n == 0 && return Dict("text_length" => length(text), "word_count" => 0, "unique_words" => 0, "text_entropy" => 0.0,
                          "entropy_bits" => 0.0, "normalised_entropy" => 0.0, "complexity_score" => 0.0)
    freq = Dict{String, Int}()
    for w in words
        freq[w] = get(freq, w, 0) + 1
    end
    vocab = sort(collect(keys(freq)))                        # numpy.unique order, then a stable count sort
    counts = [freq[w] for w in vocab]
    p = counts ./ n
    h = -sum(p .* log.(p))
    order = sortperm(-counts; alg = MergeSort)
    f = Float64.(counts[order])
    rep = f[f .>= 2]
    zipf = nothing
    if length(rep) >= 3
        x = log.(1:length(rep))
        y = log.(rep)
        slope = sum((x .- mean(x)) .* (y .- mean(y))) / sum((x .- mean(x)) .^ 2)
        zipf = -slope
    end
    norm = length(vocab) > 1 ? h / log(length(vocab)) : 0.0
    Dict("text_length" => length(text), "word_count" => n, "unique_words" => length(vocab),
         "average_word_length" => mean(length.(words)), "text_entropy" => h, "entropy_bits" => h / log(2),
         "normalised_entropy" => norm, "vocabulary_richness" => length(vocab) / n, "zipf_exponent" => zipf,
         "complexity_score" => norm, "word_frequency" => Dict(vocab[i] => counts[i] for i in order[1:min(top, end)]))
end

getp(p, key, default) = (p !== nothing && haskey(p, key)) ? p[key] : default

function optimize(rows, method::AbstractString, params)
    M = to_matrix(rows)
    method == "poly" && (method = "chebyshev_projection")
    kept = nothing
    if method == "chebyshev_projection"
        out, diag = chebyshev_projection(M; degree = round(Int, Float64(getp(params, :degree, 8))),
                                         sparsity = Float64(getp(params, :sparsity, 0.0)),
                                         basis = String(getp(params, :basis, "chebyshev")))
        objective = 0.5 * sum(abs2, out .- M)
        kept = diag["coefficients_kept"]
    elseif method == "sparsity"
        lambda = Float64(getp(params, :lambda, 0.1 * maximum(abs, M)))     # default: LiMp's 10 % of max
        out, diag = soft_threshold(M; lambda = lambda)
        objective = 0.5 * sum(abs2, out .- M) + lambda * sum(abs, out)
        kept = count(!iszero, out)
    elseif method == "rank"
        if params !== nothing && haskey(params, :tau)
            tau = Float64(params[:tau])
            out, diag = singular_value_threshold(M; tau = tau)
            objective = 0.5 * sum(abs2, out .- M) + tau * diag["nuclear_after"]
            r = diag["rank_after"]
        else                                                  # no τ: rank chosen by BIC, hard truncation
            r = auto_select(M)[2]["settings"]["rank"]["rank"]
            F = svd(M)
            out = F.U[:, 1:r] * Diagonal(F.S[1:r]) * F.Vt[1:r, :]
            diag = Dict{String, Any}("rank_selected_by_bic" => r)
            objective = 0.5 * sum(abs2, out .- M)
        end
        kept = r * (size(M, 1) + size(M, 2) - r)
    elseif method == "structure"
        out, diag = additive_structure(M)
        objective = 0.5 * sum(abs2, out .- M)
        kept = diag["parameters"]
    elseif method == "auto"
        out, diag = auto_select(M)
        objective = 0.5 * sum(abs2, out .- M)
    else
        error("unknown method $method; one of $(join(METHODS, ", ")), poly")
    end
    diag = Dict{String, Any}(diag)
    kept === nothing || (diag["compression_ratio"] = max(0.0, 1.0 - kept / length(M)))
    rowsout = [out[i, :] for i in 1:size(out, 1)]
    Dict("optimized_matrix" => rowsout, "rows" => size(out, 1), "cols" => size(out, 2), "method" => method,
         "params" => params === nothing ? Dict() : params, "diagnostics" => diag,
         "compression_ratio" => get(diag, "compression_ratio", nothing),
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
        name in ("ping", "health_check") && return Dict("ok" => true)
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
        elseif name in ("matrix_to_polynomials", "analyze_matrix_structure")
            return matrix_structure(to_matrix(args[1]))
        elseif name == "create_polynomials"
            vars = length(args) > 1 ? String.(collect(args[2])) : ["x$j" for j in 1:length(args[1][1])]
            return create_polynomials(args[1], vars)
        elseif name == "analyze_polynomials"
            return analyze_polynomials(args[1])
        elseif name == "analyze_text_structure"
            return analyze_text_structure(String(args[1]))
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
        path == "/" && return jsonresp(health())                 # LiMp JuliaClient.test_connection
        path == "/healthz" && return HTTP.Response(200, ["Content-Type" => "text/plain"], "ok")
        path == "/health" && return jsonresp(health())
        return jsonresp(Dict("error" => "not found"); status = 404)
    elseif HTTP.method(req) == "POST"
        try
            body = isempty(req.body) ? JSON3.read("{}") : JSON3.read(String(req.body))
            return jsonresp(handle(path, body))
        catch e
            e isa KeyError && e.key isa AbstractString && return jsonresp(Dict("error" => "not found"); status = 404)
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
