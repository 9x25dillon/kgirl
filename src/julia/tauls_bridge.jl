#!/usr/bin/env julia
# TA-ULS Julia bridge: weight-matrix stability, KFP fluctuation and entropy profiles over HTTP.
#
#   julia --project=@tauls -e 'using Pkg; Pkg.add(["HTTP", "JSON3"])'   # once
#   julia --project=@tauls src/julia/tauls_bridge.jl [port]              # default 9000
#
# Protocol (the one julia_server_script.jl and Fractal_cascade_simulation's JuliaBackend use):
#   POST /        {"function": name, "args": [matrix as a list of rows, ...]}  → JSON object
#   POST /health  {}                                                          → {"ok": true, ...}
#
# Formulas are term-for-term those of src/kgirl/tauls/julia_bridge.py (the numpy reference), with
# population variances (corrected = false), so the two backends report the same numbers. Analysis only:
# no function modifies a matrix, and there is no random noise anywhere.

using HTTP, JSON3, LinearAlgebra, Statistics

pvar(x) = var(x; corrected = false)                                    # population variance (numpy default)
pvar(A::AbstractMatrix; dims) = var(A; dims = dims, corrected = false)

function to_matrix(rows)::Matrix{Float64}
    isempty(rows) && error("empty matrix")
    m, n = length(rows), length(rows[1])
    A = Matrix{Float64}(undef, m, n)
    for i in 1:m
        length(rows[i]) == n || error("ragged matrix at row $i")
        for j in 1:n
            A[i, j] = Float64(rows[i][j])
        end
    end
    A
end

function stability_analysis(A::Matrix{Float64})
    s = svdvals(A)                                   # descending
    s2 = s .^ 2
    k = length(s)
    cond = s[end] > 0 ? s[1] / s[end] : Inf
    control = isfinite(cond) ? 1 / (1 + cond / 100) : 0.0
    learning = 1 / (1 + pvar(s2) / max(mean(s2), 1e-30))
    overall = (control + learning) / 2
    p = s2 ./ max(sum(s2), 1e-30)
    ent = k > 1 ? -sum(x * log(x) for x in p if x > 0) / log(k) : 0.0
    Dict("sigma_max" => s[1], "sigma_min" => s[end], "condition" => cond,
         "stable_rank" => sum(s2) / max(s2[1], 1e-30), "frobenius" => sqrt(sum(s2)),
         "control_stability" => control, "learning_stability" => learning, "overall_stability" => overall,
         "stability_class" => overall > 0.7 ? "stable" : overall > 0.4 ? "marginal" : "unstable",
         "spectral_entropy" => ent)
end

function kfp_profile(A::Matrix{Float64})
    rv = vec(pvar(A; dims = 2))
    cv = vec(pvar(A; dims = 1))
    total = mean(rv) + mean(cv)
    hot = mean((rv[i] + cv[j]) / 2 > total for i in eachindex(rv), j in eachindex(cv))
    Dict("kfp_fluctuation" => total, "kfp_hot_fraction" => hot, "row_var_max" => maximum(rv),
         "col_var_max" => maximum(cv))
end

function entropy_profile(A::Matrix{Float64})
    a = vec(abs.(A))
    p = a ./ max(sum(a), 1e-30)
    Dict("weight_entropy" => length(p) > 1 ? -sum(x * log(x) for x in p if x > 0) / log(length(p)) : 0.0)
end

const FUNCTIONS = Dict("stability_analysis" => stability_analysis, "kfp_profile" => kfp_profile,
                       "entropy_profile" => entropy_profile)

jsonresp(obj; status = 200) = HTTP.Response(status, ["Content-Type" => "application/json"], JSON3.write(obj))
finite(d) = Dict(k => (v isa Float64 && !isfinite(v) ? (v > 0 ? 1e308 : -1e308) : v) for (k, v) in d)

function route(req::HTTP.Request)
    HTTP.method(req) == "POST" || return jsonresp(Dict("error" => "POST only"); status = 405)
    if req.target == "/health"
        return jsonresp(Dict("ok" => true, "service" => "tauls_bridge", "functions" => collect(keys(FUNCTIONS)),
                             "julia" => string(VERSION)))
    elseif req.target == "/"
        try
            body = JSON3.read(String(req.body))
            name = String(body["function"])
            haskey(FUNCTIONS, name) || return jsonresp(Dict("error" => "unknown function $name"); status = 400)
            args = body["args"]
            length(args) == 1 || return jsonresp(Dict("error" => "$name takes one matrix"); status = 400)
            return jsonresp(finite(FUNCTIONS[name](to_matrix(args[1]))))
        catch e
            return jsonresp(Dict("error" => sprint(showerror, e)); status = 400)
        end
    end
    jsonresp(Dict("error" => "not found"); status = 404)
end

if abspath(PROGRAM_FILE) == @__FILE__
    port = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 9000
    @info "TA-ULS Julia bridge" port
    HTTP.serve(route, "127.0.0.1", port)
end
