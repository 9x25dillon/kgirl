# ChaosRAGJulia - Single-file server
using HTTP, JSON3, LibPQ, DSP, UUIDs, Dates, Statistics, Random

const DBURL = get(ENV, "DATABASE_URL", "postgres://user:pass@localhost:5432/chaos")
const OPENAI_MODEL_EMB = "text-embedding-3-large"
const OPENAI_MODEL_CHAT = "gpt-4o-mini"
const POOL = LibPQ.Connection(DBURL)

function json(req)::JSON3.Object
    body = String(take!(req.body)); JSON3.read(body)
end
resp(obj; status::Int=200) = HTTP.Response(status, ["Content-Type"=>"application/json"], JSON3.write(obj))

function execsql(sql::AbstractString)
    for stmt in split(sql, ';')
        s = strip(stmt)
        isempty(s) && continue
        try
            execute(POOL, s)
        catch e
            @warn "SQL exec warning" stmt=s exception=(e, catch_backtrace())
        end
    end
end
const SCHEMA = raw"""CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS hd_nodes (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  label TEXT NOT NULL,
  payload JSONB NOT NULL,
  coords DOUBLE PRECISION[] NOT NULL,
  unitary_tag TEXT,
  embedding VECTOR(1536),
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS hd_edges (
  src UUID REFERENCES hd_nodes(id) ON DELETE CASCADE,
  dst UUID REFERENCES hd_nodes(id) ON DELETE CASCADE,
  weight DOUBLE PRECISION DEFAULT 1.0,
  nesting_level INT DEFAULT 0,
  attrs JSONB DEFAULT '{}'::jsonb,
  PRIMARY KEY (src, dst)
);

CREATE TABLE IF NOT EXISTS hd_docs (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  source TEXT,
  kind TEXT,
  content TEXT,
  meta JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_hd_nodes_embedding ON hd_nodes USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);

CREATE TABLE IF NOT EXISTS tf_hht (
  asset TEXT NOT NULL,
  ts_start TIMESTAMPTZ NOT NULL,
  ts_end   TIMESTAMPTZ NOT NULL,
  imf_idx  INT NOT NULL DEFAULT 1,
  inst_freq DOUBLE PRECISION[] NOT NULL,
  inst_amp  DOUBLE PRECISION[] NOT NULL,
  burst BOOLEAN NOT NULL,
  features JSONB NOT NULL,
  PRIMARY KEY (asset, ts_start, imf_idx)
);

CREATE INDEX IF NOT EXISTS idx_tf_hht_asset_time ON tf_hht (asset, ts_start, ts_end);

CREATE TABLE IF NOT EXISTS state_telemetry (
  ts TIMESTAMPTZ PRIMARY KEY DEFAULT now(),
  asset TEXT NOT NULL,
  realized_vol DOUBLE PRECISION NOT NULL,
  entropy DOUBLE PRECISION NOT NULL,
  mod_intensity_grad DOUBLE PRECISION DEFAULT 0.0,
  router_noise DOUBLE PRECISION DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_state_tel_asset_ts ON state_telemetry (asset, ts);""" 
execsql(SCHEMA)

module OpenAIClient
using HTTP, JSON3, Random, Statistics
function fake_embed(text::AbstractString, dim::Int=1536)
    seed = UInt32(hash(text) % 0xffffffff); rng = Random.MersenneTwister(seed)
    v = rand(rng, Float32, dim); v ./= sqrt(sum(v.^2) + 1e-6f0); return v
end
function embed(text::AbstractString; model::AbstractString="text-embedding-3-large", dim::Int=1536)
    key = get(ENV, "OPENAI_API_KEY", nothing); isnothing(key) && return fake_embed(text, dim)
    try
        resp = HTTP.post("https://api.openai.com/v1/embeddings";
            headers = ["Authorization"=>"Bearer $key","Content-Type"=>"application/json"],
            body = JSON3.write((; input=text, model=model)))
        if resp.status != 200; return fake_embed(text, dim); end
        data = JSON3.read(String(resp.body)); return Float32.(data["data"][1]["embedding"])
    catch; return fake_embed(text, dim); end
end
function chat(system::AbstractString, prompt::AbstractString; model::AbstractString="gpt-4o-mini")
    key = get(ENV, "OPENAI_API_KEY", nothing); isnothing(key) && return "(stub) " * prompt[1:min(end, 400)]
    body = JSON3.write(Dict("model"=>model,"messages"=>[Dict("role"=>"system","content"=>system),Dict("role"=>"user","content"=>prompt)],"temperature"=>0.2))
    try
        resp = HTTP.post("https://api.openai.com/v1/chat/completions";headers=["Authorization"=>"Bearer $key","Content-Type"=>"application/json"],body=body)
        if resp.status != 200; return "(stub) " * prompt[1:min(end, 400)]; end
        data = JSON3.read(String(resp.body)); return String(data["choices"][1]["message"]["content"])
    catch; return "(stub) " * prompt[1:min(end, 400)]; end
end
end

module EEMD
using Interpolations, Statistics
function extrema_idx(x::AbstractVector{<:Real})
    n = length(x); max_idx = Int[]; min_idx = Int[]
    @inbounds for i in 2:n-1
        if x[i] > x[i-1] && x[i] > x[i+1]; push!(max_idx,i)
        elseif x[i] < x[i-1] && x[i] < x[i+1]; push!(min_idx,i) end
    end; return min_idx, max_idx
end
function envelope(x::Vector{Float64}, idx::Vector{Int})
    n = length(x); if length(idx) < 2; return collect(range(x[1], x[end], length=n)); end
    xi = Float64.(idx); yi = x[idx]
    itp = Interpolations.CubicSplineInterpolation(xi, yi, extrapolation_bc=Interpolations.Line())
    [itp(t) for t in 1:n]
end
function sift_one(x::Vector{Float64}; max_sift::Int=100, stop_tol::Float64=0.05)
    h = copy(x)
    for _ in 1:max_sift
        mins, maxs = extrema_idx(h); if length(mins)+length(maxs) < 2; break; end
        env_low = envelope(h, mins); env_high = envelope(h, maxs); m = @. (env_low + env_high)/2
        prev = copy(h); @. h = h - m
        if mean(abs.(m)) / (mean(abs.(prev)) + 1e-9) < stop_tol; break; end
        zc = sum(h[1:end-1] .* h[2:end] .< 0); mins2, maxs2 = extrema_idx(h)
        if abs((length(mins2)+length(maxs2)) - zc) ≤ 1; break; end
    end; return h
end
function emd(x::Vector{Float64}; max_imfs::Int=5)
    r = copy(x); imfs = Vector{Vector{Float64}}()
    for _ in 1:max_imfs
        imf = sift_one(r); push!(imfs, imf); r .-= imf
        mins,maxs = extrema_idx(r); if length(mins)+length(maxs) < 2; break; end
    end; return imfs, r
end
function eemd(x::Vector{Float64}; ensemble::Int=30, noise_std::Float64=0.2, max_imfs::Int=5)
    n = length(x); imf_accum = [zeros(n) for _ in 1:max_imfs]; counts = zeros(Int, max_imfs)
    for _ in 1:ensemble
        noise = noise_std * std(x) * randn(n); imfs, _ = emd(x .+ noise; max_imfs=max_imfs)
        for (k, imf) in enumerate(imfs); imf_accum[k] .+= imf; counts[k] += 1; end
    end
    imf_avg = Vector{Vector{Float64}}(); for k in 1:max_imfs; if counts[k] > 0; push!(imf_avg, imf_accum[k]/counts[k]); end; end
    return imf_avg
end
end

function hilbert_features(x::Vector{Float64}; Fs::Float64=1.0)
    z = DSP.hilbert(x); amp = abs.(z); phase = angle.(z)
    for i in 2:length(phase)
        Δ = phase[i] - phase[i-1]
        if Δ >  π; phase[i:end] .-= 2π; end
        if Δ < -π; phase[i:end] .+= 2π; end
    end
    inst_f = vcat(0.0, diff(phase)) .* Fs ./ (2π); return inst_f, amp
end

struct MixOut; stress::Float64; w_vec::Float64; w_graph::Float64; w_hht::Float64; top_k::Int; end
function route_mix(vol::Float64, ent::Float64, grad::Float64, base_k::Int=12)::MixOut
    stress = 1 / (1 + exp(-(1.8*vol + 1.5*ent + 0.8*abs(grad))))
    w_vec   = clamp(0.5 + 0.4*(1 - stress), 0.2, 0.9)
    w_graph = clamp(0.2 + 0.6*stress,       0.05,0.8)
    w_hht   = clamp(0.1 + 0.5*stress,       0.05,0.7)
    top_k   = max(4, Int(round(base_k * (0.7 + 0.8*(1 - stress)))))
    return MixOut(stress, w_vec, w_graph, w_hht, top_k)
end

router = HTTP.Router()

HTTP.@register router "POST" "/chaos/rag/index" function(req)
    d = json(req); docs = get(d, :docs, JSON3.Array()); count = 0
    for doc in docs
        src = get(doc,:source, nothing); kind = get(doc,:kind, nothing)
        content = String(get(doc,:content, "")); meta = JSON3.write(get(doc,:meta, JSON3.Object()))
        r = execute(POOL, "INSERT INTO hd_docs (source,kind,content,meta) VALUES ($1,$2,$3,$4) RETURNING id", (src,kind,content,meta)); doc_id = first(r)[1]
        emb = OpenAIClient.embed(content; model=OPENAI_MODEL_EMB)
        coords = [0.0,0.0,0.0]; payload = JSON3.write(JSON3.Object("doc_id"=>doc_id, "snippet"=>first(split(content, '\n'))))
        execute(POOL, "INSERT INTO hd_nodes (id,label,payload,coords,unitary_tag,embedding) VALUES ($1,$2,$3,$4,$5,$6) ON CONFLICT (id) DO NOTHING",
                (doc_id, "doc", payload, coords, "identity", emb))
        count += 1
    end
    resp(JSON3.Object("inserted"=>count))
end

HTTP.@register router "POST" "/chaos/telemetry" function(req)
    b = json(req)
    execute(POOL, "INSERT INTO state_telemetry (asset, realized_vol, entropy, mod_intensity_grad, router_noise) VALUES ($1,$2,$3,$4,$5)",
            (String(b[:asset]), Float64(b[:realized_vol]), Float64(b[:entropy]), Float64(get(b,:mod_intensity_grad,0.0)), Float64(get(b,:router_noise,0.0))))
    resp(JSON3.Object("ok"=>true))
end

HTTP.@register router "POST" "/chaos/hht/ingest" function(req)
    b = json(req)
    asset = String(b[:asset]); xs = Vector{Float64}(b[:x]); ts = Vector{String}(b[:ts]); Fs = Float64(get(b,:fs,1.0))
    max_imfs = Int(get(b,:max_imfs, 4))
    imfs = EEMD.eemd(xs; ensemble=Int(get(b,:ensemble,30)), noise_std=Float64(get(b,:noise_std,0.2)), max_imfs=max_imfs)
    for (k, imf) in enumerate(imfs)
        inst_f, inst_a = hilbert_features(imf; Fs=Fs)
        thrp = Float64(get(b,:amp_threshold_pct,0.8)); sorted = sort(inst_a)
        idx = Int(clamp(round((length(sorted)-1)*thrp)+1,1,length(sorted))); thr = sorted[idx]
        burst = any(>=(thr), inst_a)
        feats = JSON3.write(JSON3.Object("Fs"=>Fs,"thr"=>thr,"imf_power"=>sum(inst_a.^2)/length(inst_a),"len"=>length(imf)))
        sql = """
          INSERT INTO tf_hht (asset, ts_start, ts_end, imf_idx, inst_freq, inst_amp, burst, features)
          VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
          ON CONFLICT (asset, ts_start, imf_idx) DO UPDATE
            SET inst_freq=EXCLUDED.inst_freq, inst_amp=EXCLUDED.inst_amp, burst=EXCLUDED.burst, features=EXCLUDED.features
        """
        execute(POOL, sql, (asset, ts[1], ts[end], k, inst_f, inst_a, burst, feats))
    end
    resp(JSON3.Object("ok"=>true, "imfs"=>length(imfs)))
end

HTTP.@register router "POST" "/chaos/graph/entangle" function(req)
    b = json(req); pairs = Vector{Tuple{String,String}}()
    for p in get(b,:pairs, JSON3.Array()); push!(pairs, (String(p[1]), String(p[2]))); end
    level = Int(get(b,:nesting_level,0)); w = Float64(get(b,:weight,1.0)); attrs = JSON3.write(get(b,:attrs, JSON3.Object()))
    tx = LibPQ.Transaction(POOL)
    try
        for (src,dst) in pairs
            execute(POOL, """
              INSERT INTO hd_edges (src,dst,weight,nesting_level,attrs)
              VALUES ($1,$2,$3,$4,$5)
              ON CONFLICT (src,dst) DO UPDATE
                SET weight=EXCLUDED.weight, nesting_level=EXCLUDED.nesting_level, attrs=EXCLUDED.attrs
            """, (src,dst,w,level,attrs))
        end
        commit(tx); resp(JSON3.Object("ok"=>true))
    catch e
        rollback(tx); resp(JSON3.Object("error"=>string(e)); status=500)
    end
end

HTTP.@register router "GET" r"^/chaos/graph/([0-9a-fA-F-]+)$" function(req, caps)
    id = caps.captures[1]
    row = first(execute(POOL, "SELECT id,label,payload,coords,unitary_tag,created_at FROM hd_nodes WHERE id=$1", (id,)), nothing)
    isnothing(row) && return resp(JSON3.Object("error"=>"not found"); status=404)
    edges = execute(POOL, "SELECT src,dst,weight,nesting_level,attrs FROM hd_edges WHERE src=$1 OR dst=$1", (id,))
    ej = JSON3.Array(); for e in edges; push!(ej, JSON3.Object("src"=>e[1], "dst"=>e[2], "weight"=>e[3], "nesting_level"=>e[4], "attrs"=>JSON3.read(String(e[5])))); end
    nj = JSON3.Object("id"=>row[1], "label"=>row[2], "payload"=>JSON3.read(String(row[3])), "coords"=>row[4], "unitary_tag"=>row[5], "created_at"=>string(row[6]))
    resp(JSON3.Object("node"=>nj, "edges"=>ej))
end

HTTP.@register router "POST" "/chaos/rag/query" function(req)
    b = json(req); q = String(get(b,:q,"")); k = Int(get(b,:k, 12))
    emb = OpenAIClient.embed(q; model=OPENAI_MODEL_EMB)
    asset = occursin(r"ETH"i, q) ? "ETH" : "BTC"
    tel = first(execute(POOL, """
        SELECT realized_vol, entropy, mod_intensity_grad FROM state_telemetry
        WHERE asset=$1 AND ts > now()- interval '30 minutes'
        ORDER BY ts DESC LIMIT 1
    """, (asset,)), nothing)
    vol = isnothing(tel) ? 0.1 : Float64(tel[1]); ent = isnothing(tel) ? 0.1 : Float64(tel[2]); grad = isnothing(tel) ? 0.0 : Float64(tel[3])
    mix = route_mix(vol, ent, grad, k)
    kv = max(2, ceil(Int, mix.top_k * mix.w_vec)); kg = max(1, ceil(Int, mix.top_k * mix.w_graph)); kh = max(1, ceil(Int, mix.top_k * mix.w_hht))
    rows = execute(POOL, "SELECT id,payload,(embedding <-> $1::vector) AS score FROM hd_nodes ORDER BY embedding <-> $1::vector LIMIT $2", (emb, kv))
    first_id = isempty(rows) ? nothing : rows[1][1]
    grows = isnothing(first_id) ? [] : execute(POOL, "SELECT n.id,n.payload FROM hd_edges e JOIN hd_nodes n ON n.id=e.dst WHERE e.src=$1 ORDER BY e.weight DESC LIMIT $2", (first_id, kg))
    hrows = execute(POOL, "SELECT asset, ts_start, ts_end, features FROM tf_hht WHERE asset=$1 AND ts_end > now()- interval '1 hour' AND burst = TRUE ORDER BY ts_end DESC LIMIT $2", (asset, kh))
    ctx_parts = String[]
    for r in rows; push!(ctx_parts, String(r[2])); end
    for g in grows; push!(ctx_parts, String(g[2])); end
    for h in hrows; push!(ctx_parts, "HHT " * JSON3.write(JSON3.Object("asset"=>h[1], "ts_start"=>h[2], "ts_end"=>h[3], "features"=>JSON3.read(String(h[4]))))); end
    context = join(ctx_parts[1:min(end, 8)], "\n---\n")
    sys = "You are a crypto analytics assistant. Use context faithfully. Be concise and regime-aware."
    answer = OpenAIClient.chat(sys, "Query: " * q * "\n\nContext:\n" * context; model=OPENAI_MODEL_CHAT)
    out = JSON3.Object("router"=>JSON3.Object("stress"=>mix.stress,"mix"=>JSON3.Object("vector"=>mix.w_vec,"graph"=>mix.w_graph,"hht"=>mix.w_hht),"top_k"=>mix.top_k),
                       "answer"=>answer, "hits"=>ctx_parts[1:min(end,8)])
    resp(out)
end

println("Chaos RAG Julia (single-file) on 0.0.0.0:8081"); HTTP.serve(router, ip"0.0.0.0", 8081)
