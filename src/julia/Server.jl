module ChaosServer

using HTTP, JSON3, Logging, Dates, Symbolics, WebSockets

const ALLOWED_FUNCS = Set(["SUM","MEAN","VAR","DIFF","SIMPLIFY"])  # extend as needed

struct AppState
    started_at::DateTime
    http_count::Int
    ws_count::Int
end
const STATE = Ref{AppState}()

_json(x) = JSON3.write(x)

function _parse_symbolic_call(s::AbstractString)
    m = match(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)$", strip(s))
    if m === nothing
        return Dict("name"=>nothing, "args"=>String[])
    end
    name = uppercase(String(m.captures[1]))
    args_str = String(m.captures[2])
    args = isempty(strip(args_str)) ? String[] : [strip(x) for x in split(args_str, ",")]
    return Dict("name"=>name, "args"=>args)
end

function _eval_symbolic(name::String, args::Vector{String})
    if !(name in ALLOWED_FUNCS)
        return Dict("ok"=>false, "error"=>"function not allowed", "name"=>name)
    end
    try
        if name == "SUM"
            vals = parse.(Float64, args)
            return Dict("ok"=>true, "result"=>sum(vals))
        elseif name == "MEAN"
            vals = parse.(Float64, args)
            return Dict("ok"=>true, "result"=>sum(vals)/max(length(vals),1))
        elseif name == "VAR"
            vals = parse.(Float64, args)
            μ = sum(vals)/max(length(vals),1)
            v = sum((x-μ)^2 for x in vals)/max(length(vals),1)
            return Dict("ok"=>true, "result"=>v)
        elseif name == "DIFF"
            f = Symbolics.parse_expr(args[1])
            sym = Symbolics.parse_expr(args[2])
            return Dict("ok"=>true, "result"=>string(Symbolics.derivative(f, sym)))
        elseif name == "SIMPLIFY"
            expr = Symbolics.parse_expr(args[1])
            return Dict("ok"=>true, "result"=>string(Symbolics.simplify(expr)))
        end
    catch e
        return Dict("ok"=>false, "error"=>string(e), "name"=>name)
    end
end

# HTTP routes
function route(req::HTTP.Request)
    try
        if req.target == "/health"
            return HTTP.Response(200, _json(Dict(
                "ok"=>true,
                "service"=>"Chaos Julia Server",
                "started_at"=>string(STATE[].started_at),
                "http_count"=>STATE[].http_count,
                "ws_count"=>STATE[].ws_count,
            )))
        elseif req.target == "/v1/symbolic/parse" && HTTP.method(req) == "POST"
            data = JSON3.read(String(req.body))
            parsed = _parse_symbolic_call(get(data, "text", ""))
            STATE[].http_count += 1
            return HTTP.Response(200, _json(Dict("ok"=>true, "parsed"=>parsed)))
        elseif req.target == "/v1/symbolic/eval" && HTTP.method(req) == "POST"
            data = JSON3.read(String(req.body))
            name = uppercase(String(get(data, "name", "")))
            args = Vector{String}(get(data, "args", String[]))
            result = _eval_symbolic(name, args)
            STATE[].http_count += 1
            return HTTP.Response(200, _json(result))
        else
            return HTTP.Response(404, _json(Dict("ok"=>false, "error"=>"not found")))
        end
    catch e
        @warn "Route error" error=e
        return HTTP.Response(500, _json(Dict("ok"=>false, "error"=>string(e))))
    end
end

# WebSocket handler
function ws_handler(ws)
    try
        while !eof(ws)
            data = String(readavailable(ws))
            msg = JSON3.read(data)
            if get(msg, "type", "") == "parse"
                parsed = _parse_symbolic_call(get(msg, "text", ""))
                write(ws, _json(Dict("type"=>"parse_result", "parsed"=>parsed)))
            elseif get(msg, "type", "") == "eval"
                name = uppercase(String(get(msg, "name", "")))
                args = Vector{String}(get(msg, "args", String[]))
                result = _eval_symbolic(name, args)
                write(ws, _json(Dict("type"=>"eval_result", "result"=>result)))
            elseif get(msg, "type", "") == "batch_eval"
                calls = get(msg, "calls", [])
                results = [_eval_symbolic(c["name"], c["args"]) for c in calls]
                write(ws, _json(Dict("type"=>"batch_eval_result", "results"=>results)))
            else
                write(ws, _json(Dict("type"=>"error", "error"=>"unknown message type")))
            end
            STATE[].ws_count += 1
        end
    catch e
        @warn "WebSocket error" error=e
    end
end

function start(; host="0.0.0.0", http_port::Integer=8088, ws_port::Integer=8089)
    STATE[] = AppState(now(), 0, 0)
    @info "Starting Chaos Julia Server" host http_port ws_port
    @async HTTP.serve(route, host, http_port; verbose=false)
    @async WebSockets.listen(host, ws_port, ws_handler)
    @info "Servers started. Ctrl+C to stop."
    try
        while true
            sleep(1)
        end
    catch
        @info "Shutting down"
    end
end

end # module
