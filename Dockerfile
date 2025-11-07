FROM julia:1.10-bullseye

ENV JULIA_PROJECT=/app
WORKDIR /app

# System deps for LibPQ and SSL
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    libpq5 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# Copy code
COPY server.jl /app/server.jl

# Pre-install Julia deps to cache layers
RUN julia -e 'using Pkg; Pkg.activate("."); Pkg.add(["HTTP","JSON3","LibPQ","DSP","UUIDs","Dates","Statistics","Random","Interpolations"]); Pkg.precompile()'

EXPOSE 8081
CMD ["julia", "/app/server.jl"]