#!/usr/bin/env bash
set -euo pipefail
cd /workspace

install_portable_julia() {
  echo "Installing portable Julia..." >&2
  arch=$(uname -m)
  ver_full=${JULIA_VERSION:-"1.10.4"}
  ver_mm=$(echo "$ver_full" | awk -F. '{print $1 "." $2}')
  case "$arch" in
    x86_64)
      path="bin/linux/x64/${ver_mm}"; file="julia-${ver_full}-linux-x86_64.tar.gz" ;;
    aarch64|arm64)
      path="bin/linux/aarch64/${ver_mm}"; file="julia-${ver_full}-linux-aarch64.tar.gz" ;;
    *) echo "Unsupported arch: $arch" >&2; return 1;;
  esac
  urls=(
    "https://julialang-s3.julialang.org/${path}/${file}"
    "https://mirrors.ustc.edu.cn/julia-releases/${path}/${file}"
  )
  for url in "${urls[@]}"; do
    echo "Downloading $url" >&2 || true
    if curl -fL --retry 3 --retry-delay 2 "$url" -o julia.tar.gz; then
      top_dir=$(tar -tzf julia.tar.gz | head -1 | cut -d/ -f1)
      tar -xzf julia.tar.gz
      rm -f julia.tar.gz
      if [ -n "$top_dir" ] && [ -d "$top_dir" ]; then
        rm -rf julia-portable
        mv "$top_dir" julia-portable
      fi
      if [ -x /workspace/julia-portable/bin/julia ]; then
        echo "Installed portable Julia to /workspace/julia-portable" >&2
        return 0
      fi
    fi
  done
  echo "Failed to download Julia tarball" >&2
  return 1
}

ensure_julia() {
  if command -v julia >/dev/null 2>&1; then return 0; fi
  if [ -x /workspace/julia-portable/bin/julia ]; then
    export PATH="/workspace/julia-portable/bin:$PATH"; return 0
  fi
  if command -v apt-get >/dev/null 2>&1; then
    echo "Attempting Julia install via apt (non-fatal)..." >&2
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y || true
    apt-get install -y --no-install-recommends julia ca-certificates curl || true
    if command -v julia >/dev/null 2>&1; then return 0; fi
  fi
  install_portable_julia || true
  if [ -x /workspace/julia-portable/bin/julia ]; then
    export PATH="/workspace/julia-portable/bin:$PATH"; return 0
  fi
  echo "Julia is not installed and portable install failed." >&2
  return 1
}

ensure_julia

# Install/instantiate deps
julia --project=/workspace -e 'using Pkg; Pkg.activate("."); Pkg.add(["HTTP","JSON3","LibPQ","DSP","UUIDs","Dates","Statistics","Random","Interpolations"]); Pkg.precompile()'

if [[ "${START_SERVER:-0}" == "1" ]]; then
  exec julia --project=/workspace /workspace/server.jl
else
  echo "Dependencies installed. To start server: START_SERVER=1 bash /workspace/run.sh" >&2
fi