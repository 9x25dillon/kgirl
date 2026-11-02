#!/usr/bin/env bash
# Build kgvecd for the R7800 (32-bit ARM, no libc needed) and for this machine.
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
src=$here/src/kgirl/harness/atlas/router/kgvecd.c
out=${1:-$here/build/router}
mkdir -p "$out"
clang --target=armv7a-linux-musleabihf -marm -mcpu=cortex-a15 -mfpu=neon-vfpv4 -mfloat-abi=hard \
  -O3 -ffreestanding -fno-stack-protector -fomit-frame-pointer -fno-pic -static -nostdlib -fuse-ld=lld \
  "$src" -o "$out/kgvecd.arm"
clang -O3 -march=native "$src" -o "$out/kgvecd.native"
ls -l "$out"/kgvecd.*
