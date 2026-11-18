#!/bin/sh
# Compile and run every CPU test in tests/. No GPU needed.
# Usage, from anywhere:  ./tests/run.sh
set -e
cd "$(dirname "$0")/.."
mkdir -p build

found=0
for f in tests/*.cpp; do
  [ -e "$f" ] || continue
  found=1
  name=$(basename "$f" .cpp)
  # -ffp-contract=off: do not fuse multiply-add, same reason as -fmad=false on the GPU.
  g++ -std=c++17 -O2 -Wall -Wextra -ffp-contract=off -Isrc "$f" -o "build/$name"
  echo "== $name"
  "./build/$name"
done

[ "$found" = 1 ] || echo "No tests yet: add a .cpp file to tests/"
