#!/bin/sh
# RTL-vs-Python-model check. Requires python3 + numpy, iverilog + vvp; yosys is optional.
# Regenerates the golden vectors from approx.py, runs the self-checking testbench and exits
# non-zero unless it prints "TB PASS".
set -e
cd "$(dirname "$0")"
python3 gen_vectors.py
iverilog -g2005 -o sim.vvp approx_mul.v mac_array.v tb_mac_array.v
vvp -n sim.vvp | tee sim.log
grep -q "^TB PASS" sim.log || { echo "testbench did not pass" >&2; exit 1; }
# optional generic (NOT PolarFire) synthesis cell counts, 8 lanes:
if command -v yosys >/dev/null 2>&1; then
  for m in "0 0" "1 6" "2 4"; do set -- $m
    yosys -p "read_verilog mac_array.v approx_mul.v; hierarchy -top mac_array -chparam LANES 8 -chparam MODE $1 -chparam K $2; synth -flatten; stat" 2>&1 | grep "Number of cells" | tail -1 | sed "s/^/MODE=$1 K=$2 /"
  done
fi
