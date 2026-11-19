#!/bin/sh
# Run from rtl/. Requires iverilog + vvp (and optionally yosys). Generates vectors first.
set -e
python3 gen_vectors.py
iverilog -g2005 -o sim approx_mul.v mac_array.v tb_mac_array.v
vvp sim | tee sim.log
# optional generic (NOT PolarFire) synthesis cell counts, 8 lanes:
for m in "0 0" "1 6" "2 4"; do set -- $m
  yosys -p "read_verilog mac_array.v approx_mul.v; hierarchy -top mac_array -chparam LANES 8 -chparam MODE $1 -chparam K $2; synth -flatten; stat" 2>&1 | grep "Number of cells" | tail -1 | sed "s/^/MODE=$1 K=$2 /"
done
