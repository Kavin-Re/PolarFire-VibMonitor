# Contributing to PolarFire-VibMonitor

Thanks for your interest. PolarFire-VibMonitor is a software model (synthetic data, numpy MLP, approximate multipliers) plus an RTL skeleton for a vibration monitor. Nothing here is measured on a PolarFire device.

## Getting started

Python 3 with numpy and matplotlib. To reproduce the results:

```
python3 build_data.py            # about 6 minutes
python3 run_experiments.py       # 1-3 minutes
python3 figures.py
python3 make_results_md.py
```

## Testing

```
python3 -m unittest discover -s tests
cd rtl && ./run_sim.sh           # needs iverilog + vvp; yosys optional (generic cell counts only)
```

## Pull requests

- Open an issue first for larger changes.
- Keep PRs focused and fill in the pull request template.
- Follow `.editorconfig` and match the style of surrounding code.
- Say which simulator and tool versions you used for RTL changes.

## Conduct

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
