#!/usr/bin/env bash
# Reproduce the Flappy Bird study (about 5-20 min on an 8-core laptop; the lesion step is the slowest).
# Needs the FlyWire tables in data/raw (see data/README.md).
set -e
cd "$(dirname "$0")"
python -c "from src import connectome as C; import json; json.dump(C.build(), open('results/circuit_stats.json','w'), indent=1)"
for s in 0 1 2 3 4; do python experiments/train.py --seed $s; done
python experiments/reference_agents.py
python experiments/lesion.py
python experiments/export_demo.py
python figures/make_all.py
python paper/make_numbers.py
