# Flappy Bird with a fly looming-escape circuit

One of three small hobby studies in which a circuit from the fruit fly connectome plays a game. The other two: [Snake](https://github.com/burakgulmez/fly-snake), [Pong](https://github.com/burakgulmez/fly-pong).

A fixed network wired like the fruit fly's **looming-escape circuit** decides when to flap:
- **Inputs:** 314 loom-detecting neurons (LPLC2, LC4)
- **In between:** 60 intermediate neurons
- **Outputs:** 10 escape descending neurons, including the giant fiber DNp01

That is 384 neurons and 5,810 fixed connections. Only 64 numbers are trained.

**[Open the demo](demo/index.html)**: watch it fly, see the neurons, or silence the intermediates or the giant fiber.

## Results

- **All 5 training runs** passed the maximum of **100 pipes in all 200 unseen test games**, after about
  50 s of training each.
- **Comparison:** a hand-written rule averages 84.2 pipes (68% wins); random flapping averages 0.07.
- **Silencing all 60 intermediate neurons changes nothing:** the loom detectors also connect directly to
  the escape neurons. Each trained network has a different single weak spot (for example, one fails
  without the giant fiber).

Details are in [`paper/main.tex`](paper/main.tex).

## How it works

- **Game** (`src/game.py`): numba; gravity 0.002, flap velocity 0.02, pipe gap 0.30, gap centres from a
  seeded xorshift; a game is won at 100 pipes.
- **Circuit** (`src/connectome.py`): LPLC2 + LC4 → neurons with ≥ 40 synapses from the inputs and to the
  outputs → DNp01/02/04/06/11.
- **Network and training** (`src/brain.py`, `experiments/train.py`): rate model with 4 updates per frame,
  6 readings → 8 input groups; one readout weight per escape-neuron type; flap if the readout > 0.
  CMA-ES, 1M frames per run.

## Reproduce

```bash
# FlyWire tables in data/raw (see data/README.md)
bash reproduce.sh
python -m pytest tests
```

## Layout

```
src/            game engine (numba), circuit extraction, network
experiments/    training and analysis scripts
results/        logged results (CSV / JSON) used by the paper
figures/        make_all.py -> figures
paper/          main.tex + make_numbers.py (every number is generated from results/)
arxiv/          self-contained arXiv submission package
demo/           static HTML/JS demo (index.html, sim.js, weights.js) and a short video
tests/          pytest (game rules, Python/JS equivalence)
tools/          reference fetcher and verifier (OpenAlex, Crossref), arXiv packaging, video recorder
data/           where the FlyWire tables go (see data/README.md)
```

Install with `pip install -r requirements.txt` (Python 3.10+). Node.js is only needed for the
Python/JS equivalence test and for recording the demo video.

## Data and credit

The connectome data come from the FlyWire project (FAFB v783) via the
[FlyWire Codex](https://codex.flywire.ai) download page and are not included here.
Please follow the FlyWire citation guidelines when using the data.

This is not a model of how flies think, and flies do not play Flappy Bird. It is a small,
reproducible experiment with real connectome data.

## License

Code: MIT (see `LICENSE`). The FlyWire data are subject to their own terms.
