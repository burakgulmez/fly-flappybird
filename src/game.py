"""Deterministic Flappy Bird (numba). World units: height 1, bird at x = BIRD_X.

Pipe k has left edge x_k = FIRST_X + k*SPACING - frame*SPEED and gap centre gaps[k]
(pre-drawn with the same xorshift32 as the JS demo). Game ends on hitting the floor,
ceiling or a pipe, or when MAX_SCORE pipes have been passed (a "win").
"""
import numpy as np
from numba import njit

MASK32 = 0xFFFFFFFF
GRAV = 0.0020
FLAP_V = 0.0200
MAX_FALL = 0.0400
SPEED = 0.0100
SPACING = 0.55
PIPE_W = 0.10
GAP = 0.30
BIRD_X = 0.25
R = 0.025
FIRST_X = 1.0
MAX_SCORE = 100
N_GAPS = MAX_SCORE + 4
N_FEAT = 6
FEATURE_NAMES = ["height", "velocity", "pipe_dist", "gap_offset", "to_gap_top", "to_gap_bottom"]

# state indices (float64 array)
Y, VY, FRAME, SCORE, DONE, CAUSE = range(6)
CAUSE_NAMES = ["alive", "floor", "ceiling", "pipe", "win"]


@njit(cache=True)
def rng_next(x):
    x ^= (x << 13) & MASK32
    x ^= x >> 17
    x ^= (x << 5) & MASK32
    return x & MASK32


@njit(cache=True)
def new_game(seed):
    x = (seed * 2654435761 + 12345) & MASK32
    if x == 0:
        x = 1
    for _ in range(4):
        x = rng_next(x)
    gaps = np.zeros(N_GAPS)
    for k in range(N_GAPS):
        x = rng_next(x)
        gaps[k] = 0.25 + 0.5 * ((x % 10000) / 9999.0)
    st = np.zeros(6)
    st[Y] = 0.5
    return st, gaps


@njit(cache=True)
def pipe_x(st, k):
    return FIRST_X + k * SPACING - st[FRAME] * SPEED


@njit(cache=True)
def next_pipe(st):
    """Index of the first pipe whose right edge is not yet behind the bird."""
    k = int(st[SCORE])
    while pipe_x(st, k) + PIPE_W < BIRD_X - R:
        k += 1
    return k


@njit(cache=True)
def step(st, gaps, action):
    if st[DONE] > 0:
        return
    if action == 1:
        st[VY] = FLAP_V
    else:
        st[VY] = st[VY] - GRAV
        if st[VY] < -MAX_FALL:
            st[VY] = -MAX_FALL
    st[Y] = st[Y] + st[VY]
    st[FRAME] = st[FRAME] + 1.0
    y = st[Y]
    if y - R < 0.0:
        st[DONE] = 1.0
        st[CAUSE] = 1.0
        return
    if y + R > 1.0:
        st[DONE] = 1.0
        st[CAUSE] = 2.0
        return
    k = int(st[SCORE])
    for j in range(k, k + 3):
        px = pipe_x(st, j)
        if BIRD_X + R > px and BIRD_X - R < px + PIPE_W:
            if y - R < gaps[j] - GAP / 2 or y + R > gaps[j] + GAP / 2:
                st[DONE] = 1.0
                st[CAUSE] = 3.0
                return
    while pipe_x(st, int(st[SCORE])) + PIPE_W < BIRD_X - R:
        st[SCORE] = st[SCORE] + 1.0
        if st[SCORE] >= MAX_SCORE:
            st[DONE] = 1.0
            st[CAUSE] = 4.0
            return


@njit(cache=True)
def features(st, gaps, u):
    k = next_pipe(st)
    y = st[Y]
    g = gaps[k]
    u[0] = 2.0 * y - 1.0
    u[1] = st[VY] / MAX_FALL
    u[2] = (pipe_x(st, k) - BIRD_X) / SPACING
    u[3] = (g - y) * 4.0
    u[4] = (g + GAP / 2 - y) * 4.0
    u[5] = (y - (g - GAP / 2)) * 4.0


@njit(cache=True)
def heuristic_action(st, gaps):
    """Reference rule: flap if the height predicted 3 frames ahead is below gap centre - 0.08."""
    k = next_pipe(st)
    if st[Y] + 3.0 * st[VY] < gaps[k] - 0.08:
        return 1
    return 0
