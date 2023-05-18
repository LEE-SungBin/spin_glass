from typing import Any
import numpy as np
import numpy.typing as npt
from numba import jit, njit, prange
import sys

from src.dataclass import (
    Input, Lattice, Parameter, Train, Save,
    Processed_Input, Topology, Conjugate, Result
)


def execute_metropolis_update(
    input: Input, processed_input: Processed_Input, J: npt.NDArray[np.float64], system_state: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    lattice, parameter, topology, conjugate = (
        input.lattice,
        input.parameter,
        processed_input.topology,
        processed_input.conjugate
    )

    size, dimension, state, T, H, interaction_point, conjugate_state, complex_ghost = (
        lattice.size,
        lattice.dimension,
        lattice.state,
        parameter.T,
        parameter.H,
        topology.interaction_point,
        conjugate.conjugate_state,
        conjugate.complex_ghost,
    )

    rng = np.random.default_rng()
    flip_coord = np.arange(size**dimension)
    rng.shuffle(flip_coord)
    flip_prob = rng.random(size**dimension)

    return update_system_state(flip_coord, flip_prob, system_state, state, H, T, complex_ghost, conjugate_state, J, interaction_point)


@njit
def update_system_state(flip_coord, flip_prob, system_state, state, H, T, complex_ghost, conjugate_state, J, interaction_point):

    for i, x in enumerate(flip_coord):
        p = flip_prob[i]

        prob = np.zeros(state + 1)

        interaction = - H * complex_ghost
        for point in interaction_point[x]:
            interaction -= J[x][point] * system_state[point]

        prob[1:] = np.exp(
            np.real(- conjugate_state * interaction) / T)

        prob = np.cumsum(prob / prob.sum())

        # system_state[x] = complex_state[(prob > p).argmax() - 1]

        for j in range(state):
            if (p <= prob[j+1] and p >= prob[j]):  # probabilistic update
                system_state[x] = np.exp(j/state*2*np.pi*1j)
                break

    return system_state
