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

    size, dimension, state, T, H, interaction_point, complex_state, conjugate_state, complex_ghost = (
        lattice.size,
        lattice.dimension,
        lattice.state,
        parameter.T,
        parameter.H,
        topology.interaction_point,
        conjugate.complex_state,
        conjugate.conjugate_state,
        conjugate.complex_ghost,
    )

    rng = np.random.default_rng()
    flip_coord = np.arange(size**dimension)
    rng.shuffle(flip_coord)
    prob = rng.random(size=size**dimension)
    idx = rng.integers(state-1, size=size**dimension)

    return update_system_state(flip_coord, system_state, prob, idx, state, H, T, complex_ghost, complex_state, conjugate_state, J, interaction_point)


@njit
def update_system_state(flip_coord, system_state, prob, idx, state, H, T, complex_ghost, complex_state, conjugate_state, J, interaction_point):

    for i, x in enumerate(flip_coord):
        index_list = np.arange(state)
        angle = np.int64(
            np.round((np.angle(system_state[x])/2/np.pi*state) % state))
        index_list = np.delete(index_list, angle)
        # rng = np.random.default_rng()
        proposal = index_list[idx[i]]

        current_energy, flip_energy = 0, 0

        interaction = H * complex_ghost
        for point in interaction_point[x]:
            interaction += J[x][point] * system_state[point]

        current_energy = np.real(- conjugate_state[angle] * interaction)
        flip_energy = np.real(- conjugate_state[proposal] * interaction)

        if flip_energy <= current_energy:
            system_state[x] = complex_state[proposal]

        else:
            if (prob[i] <= np.exp(- (flip_energy - current_energy) / T)):
                system_state[x] = complex_state[proposal]

    return system_state
