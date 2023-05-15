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
    input: Input,
    processed_input: Processed_Input,
    J: npt.NDArray[np.float64],
    system_state: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    lattice, conjugate = (
        input.lattice,
        processed_input.conjugate
    )

    state, size, dimension, complex_state = (
        lattice.state,
        lattice.size,
        lattice.dimension,
        conjugate.complex_state,
    )

    rng = np.random.default_rng()

    flip_coord = np.arange(size**dimension)
    rng.shuffle(flip_coord)
    flip_prob = rng.random(size**dimension)

    for i, x in enumerate(flip_coord):
        p = flip_prob[i]
        # probability [0, a_1, a_2, 1] where a_i-a_(i-1) =
        # exp(-beta*E_i)/Z, a_0 = 0, a_n = 1
        prob = partition_function(input, processed_input, J, system_state, x)
        # print(prob)
        # print(prob)
        # system_state[x] = complex_state[(prob > p).argmax() - 1]

        for j in range(state):
            if (p <= prob[j+1] and p >= prob[j]):  # probabilistic update
                system_state[x] = complex_state[j]
                break

    return system_state


def partition_function(
    input: Input,
    processed_input: Processed_Input,
    J: npt.NDArray[np.float64],
    system_state: npt.NDArray[np.complex128],
    x: int,
) -> npt.NDArray[np.float64]:

    lattice, parameter, topology, conjugate = (
        input.lattice,
        input.parameter,
        processed_input.topology,
        processed_input.conjugate
    )

    state, T, H, interaction_point, conjugate_state, complex_ghost = (
        lattice.state,
        parameter.T,
        parameter.H,
        topology.interaction_point,
        conjugate.conjugate_state,
        conjugate.complex_ghost,
    )

    #! Use numpy function as much as possible
    # prob, norm = np.zeros(state + 1), 0
    # temp = 0
    # for value in interaction_point[x]:
    #     temp += system_state[value]
    # for i in range(state):
    #     prob[i + 1] = exp(np.real(conjugate_state[i] * (H * ghost + J * temp)) / T)
    #     norm += prob[i + 1]
    # for i in range(state):
    #     # prob[i + 1] = prob[i] + prob[i + 1] / norm
    #     prob[i + 1] = (prob[i] + prob[i + 1]) / norm    ##?

    prob = np.zeros(state + 1)
    interaction = - H * complex_ghost - \
        np.tensordot(J[x][interaction_point[x]],
                     system_state[interaction_point[x]], (0, 0))
    prob[1:] = np.exp(
        np.real(- conjugate_state * interaction) / T)

    # list of probability [0, a_1, a_2, 1] where a_i-a_(i-1)
    # = exp(-beta*E_i)/Z, a_0 = 0, a_n = 1
    prob = np.cumsum(prob / prob.sum())
    return prob
