from src.function import (
    magnetization,
    get_spin_glass,
    hamiltonian,
    kurtosis,
    time_correlation,
    space_correlation,
)
from src.dataclass import (
    Input, Lattice, Parameter, Train, Save,
    Processed_Input, Topology, Conjugate, Result
)


import numpy as np
import numpy.typing as npt
import time


def get_result(
    input: Input,
    processed_input: Processed_Input,
    raw_output: npt.NDArray,
    J: npt.NDArray,
) -> tuple[np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, npt.NDArray]:
    order, suscept, binder, spin_order, spin_suscept, spin_binder = get_order_parameter(
        input, processed_input, raw_output)

    energy, specific = get_total_energy(input, processed_input, raw_output, J)
    correlation = get_correlation_function(input, processed_input, raw_output)

    return order, suscept, binder, spin_order, spin_suscept, spin_binder, energy, specific, correlation


def get_order_parameter(
        input: Input,
        processed_input: Processed_Input,
        raw_output: npt.NDArray,
) -> tuple[np.float64, np.float64, np.float64, np.float64, np.float64, np.float64]:

    size, dimension, T, conjugate_ghost = (
        input.lattice.size,
        input.lattice.dimension,
        input.parameter.T,
        processed_input.conjugate.conjugate_ghost,
    )
    order = magnetization(raw_output, conjugate_ghost)
    spin_glass = get_spin_glass(raw_output)

    return (
        np.average(order),
        np.std(order)**2 * size**dimension / T,
        1 - kurtosis(order.astype(np.float128)) / 3.0,
        np.average(spin_glass),
        np.std(spin_glass)**2 * size**dimension / T,
        1 - kurtosis(spin_glass) / 3.0,
    )


def get_total_energy(
        input: Input,
        processed_input: Processed_Input,
        raw_output: npt.NDArray,
        J: npt.NDArray,
) -> tuple[np.float64, np.float64]:

    size, dimension, T, H, conjugate_ghost = (
        input.lattice.size,
        input.lattice.dimension,
        input.parameter.T,
        input.parameter.H,
        processed_input.conjugate.conjugate_ghost,
    )
    # data = np.array(data)

    temp = hamiltonian(
        raw_output, conjugate_ghost, J, H)

    return np.average(temp), np.std(temp) ** 2 * size**dimension / T**2


def get_correlation_function(
    input: Input,
    processed_input: Processed_Input,
    raw_output: npt.NDArray,
) -> npt.NDArray:

    distance, irreducible_distance = (
        processed_input.topology.distance,
        processed_input.topology.irreducible_distance,
    )

    G_ij = space_correlation(raw_output)  # G(i,j)

    correlation = np.zeros_like(irreducible_distance)
    for i, irr in enumerate(irreducible_distance):
        correlation[i] = G_ij[(distance == irr)].mean()  # G(i,j) -> G(|i-j|)

    return correlation
