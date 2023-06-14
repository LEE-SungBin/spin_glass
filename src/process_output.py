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
) -> Result:

    now = time.perf_counter()
    result = get_order_parameter(input, processed_input, raw_output)
    # print(f"order parameter processed, {time.perf_counter()-now}s")

    now = time.perf_counter()
    energy, specific = get_total_energy(input, processed_input, raw_output, J)
    result.energy = energy
    result.specific_heat = specific
    # print(f"energy processed, {time.perf_counter()-now}s")

    now = time.perf_counter()
    correlation = get_correlation_function(input, processed_input, raw_output)
    result.correlation_function = correlation
    # print(f"correlation function processed, {time.perf_counter()-now}s")

    result.irreducible_distance = processed_input.topology.irreducible_distance
    result.autocorrelation = np.zeros(input.train.iteration)
    result.time = 0.0

    return result


def get_order_parameter(
        input: Input,
        processed_input: Processed_Input,
        raw_output: npt.NDArray,
) -> Result:

    size, dimension, T, conjugate_ghost = (
        input.lattice.size,
        input.lattice.dimension,
        input.parameter.T,
        processed_input.conjugate.conjugate_ghost,
    )
    order = magnetization(raw_output, conjugate_ghost)
    spin_glass = get_spin_glass(raw_output)

    return Result(
        order_parameter=np.average(order),
        susceptibility=np.std(order)**2 * size**dimension / T,
        # ! overflow at size>128 if np.float64
        binder_cumulant=1-kurtosis(order.astype(np.float128))/3.0,
        spin_glass_order=np.average(spin_glass),
        spin_glass_suscept=np.std(spin_glass)**2 * size**dimension / T,
        # ! overflow at size>128 if np.float64
        spin_glass_binder=1-kurtosis(spin_glass.astype(np.float128)) / 3.0,
        energy=0.0,
        specific_heat=0.0,
        correlation_function=np.array([]),
        irreducible_distance=np.array([]),
        autocorrelation=np.array([]),
        time=0.0
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

    now = time.perf_counter()
    # G(i,j) [size**dimension, size**dimension]
    G_ij = space_correlation(raw_output)
    # print(f"G(i,j) processed, time: {time.perf_counter()-now}s")

    now = time.perf_counter()
    correlation = np.zeros_like(irreducible_distance)
    for i, irr in enumerate(irreducible_distance):
        correlation[i] = G_ij[(distance == irr)].mean()  # ! G(i,j) -> G(|i-j|)
    # print(f"G|i-j| processed, time: {time.perf_counter()-now}s")

    return correlation
