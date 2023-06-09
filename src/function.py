import numpy as np
import numpy.typing as npt
# from numba import jit, njit, prange


def kurtosis(
    arr: npt.NDArray
) -> np.float64:
    length = len(arr)

    return (
        np.einsum("i,i,i,i->", arr, arr, arr, arr, optimize=True)
        / length)/(np.einsum("i,i->", arr, arr, optimize=True)/length)**2  # ! overflow at size>128 if np.float64


def magnetization(
    array: npt.NDArray,
    conjugate_ghost: npt.NDArray
) -> npt.NDArray:
    """
    array: [measurement, size**dim]
    conjugate_ghost: [size**dim]
    """

    length = np.size(array[0])

    return np.real(
        np.tensordot(conjugate_ghost, array, (0, 1)) / length
    )


def get_spin_glass(
    array: npt.NDArray,
) -> npt.NDArray:
    """
    array: [measurement, size**dim]
    """

    measurement = np.size(array[:, 0])

    spin_glass = np.einsum(
        "ij->j", array, optimize=True) / measurement

    return np.real(spin_glass*np.conjugate(spin_glass))


def hamiltonian(
    array: npt.NDArray,
    conjugate_ghost: npt.NDArray,
    J: npt.NDArray,
    H: float,
) -> npt.NDArray:
    """
    array: [measurement, size**dim]
    conjugate_ghost: [size**dim]
    J: [size**dim, size**dim]
    """

    return np.real(
        - H * np.tensordot(conjugate_ghost, array, (0, 1))
        - np.einsum("ji,ij->i", np.tensordot(J, array, (0, 1)),
                    np.conjugate(array), optimize=True) / 2.0
    ) / np.size(array[0])


# return autocorrelation <sigma(t=0),sigmma(t)>
def time_correlation(
    arr1: npt.NDArray[np.complex128],
    arr2: npt.NDArray[np.complex128],
    length: int
) -> float:
    return np.real(np.vdot(arr1, arr2)).item() / length


# return connected-correlation <sigma(i),sigma(j)>-<sigma(i)><sigma(j)> between two point in arr
def space_correlation(
    array: npt.NDArray,
) -> npt.NDArray:
    """
    array: [measurement, size**dim]
    return: [size**dim, size**dim]
    """
    measurement = np.size(array[:, 0])
    length = np.size(array[0])

    average = np.einsum("ij->j", array, optimize=True) / measurement
    corr = np.tensordot(np.conjugate(array), array, (0, 0)) / measurement

    return np.real(corr - np.tensordot(np.conjugate(average), average, axes=0))
    # return sigma_i_sigma_j(array, measurement, length)


# @njit(parallel=True)  # ! is njit faster than np.tensordot? probabily not!
# def sigma_i_sigma_j(array, measurement, length):
#     avg = np.zeros(length)
#     corr = np.zeros((length, length))

#     for i in prange(length):
#         for j in prange(measurement):
#             avg[i] = avg[i] + array[j, i]

#     avg = avg / measurement

#     for i in prange(length):
#         for j in prange(length):
#             for k in prange(measurement):
#                 corr[i][j] = corr[i][j] + \
#                     np.conjugate(array[k, i]) * array[k, j]

#     corr = corr / measurement

#     for i in prange(length):
#         for j in prange(length):
#             corr[i][j] = corr[i][j] - avg[i] * avg[j]

#     return np.real(corr)


def column_average_2d(arrs: list[np.ndarray]) -> npt.NDArray[np.float64]:
    # return np.array([np.abs(arr.mean()) for arr in arrs])
    # return arr.mean(axis=1)

    length = []
    for row in arrs:
        length.append(np.size(row))

    size = np.max(np.array(length))
    temp = [[] for _ in range(size)]

    for row in arrs:
        for i in range(np.size(row)):
            temp[i].append(row[i])

    return np.array([abs(np.average(temp[i])) for i in range(size)])
