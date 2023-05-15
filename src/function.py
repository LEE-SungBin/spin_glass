import numpy as np
import numpy.typing as npt
from numba import jit, njit, prange


def kurtosis(
    arr: npt.NDArray
) -> np.float64:
    length = len(arr)

    return (
        np.einsum("i,i,i,i->", arr, arr, arr, arr, optimize=True)
        / length)/(np.einsum("i,i->", arr, arr, optimize=True)/length)**2


def magnetization(
    array: npt.NDArray,
    conjugate_ghost: npt.NDArray
) -> npt.NDArray:
    """
    array: [measurement, size**dim]
    conjugate_ghost: [size**dim]
    """

    return np.real(
        np.tensordot(conjugate_ghost, array, (0, 1)) / np.size(array[0])
    )


def get_spin_glass(
    array: npt.NDArray,
    complex_ghost: complex
) -> npt.NDArray:
    """
    array: [measurement, size**dim]
    """

    spin_glass = np.real(np.conjugate(complex_ghost) *
                         np.einsum("ij->j", array, optimize=True) / np.size(array[:, 0]))

    return spin_glass**2
    # return (spin_glass**2).sum() / np.size(array[0])

# return hamiltonian <-J*sigma(i)sigma(j)-h*sigma(i)>


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


# return connected-correlation <sigma(i),sigma(j)>-<sigma(i)><sigma(j)> between arr1 and arr2
def space_correlation(
    array: npt.NDArray,
) -> npt.NDArray:
    """
    array: [measurement, size**dim]
    """
    measurement = np.size(array[:, 0])

    average = np.einsum("ij->j", array, optimize=True) / measurement
    corr = np.tensordot(np.conjugate(array), array, (0, 0)) / measurement

    return np.real(corr - np.tensordot(np.conjugate(average), average, axes=0))


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
