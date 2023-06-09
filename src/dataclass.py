import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, asdict, field
import hashlib
import json
from os import listdir
from os.path import isfile, join
import pandas as pd
from pathlib import Path
from typing import Any
import pickle


@dataclass
class Lattice:
    state: int
    size: int
    dimension: int
    ghost: int
    initial: str

    def __post__init__(self) -> None:
        assert self.ghost in range(self.state)
        assert self.initial in ["uniform", "random"]


@dataclass
class Parameter:
    T: float = field(init=False)
    H: float = field(init=False)
    Tc: float
    Hc: float
    Jm: float
    Jv: float
    mode: str
    variable: str
    multiply: float
    base: float
    exponent: float

    def __post__init__(self) -> None:
        assert self.mode in ["normal", "critical", "manual"]
        assert self.variable in ["T", "H"]


@dataclass
class Train:
    iteration: int
    sweep: int
    measurement: int
    interval: int
    ensemble: int
    max_workers: int
    threshold: float
    recent: int


# @dataclass
# class Correlation:
#     max_distance: float = field(init=False)
#     num_of_i_points: int = field(init=False)


@dataclass
class Save:
    environment: str
    location: str
    save: bool

    def __init__(self, environment, location, save):
        self.environment = environment
        self.location = location
        self.save = save

    def __post__init__(self) -> None:
        assert self.environment in ["server", "local"]
        assert self.environment in ["result", "temp"]


@dataclass
class Input:
    lattice: Lattice
    parameter: Parameter
    train: Train
    # correlation: Correlation
    save: Save

    # def to_json(self, file_name: Path) -> None:
    #     """Store information into json file: human-readable"""
    #     data = asdict(self)

    #     with open(file_name, "w") as file:
    #         json.dump(data, file)

    def to_log(self) -> str:
        return " ".join(
            f"{log}"
            for log in [
                self.save.environment,
                self.lattice.state,
                self.lattice.size,
                self.lattice.dimension,
                np.round(self.parameter.T, 4),
                np.round(self.parameter.Jm, 4),
                np.round(self.parameter.Jv, 4),
                np.round(self.parameter.H, 4),
                self.train.iteration,
                self.train.sweep,
                self.train.measurement,
            ]
        )


@dataclass
class Topology:
    coordinate: npt.NDArray
    interaction_point: npt.NDArray
    distance: npt.NDArray
    irreducible_distance: npt.NDArray


@dataclass
class Conjugate:
    complex_state: npt.NDArray
    conjugate_state: npt.NDArray
    complex_ghost: complex
    conjugate_ghost: npt.NDArray


@dataclass
class Processed_Input:
    topology: Topology
    conjugate: Conjugate


@dataclass
class Result:
    order_parameter: float
    susceptibility: float
    binder_cumulant: float
    spin_glass_order: float
    spin_glass_suscept: float
    spin_glass_binder: float
    energy: float
    specific_heat: float
    irreducible_distance: npt.NDArray
    correlation_function: npt.NDArray
    autocorrelation: npt.NDArray
    time: float

    # def to_pickle(self, file_name: Path) -> None:
    #     """Store data into picle file: without loss of precision"""

    #     with open(file_name, "wb") as file:
    #         pickle.dump(self, file)

    def to_log(self) -> str:
        return " ".join(
            f"{log}"
            for log in [
                np.round(self.time, 2),
                np.round(self.order_parameter, 4),
                np.round(self.susceptibility, 4),
                np.round(self.binder_cumulant, 4),
                np.round(self.spin_glass_order, 4),
                np.round(self.spin_glass_suscept, 4),
                np.round(self.spin_glass_binder, 4),
                np.round(self.energy, 4),
                np.round(self.specific_heat, 4),
            ]
        )


def summarize_results(results: list[Result]) -> Result:
    order_parameter = np.abs(
        np.mean([result.order_parameter for result in results])
    ).item()

    susceptibility = np.mean([result.susceptibility for result in results])
    binder_cumulant = np.mean([result.binder_cumulant for result in results])
    spin_glass_order = np.mean([result.spin_glass_order for result in results])
    spin_glass_suscept = np.mean(
        [result.spin_glass_suscept for result in results])
    spin_glass_binder = np.mean(
        [result.spin_glass_binder for result in results])
    energy = np.mean([result.energy for result in results])
    specific_heat = np.mean([result.specific_heat for result in results])
    irreducible_distance = np.mean(
        [result.irreducible_distance for result in results], axis=0)
    correlation_function = np.mean(
        [result.correlation_function for result in results], axis=0)
    autocorrelation = np.mean(
        [result.autocorrelation for result in results], axis=0)
    time = np.mean([result.time for result in results])

    return Result(
        order_parameter=order_parameter,
        susceptibility=susceptibility,
        binder_cumulant=binder_cumulant,
        spin_glass_order=spin_glass_order,
        spin_glass_suscept=spin_glass_suscept,
        spin_glass_binder=spin_glass_binder,
        energy=energy,
        specific_heat=specific_heat,
        irreducible_distance=irreducible_distance,
        correlation_function=correlation_function,
        autocorrelation=autocorrelation,
        time=time
    )
