from __future__ import annotations

import hashlib
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.dataclass import Input, Result


@dataclass
class Data:
    state: int
    size: int
    dimension: int
    Jv: float
    temperature: npt.NDArray
    order_parameter: npt.NDArray
    susceptibility: npt.NDArray
    binder_cumulant: npt.NDArray
    spin_glass_order: npt.NDArray
    spin_glass_suscept: npt.NDArray
    spin_glass_binder: npt.NDArray
    energy: npt.NDArray
    specific_heat: npt.NDArray
    irreducible_distance: npt.NDArray
    correlation_function: npt.NDArray
    correlation_length: npt.NDArray

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Data:
        state: int = df["state"].iloc[0]
        size: int = df["size"].iloc[0]
        dimension: int = df["dimension"].iloc[0]
        Jv: float = df["Jv"].iloc[0]

        assert (df["state"] == state).all(), "More than one state present"
        assert (df["size"] == size).all(), "More than one size present"
        assert (df["dimension"] == dimension).all(
        ), "More than one dimension present"
        assert (df["Jv"] == Jv).all(), "More than one Jv present"

        return cls(
            state=state,
            size=size,
            dimension=dimension,
            Jv=Jv,
            temperature=np.array(df["T"]),
            order_parameter=np.array(df["order_parameter"]),
            susceptibility=np.array(df["susceptibility"]),
            binder_cumulant=np.array(df["binder_cumulant"]),
            spin_glass_order=np.array(df["spin_glass_order"]),
            spin_glass_suscept=np.array(df["spin_glass_suscept"]),
            spin_glass_binder=np.array(df["spin_glass_binder"]),
            energy=np.array(df["energy"]),
            specific_heat=np.array(df["specific_heat"]),
            irreducible_distance=np.array(df["irreducible_distance"]),
            correlation_function=np.array(df["correlation_function"]),
            correlation_length=np.zeros(np.size(np.array(df["T"]))),
        )


def save_result(input: Input, result: Result) -> None:
    #! Hash vs random key
    # From input, create directory to store input/result
    key = hashlib.sha1(str(input).encode()).hexdigest()[:6]
    # key = "".join(
    #     np.random.choice(list(string.ascii_lowercase + string.digits), 6)
    # )

    dir_path = Path(f"./{input.save.location}")
    dir_path.mkdir(parents=True, exist_ok=True)

    output = {
        "key": key,
    }

    output.update(asdict(input.lattice))
    output.update(asdict(input.parameter))
    output.update(asdict(input.train))
    # output.update(asdict(input.correlation))
    output.update(asdict(input.save))
    output.update(asdict(result))

    with open(dir_path / f"{key}.pkl", "wb") as file:
        pickle.dump(output, file)


def save_log(input: Input, result: Result) -> None:
    log = (
        f"{datetime.now().replace(microsecond=0)} {input.to_log()} {result.to_log()}\n"
    )

    dir_path = Path(".")
    dir_path.mkdir(parents=True, exist_ok=True)

    with open(dir_path / "log.txt", "a") as file:
        file.write(log)


def load_result(location: str, state: int | None = None, dim: int | None = None) -> pd.DataFrame:
    def filter_file(f: Path) -> bool:
        return f.is_file() and (f.suffix == ".pkl") and f.stat().st_size > 0

    # * Scan the result directory and gather result files
    result_dir = Path(f"./{location}")
    result_files = [f for f in result_dir.iterdir() if filter_file(f)]

    # * Read files
    results: list[dict[str, Any]] = []
    for file in result_files:
        with open(file, "rb") as f:
            # result = pickle.load(f)
            # if result["state"] != state or result["dim"] != dim:
            #     continue
            results.append(pickle.load(f))

    # * Concatenate to single dataframe
    df = pd.DataFrame(results)
    return df


def filter_df(
    df: pd.DataFrame,
    state: int | None = None,
    dimension: int | None = None,
    size: int | None = None,
    Jv: float | None = None,
    order: str = "T"
) -> pd.DataFrame:
    conditions: list[str] = []
    if state is not None:
        conditions.append(f"state == {state}")
    if dimension is not None:
        conditions.append(f"dimension == {dimension}")
    if size is not None:
        conditions.append(f"size == {size}")
    if Jv is not None:
        conditions.append(f"Jv == {Jv}")

    return df.query(" and ".join(conditions)).sort_values(order, ascending=True)


def get_correlation_length(
    data: Data, x_min=None, x_max=None, y_min=None
) -> npt.NDArray:
    irreducible_distance, correlation_function = (
        data.irreducible_distance,
        data.correlation_function,
    )

    if x_min is None:
        x_min = 0.0
    if x_max is None:
        x_max = data.size / 2 / np.sqrt(2)
    if y_min is None:
        y_min = 1.0e-7

    correlation_length = []

    for i, distance_list in enumerate(irreducible_distance):
        correlation = correlation_function[i]
        x, y = [], []
        for j, distance in enumerate(distance_list):
            if x_min <= distance <= x_max and correlation[j] >= y_min:
                x.append(distance)
                y.append(np.log(correlation[j]))

        x, y = np.array(x).reshape((-1, 1)), np.array(y)

        # print(x, y)
        model = LinearRegression().fit(x, y)
        correlation_length.append(-1.0 / model.coef_)

    correlation_length = np.array(correlation_length)
    return correlation_length


def get_label(data: Data, mode: str) -> str:
    if mode == "size":
        return get_size(data)
    elif mode == "Jv":
        return get_Jv(data)
    raise ValueError("label mode should be either 'size' or 'Jv'")


def get_list(df: pd.DataFrame, list_name: str) -> list:
    data_list = []

    for data in df[f"{list_name}"]:
        if data not in data_list:
            data_list.append(data)

    data_list.sort()

    return data_list


def get_size_label(size: int, dimension: int) -> str:
    label = "Size = "
    label += f" x ".join([f"{size}" for _ in range(dimension)])
    return label


def get_size(data: Data) -> str:
    size_print = f"Size = {data.size} "

    for _ in range(data.dimension - 1):
        size_print += f"x {data.size}"

    return size_print


def get_Jv(data: Data) -> str:
    return f"Jv = {data.Jv}"


def delete_result(result_dir: Path, key_names: list[str]) -> None:
    for key in key_names:
        target_file = result_dir / f"{key}.pkl"
        try:
            target_file.unlink()
        except OSError:
            print(f"Error: {key}")

# def delete_result(location: str, key_name: npt.NDArray) -> None:
#     import os

#     # Try to delete the file.

#     for key in key_name:
#         try:
#             os.remove(f"./{location}/{key}.pkl")
#         except OSError as e:
#             # If it fails, inform the user.
#             print("Error: %s - %s." % (e.filename, e.strerror))
