from __future__ import annotations

import hashlib
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import os
from os import listdir
from os.path import isfile, join

import numpy as np
import numpy.typing as npt
import json
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.dataclass import (Input, Result)


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

    if input.save.location == "result":
        with open(Path(f"./setting") / f"{key}.json", "w") as file:
            json.dump(output, file)

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


# def load_result(location: str) -> pd.DataFrame:
#     dir_path = Path(f"./{location}")
#     filenames = [f for f in listdir(dir_path) if isfile(
#         join(dir_path, f)) if ".pkl" in f]
#     list_result = []
#     for filename in filenames:
#         result = join(dir_path, filename)
#         if os.path.getsize(result) > 0:
#             with open(result, "rb") as file:
#                 results = pickle.load(file)
#                 list_result.append(results)

#     df = pd.DataFrame(list_result)  # .drop(columns=[])
#     return df


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


def load_setting() -> pd.DataFrame:
    def filter_file(f: Path) -> bool:
        return f.is_file() and (f.suffix == ".json") and f.stat().st_size > 0

    # * Scan the setting directory and gather result files
    setting_dir = Path(f"./setting")
    setting_files = [f for f in setting_dir.iterdir() if filter_file(f)]

    # * Read files
    settings: list[dict[str, Any]] = []
    for file in setting_files:
        with open(file, "rb") as f:
            # result = pickle.load(f)
            # if result["state"] != state or result["dim"] != dim:
            #     continue
            settings.append(json.load(f))

    # * Concatenate to single dataframe
    df = pd.DataFrame(settings)
    return df


def get_result(df: pd.DataFrame,
               state: int, dimension: int, size: int, Jv: float) -> Data:  # tuple[pd.DataFrame, Data]
    filter_ = (df["state"] == state) & (df["size"] == size) & (
        df["dimension"] == dimension) & (df["Jv"] == Jv)

    frame = df[filter_].sort_values("T", ascending=True)

    return Data(
        state=state,
        size=size,
        dimension=dimension,
        Jv=Jv,
        temperature=np.array(frame["T"]),
        order_parameter=np.array(frame["order_parameter"]),
        susceptibility=np.array(frame["susceptibility"]),
        binder_cumulant=np.array(frame["binder_cumulant"]),
        spin_glass_order=np.array(frame["spin_glass_order"]),
        spin_glass_suscept=np.array(frame["spin_glass_suscept"]),
        spin_glass_binder=np.array(frame["spin_glass_binder"]),
        energy=np.array(frame["energy"]),
        specific_heat=np.array(frame["specific_heat"]),
        irreducible_distance=np.array(frame["irreducible_distance"]),
        correlation_function=np.array(frame["correlation_function"]),
        correlation_length=np.zeros(np.size(np.array(frame["T"])),
                                    )
    )


def get_correlation_length(data: Data, x_min=None, x_max=None, y_min=None) -> npt.NDArray:
    irreducible_distance, correlation_function = (
        data.irreducible_distance,
        data.correlation_function,
    )

    if x_min is None:
        x_min = 0.0
    if x_max is None:
        x_max = data.size/2/np.sqrt(2)
    if y_min is None:
        y_min = 1.e-7

    correlation_length = []

    for i, distance_list in enumerate(irreducible_distance):
        correlation = correlation_function[i]
        x, y = [], []
        for j, distance in enumerate(distance_list):
            if (x_min <= distance <= x_max and correlation[j] >= y_min):
                x.append(distance)
                y.append(np.log(correlation[j]))

        x, y = np.array(x).reshape((-1, 1)), np.array(y)

        # print(x, y)
        model = LinearRegression().fit(x, y)
        correlation_length.append(-1.0/model.coef_)

    correlation_length = np.array(correlation_length)
    return correlation_length


def get_label(data: Data, mode: str) -> str:
    if (mode == "size"):
        return get_size(data)
    elif (mode == "Jv"):
        return get_Jv(data)
    raise ValueError("label mode should be either 'size' or 'Jv'")


def get_list(df: pd.DataFrame, list_name: str) -> list:
    data_list = []

    for data in df[f"{list_name}"]:
        if (data not in data_list):
            data_list.append(data)

    data_list.sort()

    return data_list


def get_size(data: Data) -> str:
    size_print = f"Size = {data.size} "

    for _ in range(data.dimension-1):
        size_print += f"x {data.size}"

    return size_print


def get_Jv(data: Data) -> str:

    return f"Jv = {data.Jv}"


def delete_result(result_dir: str, key_names: list[str]) -> None:
    for key in key_names:
        target_setting = Path(f"./setting/{key}.json")
        target_file = Path(f"./{result_dir}/{key}.pkl")

        try:
            target_setting.unlink()

        except OSError:
            print(f"No setting found for key in setting: {key}")
        
        try:
            target_file.unlink()
        
        except OSError:
            print(f"No file found for key in {result_dir}: {key}")


# def delete_result(location: str, key_name: npt.NDArray) -> None:
#     import os

#     # Try to delete the file.

#     for key in key_name:
#         try:
#             os.remove(f"./{location}/{key}.pkl")
#             os.remove(f"./setting/{key}.json")
#         except OSError as e:
#             # If it fails, inform the user.
#             print("Error: %s - %s." % (e.filename, e.strerror))
