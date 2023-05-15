import numpy as np
import string
from dataclasses import dataclass, asdict, field
import hashlib
import json
from os import listdir
from os.path import isfile, join
import pandas as pd
from pathlib import Path
from typing import Any
import pickle
from datetime import datetime
import os

from src.dataclass import (Input, Lattice, Parameter, Train, Save,
                           Processed_Input, Topology, Conjugate, Result)

from pathlib import Path


def save_result(input: Input, result: Result) -> None:
    #! Hash vs random key
    # From input, create directory to store input/result
    key = hashlib.sha1(str(input).encode()).hexdigest()[:6]
    # key = "".join(
    #     np.random.choice(list(string.ascii_lowercase + string.digits), 6)
    # )

    dir_path = Path(f"./result")
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

    if input.save.save:
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


def load_result() -> pd.DataFrame:
    dir_path = Path(f"./result")
    filenames = [f for f in listdir(dir_path) if isfile(
        join(dir_path, f)) if ".pkl" in f]
    list_result = []
    for filename in filenames:
        result = join(dir_path, filename)
        if os.path.getsize(result) > 0:
            with open(result, "rb") as file:
                results = pickle.load(file)
                list_result.append(results)

    df = pd.DataFrame(list_result)  # .drop(columns=[])
    return df
