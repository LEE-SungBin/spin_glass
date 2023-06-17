import numpy as np
import numpy.typing as npt
# from numba import jit, njit, prange
import pickle
from dataclasses import dataclass, asdict, field
import argparse
import hashlib
import json
from os import listdir
from os.path import isfile, join
# import pandas as pd
from pathlib import Path
from typing import Any
import time
import sys
import copy
import concurrent.futures as cf

from src.dataclass import (
    Input, Lattice, Parameter, Train, Save,
    Processed_Input, Topology, Conjugate, Result, summarize_results
)
from src.manage_data import save_result, save_log, load_result
from src.initial_state import get_initial_state
from src.process_input import (get_processed_input, get_T_and_H, get_J)
from src.metropolis import execute_metropolis_update
from src.function import (time_correlation, column_average_2d)
from src.process_output import get_result


def run_ensemble(
    input: Input,
    processed_input: Processed_Input,
    # J: npt.NDArray,
    ensemble_num: int
) -> tuple[int, Result]:
    # tuple[int, tuple, npt.NDArray[np.float64], int]

    (size, dimension, iteration, sweep,
     measurement, interval, recent, threshold) = (
        input.lattice.size,
        input.lattice.dimension,
        input.train.iteration,
        input.train.sweep,
        input.train.measurement,
        input.train.interval,
        input.train.recent,
        input.train.threshold,
    )

    begin = time.perf_counter()

    J = get_J(input, processed_input)  # get coupling parameter

    initial = get_initial_state(input)  # initialize state
    update = initial.copy()
    autocorr, autocorr_len = np.empty(
        iteration+1, dtype=np.float64), 1  # autocorrelation
    autocorr[0] = time_correlation(initial, initial, size**dimension)

    now = time.perf_counter()
    # Removing initial state effect until autocorrelation satsisfies certain criteria
    for _ in range(iteration):
        update = execute_metropolis_update(
            input, processed_input, J, update)
        autocorr[autocorr_len] = np.abs(time_correlation(
            update, initial, size**dimension))
        autocorr_len += 1
        if autocorr_len > recent:
            temp = autocorr[autocorr_len-recent:autocorr_len]
            if np.average(temp) < threshold or np.std(temp) < threshold:
                break

    # if (ensemble_num == 1):
    #     print(
    #         f"initial effect removed, iter: {autocorr_len}, time: {time.perf_counter()-now}s, avg: {np.average(autocorr[autocorr_len-recent:autocorr_len])}, std: {np.std(autocorr[autocorr_len-recent:autocorr_len])}")

    now = time.perf_counter()
    # Collect raw output after performing metropolis update
    raw_output = np.empty((measurement, size**dimension), dtype=np.complex128)
    for i in range(sweep):
        update = execute_metropolis_update(input, processed_input, J, update)
        if autocorr_len <= iteration:
            autocorr[autocorr_len] = np.abs(
                time_correlation(update, initial, size**dimension))
            autocorr_len += 1
        if i % interval == interval - 1:
            # ! .copy() maybe not essential?
            raw_output[int(i/interval)] = update.copy()

    # if (ensemble_num == 1):
    #     print(
    #         f"raw output collected, iter: {sweep}, time: {time.perf_counter()-now}s")

    now = time.perf_counter()
    # process raw output
    result = get_result(input, processed_input, raw_output, J)

    # if (ensemble_num == 1):
    #     print(f"raw output processed, time: {time.perf_counter()-now}s")
    result.autocorrelation = np.array(autocorr)
    result.time = time.perf_counter() - begin

    return ensemble_num, result


def sampling(
    input: Input,
    processed_input: Processed_Input,
) -> None:

    max_workers, ensemble = (
        input.train.max_workers,
        input.train.ensemble,
    )

    with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_ensemble, input, processed_input, i + 1)
            for i in range(ensemble)  # execute each ensemble
        ]

        ensemble_results: list[Result] = []
        finished = 0

        for future in cf.as_completed(futures):
            finished += 1
            ensemble_num, single_result = future.result()
            ensemble_results.append(single_result)

            # print(f"{ensemble_num}: {single_result}")

    result = summarize_results(ensemble_results)

    save_log(input, result)

    if input.save.save:
        save_result(input, result)

    # print(input)
    # print(result, "\n")


def experiment(args: argparse.Namespace) -> None:
    lattice = Lattice(
        args.state, args.size, args.dimension,
        args.ghost, args.initial)
    parameter = Parameter(
        args.Tc, args.Hc, args.Jm, args.Jv, args.mode,
        args.variable, args.multiply, args.base, args.exponent)
    train = Train(
        args.iteration, args.sweep, args.measurement, args.interval,
        args.ensemble, args.max_workers, args.threshold, args.recent)
    save = Save(args.environment, args.location, args.save)

    input = Input(lattice, parameter, train, save)
    input.parameter.T, input.parameter.H = args.T, args.H

    now = time.perf_counter()
    processed_input = get_processed_input(input)  # process input
    # print(f"input processed, time: {time.perf_counter()-now}s")

    input.parameter.T, input.parameter.H = get_T_and_H(
        input)  # get temperature and external field

    # print(input, "\n")
    sampling(input, processed_input)


parser = argparse.ArgumentParser()
"""
Lattice Condition
"""
parser.add_argument("-q", "--state", type=int, default=3)
parser.add_argument("-N", "--size", type=int, default=8)
parser.add_argument("-d", "--dimension", type=int, default=2)
parser.add_argument("-ghost", "--ghost", type=int, default=0)
parser.add_argument("-init", "--initial", type=str,
                    default="uniform", choices={"uniform", "random"})

"""
Parameter Condition
"""
parser.add_argument("-T", "--T", type=float, default=1)
parser.add_argument("-H", "--H", type=float, default=0.0)
parser.add_argument("-Tc", "--Tc", type=float, default=4.5)
parser.add_argument("-Hc", "--Hc", type=float, default=0.0)
parser.add_argument("-Jm", "--Jm", type=float, default=1.0)
parser.add_argument("-Jv", "--Jv", type=float, default=0.0)

parser.add_argument("-mode", "--mode", type=str, default="normal",
                    choices=["normal", "critical", "manual"])
parser.add_argument("-v", "--variable", type=str,
                    default="T", choices=["T", "H"])
parser.add_argument("-m", "--multiply", type=float, default=0.0001)
parser.add_argument("-b", "--base", type=float, default=2.0)
parser.add_argument("-exp", "--exponent", type=float, default=0.0)

"""
Train Condition
"""
parser.add_argument("-itr", "--iteration", type=int, default=1024)
parser.add_argument("-meas", "--measurement", type=int, default=16384)
parser.add_argument("-int", "--interval", type=int, default=1)
parser.add_argument("-ens", "--ensemble", type=int, default=256)
parser.add_argument("-max", "--max_workers", type=int, default=8)
parser.add_argument("-thre", "--threshold", type=float, default=0.05)
parser.add_argument("-rec", "--recent", type=int, default=100)

"""
Save Condition
"""
parser.add_argument("-env", "--environment", type=str,
                    default="server", choices=["server", "local"])
parser.add_argument("-loc", "--location", type=str,
                    default="result", choices=["result", "temp"])
parser.add_argument("-sav", "--save", type=bool,
                    default=True, choices=[True, False])

args = parser.parse_args()

args.sweep = args.measurement * args.interval

experiment(args)

# name1, list1 = 'exponent', [3, 4, 5, 6, 7, 8, 9, 10, 11,
#                             12, 13, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13]
# # name1, list1 = 'exponent', [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, -100, -200, -300, -400, -500, -600, -700, -800, -900, -1000]

# for var1 in list1:
#     setattr(args, name1, var1)
#     experiment(args)
