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
    Processed_Input, Topology, Conjugate, Result
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
) -> tuple[int, tuple, npt.NDArray[np.float64], int]:

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

    J = get_J(input, processed_input)

    initial = get_initial_state(input)
    update = initial.copy()
    autocorr, autocorr_len = np.empty(iteration+1, dtype=np.float64), 1
    autocorr[0] = time_correlation(initial, initial, size**dimension)

    # Removing initial state effect until autoautocorrelation satsisfies certain criteria
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

    temp = autocorr[autocorr_len-recent:autocorr_len]

    # Collect raw output after performing metropolis update
    raw_output = np.empty((measurement, size**dimension), dtype=np.complex128)
    for i in range(sweep):
        update = execute_metropolis_update(input, processed_input, J, update)
        if autocorr_len <= iteration:
            autocorr[autocorr_len] = np.abs(
                time_correlation(update, initial, size**dimension))
            autocorr_len += 1
        if i % interval == interval - 1:
            raw_output[int(i/interval)] = update.copy()

    result = get_result(input, processed_input, raw_output, J)

    return ensemble_num, result, np.array(autocorr), int(time.perf_counter() - begin)


def sampling(
    input: Input,
    processed_input: Processed_Input,
) -> None:

    max_workers, ensemble, irreducible_distance = (
        input.train.max_workers,
        input.train.ensemble,
        processed_input.topology.irreducible_distance
    )

    order, suscept, binder = [], [], []
    spin_order, spin_suscept, spin_binder = [], [], []
    energy, specific = [], []
    corr_time, corr_space = [], []
    time = []

    # J = get_J(input, processed_input)

    with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_ensemble, input, processed_input, i + 1)
            for i in range(ensemble)
        ]

        finished = 0
        for future in cf.as_completed(futures):
            finished += 1
            number, single_result, autocorr, ex_time = future.result()

            order.append(single_result[0])
            suscept.append(single_result[1])
            binder.append(single_result[2])
            spin_order.append(single_result[3])
            spin_suscept.append(single_result[4])
            spin_binder.append(single_result[5])
            energy.append(single_result[6])
            specific.append(single_result[7])
            corr_space.append(single_result[8])
            corr_time.append(autocorr)
            time.append(ex_time)

    result = Result(
        order_parameter=abs(np.average(order)).item(),
        susceptibility=np.average(suscept).item(),
        binder_cumulant=abs(np.average(binder)).item(),
        spin_glass_order=np.average(spin_order).item(),
        spin_glass_suscept=np.average(spin_suscept).item(),
        spin_glass_binder=np.average(spin_binder).item(),
        energy=np.average(energy).item(),
        specific_heat=np.average(specific).item(),
        irreducible_distance=irreducible_distance,
        correlation_function=column_average_2d(corr_space),
        autocorrelation=column_average_2d(corr_time),
        time=int(np.average(time).item())
    )

    if input.save.save:
        save_result(input, result)
        save_log(input, result)

    # print(
    #     "T: {}, Jm: {}, Jv: {}, H: {}, order: {}, suscept: {}, binder: {}, spin order: {}, spin suscept: {}, spin binder: {}, energy: {}, specific"
    #     " heat: {}".format(
    #         input.parameter.T,
    #         input.parameter.Jm,
    #         input.parameter.Jv,
    #         input.parameter.H,
    #         result.order_parameter,
    #         result.susceptibility,
    #         result.binder_cumulant,
    #         result.spin_glass_order,
    #         result.spin_glass_suscept,
    #         result.spin_glass_binder,
    #         result.energy,
    #         result.specific_heat,
    #     )
    # )

    # print(f"correlation function: {result.correlation_function}")


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

    processed_input = get_processed_input(input)
    input.parameter.T, input.parameter.H = get_T_and_H(input)

    # print(input, "\n")
    sampling(input, processed_input)


parser = argparse.ArgumentParser()
parser.add_argument("-q", "--state", type=int, default=3)
parser.add_argument("-N", "--size", type=int, default=8)
parser.add_argument("-d", "--dimension", type=int, default=2)
parser.add_argument("-Tc", "--Tc", type=float, default=1.5)
parser.add_argument("-Jm", "--Jm", type=float, default=1.0)
parser.add_argument("-Jv", "--Jv", type=float, default=0.0)
parser.add_argument("-m", "--mode", type=str, default="normal",
                    choices=["normal", "critical", "manual"])
parser.add_argument("-v", "--variable", type=str,
                    default="T", choices=["T", "H"])
parser.add_argument("-m", "--multiply", type=float, default=0.0001)
parser.add_argument("-b", "--base", type=float, default=2.0)
parser.add_argument("-exp", "--exponent", type=float, default=0.0)
parser.add_argument("-itr", "--iteration", type=int, default=1024)
parser.add_argument("-meas", "--measurement", type=int, default=16384)
parser.add_argument("-int", "--interval", type=int, default=1)
parser.add_argument("-ens", "--ensemble", type=int, default=1024)
parser.add_argument("-max", "--max_workers", type=int, default=8)
parser.add_argument("-loc", "--location", type=str,
                    default="result", choices=["result", "temp"])
args = parser.parse_args()

"""
Memory
Max(measurement * size**dim, size**(2*dim)) * max_workers

Performance
measurement * size**dim * ensembles / max_workers
"""

"""
Lattice Condition
"""
# args.state = 3
# args.size = 2**4
# args.dimension = 2
args.ghost = 0
args.initial = "uniform"  # "uniform", "random"

"""
Parameter Condition
"""
args.T, args.H = 1.0, 0.0
# q=2: 2.2692, q=3: 1.4925
# args.Tc = 1/((1-1/args.state)*np.log(1+np.sqrt(args.state)))
# args.Tc = 1.5
args.Hc = 0.0
# args.Jm = 1.0
# args.Jv = 1.0

# args.mode = "normal"  # 'normal', 'critical' and 'manual'
# args.variable = "T"  # 'T', 'J'
# args.multiply = 0.1**4
# args.base = 2.0
# args.exponent = 13

"""
Train Condition
"""
# args.iteration = 2**10
# args.measurement = 2**12
# args.interval = 2**3
args.sweep = args.measurement * args.interval
# args.ensemble = 2**3
# args.max_workers = 2**1
args.threshold = 5 * 0.1**2
args.recent = 10**2

"""
Save Condition
"""
args.environment = "server"  # "server" or "local"
# args.location = "temp"
args.save = True  # True or False

experiment(args)

# name1, list1 = 'exponent', [3, 4, 5, 6, 7, 8, 9, 10, 11,
#                             12, 13, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13]
# # name1, list1 = 'exponent', [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, -100, -200, -300, -400, -500, -600, -700, -800, -900, -1000]

# for var1 in list1:
#     setattr(args, name1, var1)
#     experiment(args)
