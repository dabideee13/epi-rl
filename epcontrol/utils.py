# Deep reinforcement learning for large-scale epidemic control
# Copyright (C) 2020  Pieter Libin, Arno Moonens, Fabian Perez-Sanjines.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to pieter.libin@ai.vub.ac.be or arno.moonens@vub.be.

#dis: derivative of the infected counts
#ddis: second derivative of the infected counts
import csv
from pathlib import Path
from typing import List, Optional

import numpy as np


def find_peaks(inf_normalized, dis, ddis, threshold):
    def around_zero(a, b):
        return (a < 0 < b) or (b < 0 < a)

    peaks = []
    for idx, i in enumerate(dis[:-1]):
        if (i == 0 or around_zero(i, dis[idx + 1])) and inf_normalized[idx] > threshold:
            if ddis[idx] < 0:
                peaks.append(idx)

    return peaks


def export_states(states: List[np.ndarray], filename: str = 'states.csv') -> None:

    # FIXME: add path to destination folder: 'data/processed/states'
    states_path = lambda filename: Path.joinpath(Path.cwd(), 'data/processed/states', filename)

    with open(states_path(filename), 'w') as f:
        write = csv.writer(f)
        write.writerow(states)


def export_rewards(rewards: List[np.ndarray], filename: str = 'rewards.csv') -> None:

    # FIXME: add path to destination folder: 'data/processed/states'
    rewards_path = lambda filename: Path.joinpath(Path.cwd(), 'data/processed/rewards', filename)

    with open(rewards_path(filename), 'w') as f:
        write = csv.writer(f)
        write.writerow(rewards)


def export_actions(actions: List[np.ndarray], filename: str = 'actions.csv') -> None:

    # FIXME: add path to destination folder: 'data/processed/states'
    actions_path = lambda filename: Path.joinpath(Path.cwd(), 'data/processed/actions', filename)

    with open(actions_path(filename), 'w') as f:
        write = csv.writer(f)
        write.writerow(actions)


def read_rewards(filename: str) -> List[str]:
    with open(filename, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        rewards = list(csv_reader)[0]
    return rewards
