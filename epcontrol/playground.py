import os
from pathlib import Path
from typing import List, Union, Dict, Callable, Optional

import numpy as np


def cm_ages(first: int, last: int) -> List[str]:

    def increment(ages: str) -> str:
        left, right = ages.split(sep='-')
        new_left, new_right = int(left) + 5, int(right) + 5
        return f'{new_left}-{new_right}'

    def form_group() -> str:
        last = int(first) + 5
        return f'{first}-{last}'

    group = form_group()
    last_index = group.split(sep='-')[1]
    groups = [group]

    while int(last_index) != last:
        groups.append(increment(group))
        group = increment(group)
        last_index = group.split(sep='-')[1]

    return groups


def cm_getter(cm_path: Union[str, Path]) -> Callable[[str], Dict[str, np.ndarray]]:

    def contact_matrix(key: Optional[str] = None) -> Dict[str, np.ndarray]:
        if key:
            return _get_contact_matrices(cm_path)[key]
        return _get_contact_matrices(cm_path)

    return contact_matrix


def _get_contact_matrices(cm_path) -> Dict[str, np.ndarray]:

    def get_files() -> List[str]:
        return [file for file in os.listdir(cm_path) if '.csv' in file]

    def get_names() -> List[str]:
        return [Path(file).stem for file in get_files()]

    return {
        name: read_contact_matrix(cm_path / file)
        for name, file in zip(get_names(), get_files())
    }


def read_contact_matrix(fn):
    return np.genfromtxt(fn, delimiter=',', dtype=np.float32)


if __name__ == '__main__':

    children = cm_ages(0, 5)
    adolescents = cm_ages(5, 20)
    adults = cm_ages(20, 65)
    elderly = cm_ages(65, 80)

    cm_indices = {
        'children': children,
        'adolescents': adolescents,
        'adults': adults,
        'elderly': elderly
    }

    path_IliganCity = Path.joinpath(Path.cwd(), 'data/contacts/contact1')
    path_Greenwich = Path.joinpath(Path.cwd(), 'data/contacts')

    cm_Greenwich = cm_getter(path_Greenwich)
    cm_IliganCity = cm_getter(path_IliganCity)
    print(cm_Greenwich())
    print(cm_IliganCity())


