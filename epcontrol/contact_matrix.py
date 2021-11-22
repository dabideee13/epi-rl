"""Generate contact matrices for different actions (quarantine levels)."""

import os
from typing import List, Any, Dict, Tuple, Union, Callable, Optional
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from epcontrol.compartments.contacts.Eames2012 import read_contact_matrix


class ContactMatrix:
    """Contact matrix generator"""

    def __init__(self, **kwargs: Any) -> None:
        self._filename = "Social_contact_matrices_Philippines.xlsx"
        self._school = self._read_data("School")
        self._work = self._read_data("Work")
        self._other_location = self._read_data("other_location")
        self._home = self._read_data("Home")
        self._contact_matrix_raw = self._get_contact(**kwargs)

    @property
    def contact_matrix_raw(self) -> pd.DataFrame:
        return self._contact_matrix_raw

    def _get_path(self, root: Path) -> Path:
        return Path.joinpath(root, 'data/contacts', self._filename)

    def _read_data(self, sheet_name: str = 'All_location') -> pd.DataFrame:

        def reader(path: Path) -> pd.DataFrame:
            return pd.read_excel(
                str(path),
                sheet_name,
                index_col="Ages"
            )

        try:
            return reader(self._get_path(Path.cwd()))
        except FileNotFoundError:
            return reader(self._get_path(Path.cwd().parent))

    def _return_zero(self, df: pd.DataFrame, mode: List[str]) -> pd.DataFrame:
        _df = df.copy(deep=True)
        _df[mode] = 0
        _df.loc[mode] = 0
        return _df

    def _compute_contact(
        self,
        df: pd.DataFrame,
        prop: float,
        sd: float
    ) -> pd.DataFrame:
        return (prop - (prop * sd)) * df

    def _get_contact(
        self,
        home: bool = True,
        prop_work: float = 1.0,
        prop_school: float = 1.0,
        prop_other: float = 1.0,
        sd_work: float = 0.0,
        sd_school: float = 0.0,
        sd_other: float = 0.0,
        home_quarantine: list = [],
        not_allowed_work: list = [],
        not_allowed_other: list = [],
        not_allowed_school: list = []
    ):

        if home_quarantine:
            self._school, self._work, self._other_location = (
                self._return_zero(self._school, home_quarantine),
                self._return_zero(self._work, home_quarantine),
                self._return_zero(self._other_location, home_quarantine)
            )

        if not_allowed_work:
            self._work = self._return_zero(self._work, not_allowed_work)

        if not_allowed_other:
            self._other_location = self._return_zero(
                self._other_location,
                not_allowed_other
            )

        if not_allowed_school:
            self._school = self._return_zero(self._school, not_allowed_school)

        contact_school, contact_work, contact_other = (
            self._compute_contact(self._school, prop_school, sd_school),
            self._compute_contact(self._work, prop_work, sd_work),
            self._compute_contact(self._other_location, prop_other, sd_other)
        )

        contact = contact_school + contact_work + contact_other

        if home:
            contact += self._home

        return contact

    def _cm_mean(
        self,
        first_group: List[str],
        second_group: List[str]
    ) -> float:
        return (
            self.contact_matrix_raw[first_group]
            .loc[second_group]
            .mean()
            .mean()
        )

    def get_contact_matrix(
        self,
        cm_indices: Dict[str, List[str]]
    ) -> pd.DataFrame:

        indices_combinations: Tuple[str, str] = list(
            itertools.product(cm_indices.keys(), repeat=2)
        )

        contact_matrix: Dict[str, list] = {
            combination[0]: list() for combination in indices_combinations
        }

        for combination in indices_combinations:
            contact_matrix[combination[0]].append(
                self._cm_mean(
                    cm_indices[combination[0]],
                    cm_indices[combination[1]]
                )
            )

        return pd.DataFrame(contact_matrix, index=contact_matrix.keys())


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


def main():

    # Define age groups
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

    closed_cm = ContactMatrix(
        # home=False,
        prop_school=0,
        prop_work=0,
        prop_other=0,
        sd_work=1,
        sd_school=1,
        sd_other=1
    )

    open_cm = ContactMatrix(
        # home=False,
        prop_school=1,
        prop_work=1,
        prop_other=1,
        sd_work=0,
        sd_school=0,
        sd_other=0
    )

    closed_cm = closed_cm.get_contact_matrix(cm_indices)
    open_cm = open_cm.get_contact_matrix(cm_indices)

    path_Iligan = Path.joinpath(Path.cwd(), 'data/contacts/contact1')
    path_Greenwich = Path.joinpath(Path.cwd(), 'data/contacts')

    closed_cm.to_csv(
        Path.joinpath(path_Iligan, 'closed_cm.csv'),
        header=False,
        index=False
    )
    open_cm.to_csv(
        Path.joinpath(path_Iligan, 'open_cm.csv'),
        header=False,
        index=False
    )

    cm_Greenwich = cm_getter(path_Greenwich)
    cm_Iligan = cm_getter(path_Iligan)
    print(cm_Greenwich())
    print(cm_Iligan())


if __name__ == '__main__':
    main()


# TODO: Add export function
# TODO: Add file manager of directories function
