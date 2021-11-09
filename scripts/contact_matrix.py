"""Generate contact matrices for different actions (quarantine levels)."""

import os
from typing import List, Any, Dict, Tuple, Union
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


class CMAgeGroup:

    def _increment(self, ages: str) -> str:
        left, right = ages.split(sep='-')
        new_left, new_right = int(left) + 5, int(right) + 5
        return f'{new_left}-{new_right}'

    def _form_group(self, first) -> str:
        last = int(first) + 5
        return f'{first}-{last}'

    def ages(self, first, last) -> List[str]:

        group = self._form_group(first)
        last_index = group.split(sep='-')[1]
        groups = [group]

        while int(last_index) != last:
            groups.append(self._increment(group))
            group = self._increment(group)
            last_index = group.split(sep='-')[1]

        return groups


class CMGetter:
    """Contact matrices retriever"""

    def __init__(self, cm_path: Union[Path, str]) -> None:
        self._cm_path = cm_path
        self._files = self._get_files()
        self._names = self._get_names()
        self._contact_matrices = self._get_contact_matrices()

    @property
    def contact_matrices(self) -> Dict[str, np.ndarray]:
        return self._contact_matrices

    def _get_files(self) -> List[str]:
        return [file for file in os.listdir(self._cm_path) if '.csv' in file]

    def _get_names(self) -> List[str]:
        return [Path(file).stem for file in self._files]

    def _get_contact_matrices(self) -> Dict[str, np.ndarray]:
        return {
            name: read_contact_matrix(self._cm_path / file)
            for name, file in zip(self._names, self._files)
        }


def main():

    # Define age groups
    cm = CMAgeGroup()
    children = cm.ages(0, 5)
    adolescents = cm.ages(5, 20)
    adults = cm.ages(20, 65)
    elderly = cm.ages(65, 80)

    cm_indices = {
        'children': children,
        'adolescents': adolescents,
        'adults': adults,
        'elderly': elderly
    }

    contact_matrix = ContactMatrix()
    print(contact_matrix.get_contact_matrix(cm_indices))

    cm_path = Path('/home/ubuntu/Temporary/epi-rl/data/contacts')
    contact_matrices = CMGetter(cm_path)
    print(contact_matrices.contact_matrices)


if __name__ == '__main__':
    main()


# TODO: Add export function
# TODO: Add file manager of directories function
