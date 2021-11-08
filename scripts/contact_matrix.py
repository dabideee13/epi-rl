"""Generate contact matrices for different actions (quarantine levels)."""

from pathlib import Path
from typing import List

import pandas as pd


class ContactMatrix:
    """Contact matrix generator"""

    def __init__(self) -> None:
        self._filename = "Social_contact_matrices_Philippines.xlsx"
        self._school = self._read_data("School")
        self._work = self._read_data("Work")
        self._other_location = self._read_data("other_location")
        self._home = self._read_data("Home")

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

    def _compute_contact(self, df: pd.DataFrame, prop: float, sd: float) -> pd.DataFrame:
        return (prop - (prop * sd)) * df

    def get_contact(
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
            self._other_location = self._return_zero(self._other_location, not_allowed_other)

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


if __name__ == '__main__':
    contact_matrix = ContactMatrix()
    print(contact_matrix.get_contact())
