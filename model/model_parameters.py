from typing import Union, List
from pathlib import Path

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from epcontrol.census.Flux import Table, SingleDistrictStub


@dataclass(frozen=True)
class Census:
    path: Union[Path, str] = 'data/great_brittain/census.csv'
    grouped_census: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'grouped_census', self._grouped_census)

    @property
    def _grouped_census(self) -> pd.DataFrame:
        return pd.read_csv(self.path, index_col=0)


@dataclass(frozen=True)
class SEIRParameters:
    grouped_census: pd.DataFrame
    district_names: List[str] = field(init=False)
    flux: Union[Table, SingleDistrictStub]

    delta: float = 0.5
    r0: float = 1.8
    rho: float = 1
    gamma: float = (1 / 1.8)
    mu: float = None
    sde: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, 'district_names', self._district_names)

        if self.mu is None:
            object.__setattr__(self, 'mu', self._compute_mu())

    def _compute_mu(self) -> float:
        return np.log(self.r0) * .6

    @property
    def _district_names(self) -> List[str]:
        return self.grouped_census.index.to_list()


if __name__ == '__main__':
    district_name = 'Greenwich'

    census = Census()
    seir_parameters = SEIRParameters(
        grouped_census=census.grouped_census,
        flux=SingleDistrictStub(district_name)
    )
    print(seir_parameters)