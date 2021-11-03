from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Sequence, Any
from enum import Enum

import numpy as np
from scipy.signal import savgol_filter
from numba import njit

import epcontrol.compartments.AgeSEIR as AgeSEIR
from epcontrol.compartments.contacts.Eames2012 import (
    Eames2012,
    age_ranges,
    read_contact_matrix,
    compute_beta,
    make_reciprocal
)
from epcontrol.utils import find_peaks


class IModel(ABC):

    @abstractmethod
    def initialize_state(self) -> None:
        pass

    @abstractmethod
    def step(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def district_idx(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def seed(self) -> None:
        pass

    @abstractmethod
    def total_infected(self) -> None:
        pass

    @abstractmethod
    def total_susceptibles(self) -> None:
        pass

    @abstractmethod
    def total_susceptibles_district(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def peak_day(self) -> None:
        pass


class BaseModel(IModel):
    def __init__(self,
                 delta: float,
                 R0: float,
                 rho: float,
                 gamma: float,
                 district_names: List[str],
                 grouped_census,
                 flux,
                 mu: float,
                 sde: bool) -> None:
        self.delta = delta
        self.district_names = district_names
        self.n_districts = len(self.district_names)

        self.ag = Eames2012
        self.n_age_groups = len(age_ranges(self.ag))
        cm_path = Path(__file__).resolve().parent.parent / "data/contacts"
        cm_school = read_contact_matrix(cm_path / "conversational_school.csv")
        cm_no_school = read_contact_matrix(cm_path / "conversational_no_school.csv")

        self.cms_school = np.empty((self.n_districts, self.n_age_groups, self.n_age_groups), dtype=np.float32)
        self.cms_no_school = np.empty((self.n_districts, self.n_age_groups, self.n_age_groups), dtype=np.float32)
        for i, (_, row) in enumerate(grouped_census.iterrows()):
            self.cms_school[i] = make_reciprocal(cm_school, row)
            self.cms_no_school[i] = make_reciprocal(cm_no_school, row)

        self.mu = mu

        self.sde = sde

        self.rho = rho
        self.gamma = gamma
        self.R0 = R0
        self.betas = np.asarray([compute_beta(R0, gamma, cm_school) for cm_school in self.cms_school], np.float32)

        self.flux = flux
        self.sde_steps = int(1 / delta)

        # make sure the districts are the same than the ones provide in the census
        assert self.district_names == list(grouped_census.index)
        assert set(self.district_names) == set(flux.district_names)

        grouped_census = grouped_census.reindex(flux.district_names)
        self.district_names = list(flux.district_names)

        assert list(grouped_census.index) == list(flux.district_names)

        self.districts_school_states = np.full((self.n_districts,), 1, np.uint8)
        self.districts_sparked = np.zeros((self.n_districts,), dtype=np.uint8)
        self.districts_lambda = np.zeros((self.n_districts,), dtype=np.float32)
        self.districts_import_threshold = -np.log(np.random.rand(self.n_districts))

        # Assumes that district_names and indices in grouped_census are in the same order
        self.districts_susceptibles = grouped_census.to_numpy(dtype=np.float32)
        self.adult_susceptibles = self.districts_susceptibles[:, Eames2012.Adults.value]

        self.seir_state = self.initialize_state()

    @abstractmethod
    def initialize_state(self) -> np.ndarray:
        pass

    def district_idx(self, district_name: str) -> int:
        return self.district_names.index(district_name)

    def seed(self, region: str) -> None:
        self.districts_sparked[self.district_names.index(region)] = 1

    @abstractmethod
    def total_infected(self) -> None:
        pass

    @abstractmethod
    def total_susceptibles(self) -> None:
        pass

    @abstractmethod
    def total_susceptibles_district(self, *args: Any, **kwargs: Any) -> None:
        pass

    def peak_day(self, infected_history):
        inf_savgol = savgol_filter(infected_history, 9, 2, deriv=1)
        inf_savgol_2 = savgol_filter(infected_history, 9, 2, deriv=2)
        inf_normalized = infected_history / np.max(infected_history)
        threshold = .2
        peaks = find_peaks(inf_normalized, inf_savgol, inf_savgol_2, threshold)
        if len(peaks) == 0:
            return 0
        else:
            return peaks[0]

    @abstractmethod
    def reset(self) -> None:
        pass

    def step(self, t: int, school_states: Sequence[int]):
        sparked_districts_indices = self.districts_sparked.nonzero()[0]
        if not self.sde:
            W = np.zeros((self.sde_steps, len(sparked_districts_indices), 3, self.n_age_groups))
        else:
            W = np.random.standard_normal((self.sde_steps, len(sparked_districts_indices), 3, self.n_age_groups))
        return self.stepper(school_states,
                     self.seir_state,
                     self.cms_school,
                     self.cms_no_school,
                     self.adult_susceptibles,
                     self.districts_school_states,
                     self.districts_sparked,
                     self.betas,
                     self.rho,
                     self.gamma,
                     self.delta,
                     self.sde_steps,
                     self.flux.Tij,
                     self.districts_lambda,
                     self.districts_import_threshold,
                     sparked_districts_indices,
                     W,
                     self.mu)

    @abstractmethod
    @njit(cache=True)
    def stepper(self, *args: Any, **kwargs: Any) -> None:
        pass
