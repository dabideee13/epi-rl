# deep reinforcement learning for large-scale epidemic control
# copyright (c) 2020  pieter libin, arno moonens, fabian perez-sanjines.

# this program is free software; you can redistribute it and/or modify
# it under the terms of the gnu general public license as published by
# the free software foundation; either version 2 of the license, or
# (at your option) any later version.

# this program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to pieter.libin@ai.vub.ac.be or arno.moonens@vub.be.

from pathlib import Path
from typing import List, Sequence, Callable, Dict
from numba import njit
import numpy as np
from scipy.signal import savgol_filter

import epcontrol.compartments.AgeSEIR as AgeSEIR
from epcontrol.compartments.contacts.Eames2012 import (
    Eames2012,
    age_ranges,
    read_contact_matrix,
    compute_beta,
    make_reciprocal,
)
from epcontrol.utils import find_peaks


class UK:
    def __init__(
        self,
        delta: float,
        R0: float,
        rho: float,
        gamma: float,
        district_names: List[str],
        grouped_census,
        flux,
        mu: float,
        sde: bool,
        contact_matrix: Callable[[str], Dict[str, np.ndarray]]
    ):
        self.delta = delta
        self.district_names = district_names
        self.n_districts = len(self.district_names)

        self.ag = Eames2012
        self.n_age_groups = len(age_ranges(self.ag))

        # FIXME: Fix hard-coded contact matrices
        # cm_path = Path(__file__).resolve().parent.parent / "data/contacts"
        # cm_school = read_contact_matrix(cm_path / "conversational_school.csv")
        # cm_no_school = read_contact_matrix(cm_path / "conversational_no_school.csv")

        # cm_school = contact_matrix('conversational_school')
        # cm_no_school = contact_matrix('conversational_no_school')

        # cm_school = contact_matrix('open_cm')
        # cm_no_school = contact_matrix('closed_cm')

        cm_open = contact_matrix('open_cm')
        cm_close = contact_matrix('closed_cm')
        cm_ecq = contact_matrix('open_cm')
        cm_mecq = contact_matrix('open_cm')
        cm_gcq = contact_matrix('open_cm')
        cm_mgcq = contact_matrix('open_cm')

        # TODO: add contact matrices here
        self.cms_open = np.empty(
            (self.n_districts, self.n_age_groups, self.n_age_groups), dtype=np.float32
        )
        self.cms_close = np.empty(
            (self.n_districts, self.n_age_groups, self.n_age_groups), dtype=np.float32
        )
        self.cms_ecq = np.empty(
            (self.n_districts, self.n_age_groups, self.n_age_groups), dtype=np.float32
        )
        self.cms_mecq = np.empty(
            (self.n_districts, self.n_age_groups, self.n_age_groups), dtype=np.float32
        )
        self.cms_gcq = np.empty(
            (self.n_districts, self.n_age_groups, self.n_age_groups), dtype=np.float32
        )
        self.cms_mgcq = np.empty(
            (self.n_districts, self.n_age_groups, self.n_age_groups), dtype=np.float32
        )

        for i, (_, row) in enumerate(grouped_census.iterrows()):
            self.cms_open[i] = make_reciprocal(cm_open, row)
            self.cms_close[i] = make_reciprocal(cm_close, row)
            self.cms_ecq[i] = make_reciprocal(cm_ecq, row)
            self.cms_mecq[i] = make_reciprocal(cm_mecq, row)
            self.cms_gcq[i] = make_reciprocal(cm_gcq, row)
            self.cms_mgcq[i] = make_reciprocal(cm_mgcq, row)

        self.mu = mu

        self.sde = sde

        self.rho = rho
        self.gamma = gamma
        self.R0 = R0
        self.betas = np.asarray(
            [compute_beta(R0, gamma, cm_open) for cm_open in self.cms_open],
            np.float32,
        )

        self.flux = flux
        self.sde_steps = int(1 / delta)

        # make sure the districts are the same than the ones provide in the census
        assert self.district_names == list(grouped_census.index)
        assert set(self.district_names) == set(flux.district_names)

        grouped_census = grouped_census.reindex(flux.district_names)
        self.district_names = list(flux.district_names)

        assert list(grouped_census.index) == list(flux.district_names)

        n_compartments = 4
        self.seir_state = np.zeros(
            (self.n_districts, n_compartments, self.n_age_groups), dtype=np.float32
        )

        self.districts_school_states = np.full((self.n_districts,), 1, np.uint8)
        self.districts_sparked = np.zeros((self.n_districts,), dtype=np.uint8)
        self.districts_lambda = np.zeros((self.n_districts,), dtype=np.float32)
        self.districts_import_threshold = -np.log(np.random.rand(self.n_districts))

        # Assumes that district_names and indices in grouped_census are in the same order
        self.districts_susceptibles = grouped_census.to_numpy(dtype=np.float32)
        self.adult_susceptibles = self.districts_susceptibles[:, Eames2012.Adults.value]

        self.seir_state[:, AgeSEIR.Compartment.S.value, :] = self.districts_susceptibles
        self.seir_state[:, AgeSEIR.Compartment.E.value, :] = (
            self.seir_state[:, AgeSEIR.Compartment.S.value, :] * 10 ** -6
        )

    def district_idx(self, district_name: str) -> int:
        return self.district_names.index(district_name)

    def reset(self) -> None:
        self.seir_state[:, AgeSEIR.Compartment.S.value, :] = self.districts_susceptibles
        self.seir_state[:, AgeSEIR.Compartment.E.value, :] = (
            self.seir_state[:, AgeSEIR.Compartment.S.value, :] * 10 ** -6
        )
        self.seir_state[
            :, [AgeSEIR.Compartment.I.value, AgeSEIR.Compartment.R.value], :
        ] = 0
        self.districts_school_states.fill(1)
        self.districts_sparked.fill(0)
        self.districts_lambda.fill(0)
        self.districts_import_threshold = -np.log(np.random.rand(self.n_districts))

    def seed(self, region: str) -> None:
        self.districts_sparked[self.district_names.index(region)] = 1

    def total_infected(self) -> float:
        return np.sum(self.seir_state[:, AgeSEIR.Compartment.I.value, :])

    def total_susceptibles(self) -> float:
        return np.sum(self.seir_state[:, AgeSEIR.Compartment.S.value, :])

    def total_susceptibles_district(self, district_idx: int) -> float:
        return np.sum(self.seir_state[district_idx, AgeSEIR.Compartment.S.value, :])

    def peak_day(self, infected_history):
        inf_savgol = savgol_filter(infected_history, 9, 2, deriv=1)
        inf_savgol_2 = savgol_filter(infected_history, 9, 2, deriv=2)
        inf_normalized = infected_history / np.max(infected_history)
        threshold = 0.2
        peaks = find_peaks(inf_normalized, inf_savgol, inf_savgol_2, threshold)
        if len(peaks) == 0:
            return 0
        else:
            return peaks[0]  # /(float)(self.n_weeks * 7)

    def step(self, t: int, school_states: Sequence[int]):
        sparked_districts_indices = self.districts_sparked.nonzero()[0]
        if not self.sde:
            W = np.zeros(
                (self.sde_steps, len(sparked_districts_indices), 3, self.n_age_groups)
            )
        else:
            W = np.random.standard_normal(
                (self.sde_steps, len(sparked_districts_indices), 3, self.n_age_groups)
            )
        return _step(
            school_states,
            self.seir_state,
            self.cms_open,
            self.cms_close,
            self.cms_ecq,
            self.cms_mecq,
            self.cms_gcq,
            self.cms_mgcq,
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
            self.mu,
        )


# @njit(cache=True)
def _step(
    school_states,
    seir_state,
    cms_open,
    cms_close,
    cms_ecq,
    cms_mecq,
    cms_gcq,
    cms_mgcq,
    adult_susceptibles,
    districts_school_states,
    districts_sparked,
    betas,
    rho,
    gamma,
    delta,
    sde_steps,
    flux_tij,
    districts_lambda,
    districts_import_threshold,
    sparked_districts_indices,
    W,
    mu,
):
    S_idx = 0
    E_idx = 1
    I_idx = 2
    R_idx = 3
    adults_idx = 2

    # the Infected/Adults column in the SEIR state
    I_comm = seir_state[:, I_idx, adults_idx]
    I_comm_prop = I_comm / adult_susceptibles

    # Overwrite school states with new ones
    districts_school_states[:] = school_states

    # for all sparked districts,
    # keep the CM (sparked_districts_cms), which depends on whether the schools are open
    sparked_open_cms_indices = (
        districts_school_states[sparked_districts_indices] == 1
    ).nonzero()[0]
    sparked_closed_cms_indices = (
        districts_school_states[sparked_districts_indices] == 0
    ).nonzero()[0]
    sparked_ecq_cms_indices = (
        districts_school_states[sparked_districts_indices] == 2
    ).nonzero()[0]
    sparked_mecq_cms_indices = (
        districts_school_states[sparked_districts_indices] == 3
    ).nonzero()[0]
    sparked_gcq_cms_indices = (
        districts_school_states[sparked_districts_indices] == 4
    ).nonzero()[0]
    sparked_mgcq_cms_indices = (
        districts_school_states[sparked_districts_indices] == 5
    ).nonzero()[0]

    sparked_districts_cms = np.empty(
        (len(sparked_districts_indices),) + cms_open[0].shape, dtype=np.float32
    )

    sparked_districts_cms[sparked_open_cms_indices] = cms_open[
        sparked_open_cms_indices
    ]
    sparked_districts_cms[sparked_closed_cms_indices] = cms_close[
        sparked_closed_cms_indices
    ]
    sparked_districts_cms[sparked_ecq_cms_indices] = cms_ecq[
        sparked_closed_cms_indices
    ]
    sparked_districts_cms[sparked_mecq_cms_indices] = cms_mecq[
        sparked_closed_cms_indices
    ]
    sparked_districts_cms[sparked_gcq_cms_indices] = cms_gcq[
        sparked_closed_cms_indices
    ]
    sparked_districts_cms[sparked_mgcq_cms_indices] = cms_mgcq[
        sparked_closed_cms_indices
    ]

    n_sparked = len(sparked_districts_indices)
    # NOTE: 4 here is number of age groups
    weighted_inf_sums = np.empty((len(sparked_districts_indices), 4), dtype=np.float32)
    for sde_step in range(sde_steps):
        # matrix, with for each sparked district (row) the relative infected for all the age groups (column)
        sparked_districts_ags_relative_I = seir_state[
            sparked_districts_indices, I_idx
        ] / np.sum(seir_state[sparked_districts_indices], axis=1)

        for idx in range(n_sparked):
            weighted_inf_sums[idx] = np.dot(
                sparked_districts_cms[idx], sparked_districts_ags_relative_I[idx]
            )

        Ss = seir_state[sparked_districts_indices, S_idx]
        Es = seir_state[sparked_districts_indices, E_idx]
        Is = seir_state[sparked_districts_indices, I_idx]
        Rs = seir_state[sparked_districts_indices, R_idx]

        SEs = (
            np.expand_dims(betas[sparked_districts_indices], axis=1)
            * weighted_inf_sums
            * Ss
        )
        EIs = rho * Es
        IRs = gamma * Is

        d_SEs = delta * SEs + np.sqrt(delta * SEs) * W[sde_step][:, 0]
        d_EIs = delta * EIs + np.sqrt(delta * EIs) * W[sde_step][:, 1]
        d_IRs = delta * IRs + np.sqrt(delta * IRs) * W[sde_step][:, 2]

        seir_state[sparked_districts_indices, AgeSEIR.Compartment.S.value] = Ss - d_SEs
        seir_state[sparked_districts_indices, AgeSEIR.Compartment.E.value] = (
            Es + d_SEs - d_EIs
        )
        seir_state[sparked_districts_indices, AgeSEIR.Compartment.I.value] = (
            Is + d_EIs - d_IRs
        )
        seir_state[sparked_districts_indices, AgeSEIR.Compartment.R.value] = Rs + d_IRs

        seir_state[sparked_districts_indices] = np.maximum(
            seir_state[sparked_districts_indices], 0
        )

    # For a district k,
    # we assume a non-homogenous Poisson process with lambda_k(t),
    # where lambda_k(t) is the import potential at time t.
    # Where lambda_k(t)=sum(I_comm(t)*flux_k)*beta,
    # where flux_i is the mobility flux towards district k,
    # i.e., flux_k = T_i,k, the k-th column of Ti,j.
    non_sparked_districts_indices = np.where(districts_sparked == 0)[0]
    if non_sparked_districts_indices.size > 0:
        M_aas = cms_open[non_sparked_districts_indices, adults_idx, adults_idx]
        flux_k = flux_tij[:, non_sparked_districts_indices]
        S_a_k = adult_susceptibles[non_sparked_districts_indices]
        lambda_k = (
            np.dot(flux_k.T, I_comm_prop)
            * betas[non_sparked_districts_indices]
            * M_aas
            * np.power(S_a_k, mu)
        )
        # lambda_k is added to a cumulative Lambda_k,
        # which is used to determine when an import event should take place,
        # based on the threshold as specified in the Cinlar algorithm
        districts_lambda[non_sparked_districts_indices] += lambda_k
        districts_sparked[non_sparked_districts_indices] = (
            districts_lambda[non_sparked_districts_indices]
            >= districts_import_threshold[non_sparked_districts_indices]
        )
