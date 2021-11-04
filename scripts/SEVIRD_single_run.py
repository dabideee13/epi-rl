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

import argparse
from pathlib import Path

from gym.envs.registration import register, make
import numpy as np
import pandas as pd
from stable_baselines import PPO2

import epcontrol.census.Flux as flux
# FIXME: Granularity, when different `Granularity` is imported, there will be a TypeError
from epcontrol.SEVIRD_environment import Granularity, Outcome
from epcontrol.UK_RL_school_weekly import run_model
from epcontrol.SEVIRD_model import SEVIRDModel
from epcontrol.wrappers import NormalizedObservationWrapper, NormalizedRewardWrapper

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--R0", type=float, required=True)
parser.add_argument("--district_name", required=True)
parser.add_argument("--budget_in_weeks", type=int, required=True)
parser.add_argument("--census", type=Path, required=True)
parser.add_argument("--runs", type=int, required=True)
parser.add_argument("--outcome", choices=["ar", "pd"], required=True)
parser.add_argument("--path", type=Path, required=True)

args = parser.parse_args()

if args.outcome == 'ar':
    outcome = Outcome.ATTACK_RATE
elif args.outcome == 'pd':
    outcome = Outcome.PEAK_DAY
else:
    raise ValueError("Wrong outcome")


def evaluate(env, model, num_steps):
    _model = env.unwrapped._model
    obs = env.reset()
    sus_before = _model.total_susceptibles()
    for _ in range(num_steps):
        action, _states = model.predict(obs)
        obs, _, _, _ = env.step(action)
    sus_after = _model.total_susceptibles()
    attack_rate = 1.0 - (sus_after / sus_before)
    peak_day = _model.peak_day(env.unwrapped.infected_history)
    return (attack_rate, peak_day, env.unwrapped.infected_history)


n_weeks = 43
granularity = Granularity.WEEK
flux = flux.SingleDistrictStub(args.district_name)
grouped_census = pd.read_csv(args.census, index_col=0)
grouped_census = grouped_census.filter(items=[args.district_name], axis=0)

delta = .5
rho = 1
gamma = (1 / 1.8)
eta = 0.5
c_v = 0.3
alpha = 0.32
zeta = 0.333
mu = np.log(args.R0) * .6
sde = True

register(id="SEVIRDsingle-v0",
         entry_point="epcontrol.SEVIRD_environment:SEVIRDEnvironment",
         max_episode_steps=n_weeks * (7 if granularity == Granularity.DAY else 1),
         kwargs=dict(grouped_census=grouped_census,
                     flux=flux,
                     r0=args.R0,
                     n_weeks=(n_weeks * 2),
                     rho=rho,
                     gamma=gamma,
                     delta=delta,
                     outcome=outcome,
                     step_granularity=granularity,
                     model_seed=args.district_name,
                     eta=eta,
                     c_v=c_v,
                     alpha=alpha,
                     zeta=zeta,
                     mu=mu,
                     sde=sde,
                     budget_per_district_in_weeks=args.budget_in_weeks))

env = make("SEVIRDsingle-v0")
env = NormalizedObservationWrapper(env)
env = NormalizedRewardWrapper(env)

no_closures = [1] * n_weeks
weekends = False
district_names = grouped_census.index.to_list()

baseline_model = SEVIRDModel(delta, args.R0, rho, gamma, district_names, grouped_census, flux, mu, sde, eta, c_v, alpha, zeta)
# TODO: Debug run_model
(baseline_pd, baseline_ar, _) = run_model(baseline_model, n_weeks, weekends, args.district_name, no_closures)

# TODO: Why is it that PPO model is not loaded in the run_model and only here?
model = PPO2.load(args.path / "params.zip")
print(args.outcome + "-improvement")
for run in range(args.runs):
    # TODO: Know the difference between baseline_ar and attack_rate and why
    # TODO: What's the purpose of evaluate?
    (attack_rate, peak_day, inf) = evaluate(env, model, n_weeks)
    if args.outcome == "ar":
        print(baseline_ar - attack_rate)
    elif args.outcome == "pd":
        print(peak_day - baseline_pd)

env.close()
