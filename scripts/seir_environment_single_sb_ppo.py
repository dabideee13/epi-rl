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
import datetime
import os
from pathlib import Path

import tensorflow as tf
from gym.envs.registration import register, make
import pandas as pd
from stable_baselines import logger
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

import epcontrol.census.Flux as flux
from epcontrol.seir_environment import Granularity
from epcontrol.seir_environment import Outcome
from epcontrol.wrappers import NormalizedObservationWrapper, NormalizedRewardWrapper
from epcontrol.contact_matrix import cm_getter

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--id", type=str, required=True)
parser.add_argument("--outcome", choices=["ar", "pd"], required=True)
parser.add_argument("--district_name", required=True)
parser.add_argument("--budget_in_weeks", type=int, required=True)
parser.add_argument("--census", type=Path, required=True)
parser.add_argument("--R0", type=float, required=True)
parser.add_argument("--monitor_path",
                    type=str,
                    default=f"/tmp/SEIR-PPO-{datetime.datetime.now():%Y-%m-%d-%H-%M-%S-%f}/")
parser.add_argument("--entropy_coef", type=float, default=0.01)
parser.add_argument("--n_hidden_layers", type=int, default=0)
parser.add_argument("--n_hidden_units", type=int, default=0)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--n_epochs", type=int, default=4)
parser.add_argument("--n_minibatches", type=int, default=4)
parser.add_argument("--n_steps", type=int, default=128)
parser.add_argument("--max_grad_norm", type=float, default=None)
parser.add_argument("--clip_range", type=float, default=0.2)

parser.add_argument("--total_timesteps", type=int, required=True)
parser.add_argument("--latency_rate", type=float, default=1)

args = parser.parse_args()

ncpu = 1
config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=ncpu,
                        inter_op_parallelism_threads=ncpu)
tf.Session(config=config).__enter__()

n_weeks = 43
granularity = Granularity.WEEK

flux = flux.SingleDistrictStub(args.district_name)
grouped_census = pd.read_csv(args.census, index_col=0)

grouped_census = grouped_census.filter(items=[args.district_name], axis=0)

outcome = None
if args.outcome == "ar":
    outcome = Outcome.ATTACK_RATE
elif args.outcome == "pd":
    outcome = Outcome.PEAK_DAY

cm_path = Path.joinpath(Path.cwd(), 'data/contacts/contact1')
contact_matrix = cm_getter(cm_path)

register(id=f"SEIRsingle-v{args.id}",
         entry_point="epcontrol.seir_environment:SEIREnvironment",
         max_episode_steps=n_weeks * (7 if granularity == Granularity.DAY else 1),
         kwargs=dict(contact_matrix=contact_matrix,
                     grouped_census=grouped_census,
                     flux=flux,
                     r0=args.R0,
                     n_weeks=n_weeks,
                     step_granularity=granularity,
                     outcome=outcome,
                     model_seed=args.district_name,
                     budget_per_district_in_weeks=args.budget_in_weeks,
                     rho=args.latency_rate))

env = make(f"SEIRsingle-v{args.id}")
env = NormalizedObservationWrapper(env)
if args.outcome == "ar":
    env = NormalizedRewardWrapper(env)
logger.configure(folder=args.monitor_path, format_strs=["csv"])

env = DummyVecEnv([lambda: env])

print(f"tensorboard --logdir {args.monitor_path}")

layers = [args.n_hidden_units]*args.n_hidden_layers

model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=args.monitor_path,
             ent_coef=args.entropy_coef, learning_rate=args.learning_rate,
             noptepochs=args.n_epochs, nminibatches=args.n_minibatches,
             n_steps=args.n_steps,
             policy_kwargs={"layers": layers})
model.learn(total_timesteps=args.total_timesteps)
model.save(os.path.join(args.monitor_path, "params"))
env.close()
