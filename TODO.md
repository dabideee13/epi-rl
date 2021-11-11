General:<br>
- TODO: Study ODE to SDE transformation to understand `W` in step method<br>
- TODO: Prepare census data for Iligan City
- TODO: Find mobility data for multi-agents
- TODO: Use **kwargs in the parameters
- FIXME: Add versions in requirements.txt

Rewards:<br>
- TODO: Make a reward function that involves vaccination<br>
- TODO: What is the meaning when the reward function returns a negative value?<br>
- TODO: Is it possible, or even logical, to get the attack rate for every timestep?<br>

Model:<br>
- TODO: Estimate values for model parameters<br>

Model parameters:<br>
- FIXME: Determine the "right" values for latency rate. Is it valid to have a latency rate from `Exposed` to `Vaccinated` compartment?

Actions:<br>
- TODO: Generate contact matrices for 6 possible actions (ECQ, MECQ, GCQ, MGCQ, open, close)

Observations:<br>
- TODO: Add a decorator for unnormalizing observations
- TODO: Understand the purpose of budget

Simulation:<br>
- TODO: Try different scenarios