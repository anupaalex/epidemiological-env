# Epidemiological Environment 
A custom OpenAI gym environment for optimising interventions during an epidemic.

After downloading the project files, go to the EpidemicInterventionEnv folder 

Make sure you have python 3.7 installed

pip install -r requirements.txt 

run pip install --editable . 

These are the various scenarios which can be set in the init function of the environment( ToDo: Pass it during initialisation)

age_based =True

capacity_based = False  

increase_bed = False

reduce_compliance = False

economic_budget_based = False

increase_peak_days = False

reduce_total_infected = True
 
Example for running this environment is in main.py which has code for using it for a custom made DQN and also DQN and PPO2 from stable baselines

To optimise hyperparameters for PPO and DQN, rl-baselines-zoo can be used as follows:

python train.py --algo ppo2 --env epidemiological-env-v0  --n-trials 1000 -n 50000  --sampler random --pruner median -optimize

python train.py --algo dqn --env epidemiological-env-v0  --n-trials 1000 -n 50000  --sampler random --pruner median -optimize