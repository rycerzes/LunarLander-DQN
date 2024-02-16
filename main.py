from src.cuda_test import cuda_availability as cuda_chk
from src.hyperparams import create_dqn,create_double_q
from src.environment import create_env
from stable_baselines3.common.evaluation import evaluate_policy
from src.actions import simple
from src.q_val import get_q_values
from src.runner import run_episode
from src.monitor import MonitorQValueCallback
import numpy as np
import matplotlib.pyplot as plt
import os

cuda_chk()
env = create_env()
dqn_model = create_dqn(env)
ddqn_model = create_double_q(env)

mean_reward, std_reward = evaluate_policy(
    dqn_model,
    dqn_model.get_env(),
    deterministic=True,
    n_eval_episodes=20,
)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# dqn_model.learn(int(1e5), log_interval=10)
monitor_ddqn_value_cb = MonitorQValueCallback()
monitor_dqn_value_cb = MonitorQValueCallback()

dqn_model.learn(int(1e5), log_interval=10, callback=monitor_dqn_value_cb)
ddqn_model.learn(int(1e5), log_interval=10, callback=monitor_ddqn_value_cb)

mean_reward, std_reward = evaluate_policy(dqn_model, dqn_model.get_env(), deterministic=True, n_eval_episodes=20)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

obs, _ = env.reset()
initial_state = obs
print(dqn_model)
print(env)
print("obs.shape:", obs.shape)
print ("env.action_space:", env.action_space)
print("observation_space_shape:",env.observation_space.shape)
print(env.step,"\n")
print(env.metadata,"\n")
print(env.observation_space,"\n")
obs = obs.flatten()
print(obs.shape)
print(obs)

q_values = get_q_values(dqn_model, initial_state)
print(q_values)
q_value_nothing = q_values[0]
q_value_left = q_values[1]
q_value_main = q_values[2]
q_value_right=q_values[3]

print(f"Q-value of the initial state left={q_value_left:.2f} nothing={q_value_nothing:.2f} right={q_value_right:.2f}")
action = np.argmax(q_values)
print(f"Action taken by the greedy policy in the initial state: {simple[action]}")

initial_q_value = q_values.max()
print(initial_q_value)

reward = run_episode()
print(f"Sum of discounted rewards: {reward:.2f}")

plt.figure(figsize=(6, 3), dpi=150)
plt.title("Evolution of max q-value for start states over time")
plt.plot(monitor_dqn_value_cb.timesteps, monitor_dqn_value_cb.max_q_values, label="DQN", color="pink")
plt.plot(monitor_ddqn_value_cb.timesteps, monitor_ddqn_value_cb.max_q_values, label="DDQN", color="purple")
plt.legend()

# Create the graphs folder if it doesn't exist
graphs_folder = "graphs"
os.makedirs(graphs_folder, exist_ok=True)

# Save the graph in the graphs folder
graph_filepath = os.path.join(graphs_folder, "q_value_evolution.png")
plt.savefig(graph_filepath)
