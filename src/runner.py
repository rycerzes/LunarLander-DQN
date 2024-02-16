from actions import simple
from stable_baselines3 import DQN
from environment import create_env
from hyperparams import create_dqn
from q_val import get_q_values

def run_episode():
    env = create_env()
    dqn_model = create_dqn(env)
    obs, _ = env.reset()
    # obs = obs.flatten()
    
    episode_rewards = []
    done = False
    i = 0

    while not done:
        i += 1

        # Retrieve q-value
        # q_values = get_q_values(dqn_model, obs)

        # Take greedy-action
        action, _ = dqn_model.predict(obs, deterministic=True)

        # print(f"Q-value of the current state \nnothing={q_values[0]:.2f} \nleft={q_values[1]:.2f} \nmain={q_values[2]:.2f} \nright={q_values[3]}")
        # print(f"Action: {simple[action]}")

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        episode_rewards.append(reward)

    sum_discounted_rewards = 0
    for i, reward in enumerate(reversed(episode_rewards)):
        sum_discounted_rewards += reward * (dqn_model.gamma ** i)

    return sum_discounted_rewards

