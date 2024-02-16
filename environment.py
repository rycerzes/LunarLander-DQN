import gymnasium as gym

mode=["rgb_array", "human", "headless"]

def create_env():
    """
    Create and return an instance of the LunarLander environment.

    Returns:
        gym.Env: An instance of the LunarLander environment.
    """
    env = gym.make("LunarLander-v2", render_mode=mode[0])
    return env

if __name__ == "__main__":
    env = create_env()