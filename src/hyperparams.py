from stable_baselines3 import DQN
from agents.DDQN import DoubleDQN

def create_dqn(env):
    """
    This function creates a DQN model with predefined parameters using the Stable Baselines3 library.
    The environment for the model is created by calling the create_env function from the environment module.

    Returns:
        DQN: A DQN model with predefined parameters.
        env: An instance of the LunarLander environment.
    """
    
    # env = create_env()
    
    dqn_model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        train_freq=4,
        gradient_steps=-1,
        gamma=0.99,
        exploration_fraction=0.12,
        exploration_final_eps=0.1,
        target_update_interval=250,
        learning_starts=0,
        buffer_size=50000,
        batch_size=128,
        learning_rate=6.3e-4,
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=2,
    )
    return dqn_model

def create_double_q(env):
    """
        This function creates and returns a DoubleDQN model with specified hyperparameters.

        Args:
            env (gym.Env): The environment for the agent to interact with.

        Returns:
            DoubleDQN: The initialized DoubleDQN model.
    """
    
    ddqn_model = DoubleDQN(
        "MlpPolicy",
        env,
        verbose=1,
        train_freq=4,
        gradient_steps=-1,
        gamma=0.99,
        exploration_fraction=0.12,
        exploration_final_eps=0.1,
        target_update_interval=250,
        learning_starts=0,
        buffer_size=50000,
        batch_size=128,
        learning_rate=6.3e-4,
        policy_kwargs=dict(net_arch=[256, 256]),
        # tensorboard_log=tensorboard_log,
        seed=2,
    )
    return ddqn_model
