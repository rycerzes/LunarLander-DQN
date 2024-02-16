import numpy as np
import torch as th
from stable_baselines3 import DQN

def get_q_values(model: DQN, obs: np.ndarray) -> np.ndarray:
    """
    Retrieve Q-values for a given observation.

    :param model: a DQN model
    :param obs: a single observation
    :return: the associated q-values for the given observation
    """
    assert model.get_env().observation_space.contains(obs), f"Invalid observation of shape {obs.shape}: {obs}"
    obs_tensor = th.tensor(obs).float().unsqueeze(0).to(model.device)
    with th.no_grad():
        q_values = model.q_net.forward(obs_tensor).cpu().numpy().squeeze()

    assert isinstance(q_values, np.ndarray), "The returned q_values is not a numpy array"
    assert q_values.shape == (4,), f"Wrong shape: (4,) was expected but got {q_values.shape}"

    return q_values

