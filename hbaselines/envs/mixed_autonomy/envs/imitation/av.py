"""Imitation variant of the environments in mixed_autonomy/envs/av.py.

All environments are included a query_action method that computes the expert
actions. The model used for the expert is specified under the "expert_model"
attribute in env_params.additional_params.
"""
from flow.controllers import IDMController
from flow.core.params import SumoCarFollowingParams

from hbaselines.envs.mixed_autonomy.envs.av import AVEnv
from hbaselines.envs.mixed_autonomy.envs.av import AVClosedEnv
from hbaselines.envs.mixed_autonomy.envs.av import AVOpenEnv


IMITATION_PARAMS = {
    # the model that the expert policy should adopt
    "expert_model": (IDMController, {
        "a": 0.3,
        "b": 2.0,
    }),
}


class AVImitationEnv(AVEnv):
    """Imitation variant of AVEnv."""

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """See parent class."""
        for p in IMITATION_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError('Env parameter "{}" not supplied'.format(p))

        self._expert_models = {}

        super(AVImitationEnv, self).__init__(
            env_params=env_params,
            sim_params=sim_params,
            network=network,
            simulator=simulator,
        )

    def query_expert(self, obs):
        """Query the expert compute the expert actions.

        Parameters
        ----------
        obs : array_like
            the environmental observation

        Returns
        -------
        array_like
            the expert action
        """
        del obs  # unused

        expert_actions = []
        for veh_id in self.rl_ids():
            # Add experts that are not currently available.
            model, params = self.env_params.additional_params["expert_model"]
            self._expert_models[veh_id] = model(
                veh_id,
                car_following_params=SumoCarFollowingParams(min_gap=0.5),
                **params)

            # Compute the expert action.
            expert_actions.append(self._expert_models[veh_id].get_action(self))

        return expert_actions


class AVClosedImitationEnv(AVClosedEnv):
    """Imitation variant of AVClosedEnv."""

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """See parent class."""
        for p in IMITATION_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError('Env parameter "{}" not supplied'.format(p))

        self._expert_models = {}

        super(AVClosedImitationEnv, self).__init__(
            env_params=env_params,
            sim_params=sim_params,
            network=network,
            simulator=simulator,
        )

    def query_expert(self, obs):
        """Query the expert compute the expert actions.

        Parameters
        ----------
        obs : array_like
            the environmental observation

        Returns
        -------
        array_like
            the expert action
        """
        del obs  # unused

        expert_actions = []
        for veh_id in self.rl_ids():
            # Add experts that are not currently available.
            model, params = self.env_params.additional_params["expert_model"]
            self._expert_models[veh_id] = model(
                veh_id,
                car_following_params=SumoCarFollowingParams(min_gap=0.5),
                **params)

            # Compute the expert action.
            expert_actions.append(self._expert_models[veh_id].get_action(self))

        return expert_actions


class AVOpenImitationEnv(AVOpenEnv):
    """Imitation variant of AVOpenEnv."""

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """See parent class."""
        for p in IMITATION_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError('Env parameter "{}" not supplied'.format(p))

        self._expert_models = {}

        super(AVOpenImitationEnv, self).__init__(
            env_params=env_params,
            sim_params=sim_params,
            network=network,
            simulator=simulator,
        )

    def query_expert(self, obs):
        """Query the expert compute the expert actions.

        Parameters
        ----------
        obs : array_like
            the environmental observation

        Returns
        -------
        array_like
            the expert action
        """
        del obs  # unused

        # Remove experts that are no longer in the environment.
        for veh_id in self._expert_models.keys():
            if veh_id not in self.rl_ids():
                del self._expert_models[veh_id]

        expert_actions = []
        for veh_id in self.rl_ids():
            # Add experts that are not currently available.
            model, params = self.env_params.additional_params["expert_model"]
            self._expert_models[veh_id] = model(
                veh_id,
                car_following_params=SumoCarFollowingParams(min_gap=0.5),
                **params)

            # Compute the expert action.
            expert_actions.append(self._expert_models[veh_id].get_action(self))

        # Pad the actions for the non-existent vehicles with zeroes.
        for _ in range(self.action_space.shape[0] - len(expert_actions)):
            expert_actions.append(0)

        return expert_actions
