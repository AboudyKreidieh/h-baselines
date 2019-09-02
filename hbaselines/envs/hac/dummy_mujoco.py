"""Utility objects and methods for testing purposes.

The contents of this file are meant to allow the Travis tests to pass in the
absence of installed (licensed) version of mujoco_py. This is needed primarily
for the environment tests.

Note that, since various mujoco_py objects in the environment class are being
replaced with dummy objects that are specifically designed to allow the tests
to pass, the functionality of some of the environments can only truly be
validated if they are passed locally after mujoco_py has been installed.
"""
import numpy as np


def load_model_from_path(path):
    if "pendulum" in path:
        model_name = "pendulum"
    elif "ur5" in path:
        model_name = "ur5"
    else:
        model_name = "unknown"

    return model_name


class MjViewer(object):

    def __init__(self, *_):
        pass


class MjSim(object):

    def __init__(self, model_name):
        self.data = DummyData(model_name)
        self.model = DummyData(model_name)

    def step(self, *_):
        pass


class DummyModel(object):

    def __init__(self, model_name):
        pass


class DummyData(object):

    def __init__(self, model_name):
        # variables that need to be defined for any mujoco data object
        self.qpos = None
        self.qvel = None
        self.ctrl = None

        # update the aforementioned variables if the model is known
        if model_name == "pendulum":
            self.qpos = np.zeros(1)
            self.qvel = np.zeros(1)
            self.ctrl = np.zeros(1)
            self.mocap_pos = np.zeros(1)
        elif model_name == "ur5":
            self.qpos = np.zeros(3)
            self.qvel = np.zeros(3)
            self.ctrl = np.zeros(3)
            self.mocap_pos = np.zeros(3)

        # variables that need to be defined for any mujoco model object
        self.actuator_ctrlrange = None

        # update the aforementioned variables if the model is known
        if model_name == "pendulum":
            self.actuator_ctrlrange = np.array([[0, 2]])
        elif model_name == "ur5":
            self.actuator_ctrlrange = np.array([[0, 3.15], [0, 5], [0, 3.15]])
