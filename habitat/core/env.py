#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union
import pickle
import torch
from scipy import ndimage, misc
import gym
import numpy as np
from gym.spaces.dict_space import Dict as SpaceDict
from habitat import config

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode, EpisodeIterator
from habitat.core.embodied_task import EmbodiedTask, Metrics
from habitat.core.simulator import Observations, Simulator
from habitat.datasets import make_dataset
from habitat.sims import make_sim
from habitat.tasks import make_task
from habitat_baselines.common.utils import quat_from_angle_axis

import matplotlib.pyplot as plt

from PIL import Image
def display_sample(rgb_obs):
    rgb_img = Image.fromarray(rgb_obs, mode="RGB")
    arr = [rgb_img]
    plt.imshow(rgb_img)
    plt.show()



class Env:
    r"""Fundamental environment class for `habitat`.

    :data observation_space: ``SpaceDict`` object corresponding to sensor in
        sim and task.
    :data action_space: ``gym.space`` object corresponding to valid actions.

    All the information  needed for working on embodied tasks with simulator
    is abstracted inside `Env`. Acts as a base for other derived environment
    classes. `Env` consists of three major components: ``dataset`` (`episodes`), ``simulator`` (`sim`) and `task` and connects all the three components
    together.
    """

    observation_space: SpaceDict
    action_space: SpaceDict
    _config: Config
    _dataset: Optional[Dataset]
    _episodes: List[Type[Episode]]
    _current_episode_index: Optional[int]
    _current_episode: Optional[Type[Episode]]
    _episode_iterator: Optional[Iterator]
    _sim: Simulator
    _task: EmbodiedTask
    _max_episode_seconds: int
    _max_episode_steps: int
    _elapsed_steps: int
    _episode_start_time: Optional[float]
    _episode_over: bool

    def __init__(
        self, config: Config, dataset: Optional[Dataset] = None
    ) -> None:
        """Constructor

        :param config: config for the environment. Should contain id for
            simulator and ``task_name`` which are passed into ``make_sim`` and
            ``make_task``.
        :param dataset: reference to dataset for task instance level
            information. Can be defined as :py:`None` in which case
            ``_episodes`` should be populated from outside.
        """

        assert config.is_frozen(), (
            "Freeze the config before creating the "
            
            "environment, use config.freeze()."
        )
        self._config = config
        self._dataset = dataset
        self._current_episode_index = None
        if self._dataset is None and config.DATASET.TYPE:
            self._dataset = make_dataset(
                id_dataset=config.DATASET.TYPE, config=config.DATASET
            )
        self._episodes = self._dataset.episodes if self._dataset else []
        self._current_episode = None
        iter_option_dict = {
            k.lower(): v
            for k, v in config.ENVIRONMENT.ITERATOR_OPTIONS.items()
        }
        iter_option_dict["seed"] = config.SEED
        self._episode_iterator = self._dataset.get_episode_iterator(
            **iter_option_dict
        )

        # load the first scene if dataset is present
        if self._dataset:
            assert (
                len(self._dataset.episodes) > 0
            ), "dataset should have non-empty episodes list"
            self._config.defrost()
            self._config.SIMULATOR.SCENE = self._dataset.episodes[0].scene_id
            self._config.freeze()

        self._sim = make_sim(
            id_sim=self._config.SIMULATOR.TYPE, config=self._config.SIMULATOR
        )
        self._task = make_task(
            self._config.TASK.TYPE,
            config=self._config.TASK,
            sim=self._sim,
            dataset=self._dataset,
        )
        self.observation_space = SpaceDict(
            {
                **self._sim.sensor_suite.observation_spaces.spaces,
                **self._task.sensor_suite.observation_spaces.spaces,
            }
        )
        self.action_space = self._task.action_space
        self._max_episode_seconds = (
            self._config.ENVIRONMENT.MAX_EPISODE_SECONDS
        )
        self._max_episode_steps = self._config.ENVIRONMENT.MAX_EPISODE_STEPS
        self._elapsed_steps = 0
        self._episode_start_time: Optional[float] = None
        self._episode_over = False
        if config.TRAINER_NAME in ["oracle", "oracle-ego"]:
            with open('oracle_maps/map300.pickle', 'rb') as handle:
                self.mapCache = pickle.load(handle)
        if config.TRAINER_NAME == "oracle-ego":
            for x,y in self.mapCache.items():
                self.mapCache[x] += 1

    @property
    def current_episode(self) -> Type[Episode]:
        assert self._current_episode is not None
        return self._current_episode

    @current_episode.setter
    def current_episode(self, episode: Type[Episode]) -> None:
        self._current_episode = episode

    @property
    def episode_iterator(self) -> Iterator:
        return self._episode_iterator

    @episode_iterator.setter
    def episode_iterator(self, new_iter: Iterator) -> None:
        self._episode_iterator = new_iter

    @property
    def episodes(self) -> List[Type[Episode]]:
        return self._episodes

    @episodes.setter
    def episodes(self, episodes: List[Type[Episode]]) -> None:
        assert (
            len(episodes) > 0
        ), "Environment doesn't accept empty episodes list."
        self._episodes = episodes

    @property
    def sim(self) -> Simulator:
        return self._sim

    @property
    def episode_start_time(self) -> Optional[float]:
        return self._episode_start_time

    @property
    def episode_over(self) -> bool:
        return self._episode_over

    @property
    def task(self) -> EmbodiedTask:
        return self._task

    @property
    def _elapsed_seconds(self) -> float:
        assert (
            self._episode_start_time
        ), "Elapsed seconds requested before episode was started."
        return time.time() - self._episode_start_time

    def get_metrics(self) -> Metrics:
        return self._task.measurements.get_metrics()

    def _past_limit(self) -> bool:
        if (
            self._max_episode_steps != 0
            and self._max_episode_steps <= self._elapsed_steps
        ):
            return True
        elif (
            self._max_episode_seconds != 0
            and self._max_episode_seconds <= self._elapsed_seconds
        ):
            return True
        return False

    def _reset_stats(self) -> None:
        self._episode_start_time = time.time()
        self._elapsed_steps = 0
        self._episode_over = False
    

    def conv_grid(
        self,
        realworld_x,
        realworld_y,
        coordinate_min = -120.3241-1e-6,
        coordinate_max = 120.0399+1e-6,
        grid_resolution = (300, 300)
    ):
        r"""Return gridworld index of realworld coordinates assuming top-left corner
        is the origin. The real world coordinates of lower left corner are
        (coordinate_min, coordinate_min) and of top right corner are
        (coordinate_max, coordinate_max)
        """
        grid_size = (
            (coordinate_max - coordinate_min) / grid_resolution[0],
            (coordinate_max - coordinate_min) / grid_resolution[1],
        )
        grid_x = int((coordinate_max - realworld_x) / grid_size[0])
        grid_y = int((realworld_y - coordinate_min) / grid_size[1])
        return grid_x, grid_y

    def reset(self) -> Observations:
        r"""Resets the environments and returns the initial observations.

        :return: initial observations from the environment.
        """
        self._reset_stats()
    
        assert len(self.episodes) > 0, "Episodes list is empty"

        self._current_episode = next(self._episode_iterator)
        self.reconfigure(self._config)
        
        # Remove existing objects from last episode
        for objid in self._sim._sim.get_existing_object_ids():  
            self._sim._sim.remove_object(objid)

        # Insert object here
        object_to_datset_mapping = {'cylinder_red':0, 'cylinder_green':1, 'cylinder_blue':2,
            'cylinder_yellow':3, 'cylinder_white':4, 'cylinder_pink':5, 'cylinder_black':6, 'cylinder_cyan':7
        }
        for i in range(len(self.current_episode.goals)):
            current_goal = self.current_episode.goals[i].object_category
            dataset_index = object_to_datset_mapping[current_goal]
            ind = self._sim._sim.add_object(dataset_index)
            self._sim._sim.set_translation(np.array(self.current_episode.goals[i].position), ind)

        observations = self.task.reset(episode=self.current_episode)

        if self._config.TRAINER_NAME in ["oracle", "oracle-ego"]:
            self.currMap = np.copy(self.mapCache[self.current_episode.scene_id])
            self.task.occMap = self.currMap[:,:,0]
            self.task.sceneMap = self.currMap[:,:,0]


        self._task.measurements.reset_measures(
            episode=self.current_episode, task=self.task
        )

        if self._config.TRAINER_NAME in ["oracle", "oracle-ego"]:
            for i in range(len(self.current_episode.goals)):
                loc0 = self.current_episode.goals[i].position[0]
                loc2 = self.current_episode.goals[i].position[2]
                grid_loc = self.conv_grid(loc0, loc2)
                objIndexOffset = 1 if self._config.TRAINER_NAME == "oracle" else 2
                self.currMap[grid_loc[0]-1:grid_loc[0]+2, grid_loc[1]-1:grid_loc[1]+2, 1] = object_to_datset_mapping[self.current_episode.goals[i].object_category] + objIndexOffset

            currPix = self.conv_grid(observations["agent_position"][0], observations["agent_position"][2])  ## Explored area marking

            if self._config.TRAINER_NAME == "oracle-ego":
                self.expose = np.repeat(
                    self.task.measurements.measures["fow_map"].get_metric()[:, :, np.newaxis], 3, axis = 2
                )
                patch = self.currMap * self.expose
            elif self._config.TRAINER_NAME == "oracle":
                patch = self.currMap

            patch = patch[currPix[0]-40:currPix[0]+40, currPix[1]-40:currPix[1]+40,:]
            patch = ndimage.interpolation.rotate(patch, -(observations["heading"][0] * 180/np.pi) + 90, order=0, reshape=False)
            observations["semMap"] = patch[40-25:40+25, 40-25:40+25, :]
        return observations

    def _update_step_stats(self) -> None:
        self._elapsed_steps += 1
        self._episode_over = not self._task.is_episode_active
        if self._past_limit():
            self._episode_over = True

        if self.episode_iterator is not None and isinstance(
            self.episode_iterator, EpisodeIterator
        ):
            self.episode_iterator.step_taken()

    def step(
        self, action: Union[int, str, Dict[str, Any]], **kwargs
    ) -> Observations:
        r"""Perform an action in the environment and return observations.

        :param action: action (belonging to `action_space`) to be performed
            inside the environment. Action is a name or index of allowed
            task's action and action arguments (belonging to action's
            `action_space`) to support parametrized and continuous actions.
        :return: observations after taking action in environment.
        """

        assert (
            self._episode_start_time is not None
        ), "Cannot call step before calling reset"
        assert (
            self._episode_over is False
        ), "Episode over, call reset before calling step"

        # Support simpler interface as well
        if isinstance(action, str) or isinstance(action, (int, np.integer)):
            action = {"action": action}

        observations = self.task.step(
            action=action, episode=self.current_episode
        )

        self._task.measurements.update_measures(
            episode=self.current_episode, action=action, task=self.task
        )

        if self._config.TRAINER_NAME in ["oracle", "oracle-ego"]:
            currPix = self.conv_grid(observations["agent_position"][0], observations["agent_position"][2])  ## Explored area marking
            if self._config.TRAINER_NAME == "oracle-ego":
                self.expose = np.repeat(
                    self.task.measurements.measures["fow_map"].get_metric()[:, :, np.newaxis], 3, axis = 2
                )
                patch = self.currMap * self.expose
            elif self._config.TRAINER_NAME == "oracle":
                patch = self.currMap
            patch = patch[currPix[0]-40:currPix[0]+40, currPix[1]-40:currPix[1]+40,:]
            patch = ndimage.interpolation.rotate(patch, -(observations["heading"][0] * 180/np.pi) + 90, order=0, reshape=False)
            observations["semMap"] = patch[40-25:40+25, 40-25:40+25, :]

        ##Terminates episode if wrong found is called
        if self.task.is_found_called == True and \
            self.task.measurements.measures[
            "sub_success"
        ].get_metric() == 0:
            self.task._is_episode_active = False

        self._update_step_stats()
        return observations

    def seed(self, seed: int) -> None:
        self._sim.seed(seed)
        self._task.seed(seed)

    def reconfigure(self, config: Config) -> None:
        self._config = config

        self._config.defrost()
        self._config.SIMULATOR = self._task.overwrite_sim_config(
            self._config.SIMULATOR, self.current_episode
        )
        self._config.freeze()

        self._sim.reconfigure(self._config.SIMULATOR)

    def render(self, mode="rgb") -> np.ndarray:
        return self._sim.render(mode)

    def close(self) -> None:
        self._sim.close()


class RLEnv(gym.Env):
    r"""Reinforcement Learning (RL) environment class which subclasses ``gym.Env``.

    This is a wrapper over `Env` for RL users. To create custom RL
    environments users should subclass `RLEnv` and define the following
    methods: `get_reward_range()`, `get_reward()`, `get_done()`, `get_info()`.

    As this is a subclass of ``gym.Env``, it implements `reset()` and
    `step()`.
    """

    _env: Env

    def __init__(
        self, config: Config, dataset: Optional[Dataset] = None
    ) -> None:
        """Constructor

        :param config: config to construct `Env`
        :param dataset: dataset to construct `Env`.
        """
        self._env = Env(config, dataset)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.reward_range = self.get_reward_range()

    @property
    def habitat_env(self) -> Env:
        return self._env

    @property
    def episodes(self) -> List[Type[Episode]]:
        return self._env.episodes

    @property
    def current_episode(self) -> Type[Episode]:
        return self._env.current_episode

    @episodes.setter
    def episodes(self, episodes: List[Type[Episode]]) -> None:
        self._env.episodes = episodes

    def reset(self) -> Observations:
        return self._env.reset()

    def get_reward_range(self):
        r"""Get min, max range of reward.

        :return: :py:`[min, max]` range of reward.
        """
        raise NotImplementedError

    def get_reward(self, observations: Observations) -> Any:
        r"""Returns reward after action has been performed.

        :param observations: observations from simulator and task.
        :return: reward after performing the last action.

        This method is called inside the `step()` method.
        """
        raise NotImplementedError

    def get_done(self, observations: Observations) -> bool:
        r"""Returns boolean indicating whether episode is done after performing
        the last action.

        :param observations: observations from simulator and task.
        :return: done boolean after performing the last action.

        This method is called inside the step method.
        """
        raise NotImplementedError

    def get_info(self, observations) -> Dict[Any, Any]:
        r"""..

        :param observations: observations from simulator and task.
        :return: info after performing the last action.
        """
        raise NotImplementedError

    def step(self, *args, **kwargs) -> Tuple[Observations, Any, bool, dict]:
        r"""Perform an action in the environment.

        :return: :py:`(observations, reward, done, info)`
        """

        observations = self._env.step(*args, **kwargs)
        reward = self.get_reward(observations, **kwargs)
        done = self.get_done(observations)
        info = self.get_info(observations)

        return observations, reward, done, info

    def seed(self, seed: Optional[int] = None) -> None:
        self._env.seed(seed)

    def render(self, mode: str = "rgb") -> np.ndarray:
        return self._env.render(mode)

    def close(self) -> None:
        self._env.close()
