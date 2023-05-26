from gym import Wrapper
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.wrapper import EnvWrapper
import uuid
from copy import deepcopy
from lxml import etree

ENV_KWARGS = dict(
    fov_range=[70, 70],
    frameskip=1,
    gamma_range=[2, 2],
    guiscale_range=[1, 1],
    resolution=[640, 360],
    cursor_size_range=[16.0, 16.0],
)

class HumanControlMineDojo(Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.task = HumanSurvival(**ENV_KWARGS)
        self._setup_spaces()

    def _setup_spaces(self) -> None:
        self.observation_space = self.task.observation_space
        self.action_space = self.task.action_space
        self.monitor_space = self.task.monitor_space

    def reset(self):
        """Resets the environment to an initial state and returns an initial observation.

        Return:
            Agent’s initial observation.
        """
        episode_id = str(uuid.uuid4())

        xml = etree.fromstring(self.env._sim_spec.to_xml(episode_id))
        raw_obs = self.env._bridge_env.reset(episode_id, [xml])[0]
        obs, info = self._process_raw_obs(raw_obs)
        self.env._prev_obs, self.env._prev_info = deepcopy(obs), deepcopy(info)
        return obs
    
    def _process_raw_obs(self, raw_obs: dict):
        info = deepcopy(raw_obs)
        if "pov" in info:
            info.pop("pov")

        obs_dict = {
            h.to_string(): h.from_hero(raw_obs) for h in self.env._sim_spec.observables
        }

        # For only the 'pov' observation, we process with the MineRL wrapper.
        bottom_env_spec = self.task
        while isinstance(bottom_env_spec, EnvWrapper):
            bottom_env_spec = bottom_env_spec.env_to_wrap

        pov_specs = [h for h in bottom_env_spec.observables if h.to_string() == 'pov']
        assert len(pov_specs) > 0, "No 'pov' observation found in the bottom env spec."
        for h in pov_specs:
            obs_dict[h.to_string()] = h.from_hero(info)

        return obs_dict, info
    
    def step(self, action: dict):
        """Run one timestep of the environment’s dynamics. Accepts an action and returns next_obs, reward, done, info.

        Args:
            action: The action of the agent in current step.

        Return:
            A tuple (obs, reward, done, info)
            - ``dict`` - Agent’s next observation.
            - ``float`` - Amount of reward returned after executing previous action.
            - ``bool`` - Whether the episode has ended.
            - ``dict`` - Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        self.env._prev_action = deepcopy(action)
        action_xml = self.env._action_obj_to_xml(action)
        step_tuple = self.env._bridge_env.step([action_xml])
        step_success, raw_obs = step_tuple.step_success, step_tuple.raw_obs
        if not step_success:
            # when step failed, return prev obs
            return self.env._prev_obs, 0, True, self.env._prev_info
        else:
            obs, info = self.env._process_raw_obs(raw_obs[0])
            self.env._prev_obs, self.env._prev_info = deepcopy(obs), deepcopy(info)
            return obs, 0, self.env.is_terminated, info
        
    def _action_obj_to_xml(self, action):
        action = deepcopy(action)

        if isinstance(self.task, EnvWrapper):
            action = self.task.unwrap_action(action)

        bottom_env_spec = self.task
        while isinstance(bottom_env_spec, EnvWrapper):
            bottom_env_spec = bottom_env_spec.env_to_wrap

        action_str = []
        for h in bottom_env_spec.actionables:
            if h.to_string() in action:
                action_str.append(h.to_hero(action[h.to_string()]))

        return "\n".join(action_str)