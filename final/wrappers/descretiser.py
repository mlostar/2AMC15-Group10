from gym import Wrapper, spaces, Env


class Discretiser(Wrapper):
    def __init__(self, env: Env, n_splits):
        assert isinstance(env.action_space, spaces.Box)

        super().__init__(env)

        self._n_splits = n_splits
        self.action_space = spaces.Discrete(n_splits)
        self._lowest_action = self.env.action_space.low
        self._highest_action = self.env.action_space.high

    def step(self, action):
        converted_action = (self._highest_action-self._lowest_action)/self._n_splits*action
        return self.env.step(converted_action)
