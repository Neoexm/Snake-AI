# snake_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional

DIRS = {
    0: np.array([0, -1]),   # left
    1: np.array([0,  1]),   # right
    2: np.array([-1, 0]),   # up
    3: np.array([1,  0]),   # down
}
OPPOSITE = {0:1, 1:0, 2:3, 3:2}

class SnakeEnv(gym.Env):
    """
    Observation: (3, H, W) float32 planes:
      plane 0: snake body (1.0 where snake is)
      plane 1: food position
      plane 2: head position
    Action: Discrete(4) [left, right, up, down]
    Reward: +1 eat, -1 death, -0.01 step
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, grid_size: int = 12, max_steps: Optional[int] = None, render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.h = self.w = grid_size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3, self.h, self.w), dtype=np.float32
        )
        self.max_steps = max_steps or 200 * grid_size
        self.render_mode = render_mode
        self.rng = np.random.default_rng()
        self.reset(seed=None)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        c = self.grid_size // 2
        self.snake = [np.array([c, c-1]), np.array([c, c]), np.array([c, c+1])]  # length 3 horizontal
        self.dir = 1  # moving right
        self._place_food()
        self.steps = 0
        self.done = False
        obs = self._obs()
        info = {}
        return obs, info

    def _place_food(self):
        occ = {tuple(p) for p in self.snake}
        free = [(i, j) for i in range(self.h) for j in range(self.w) if (i, j) not in occ]
        self.food = np.array(self.rng.choice(free))

    def _obs(self):
        grid = np.zeros((3, self.h, self.w), dtype=np.float32)
        for y, x in self.snake:
            grid[0, y, x] = 1.0
        fy, fx = self.food
        grid[1, fy, fx] = 1.0
        hy, hx = self.snake[-1]
        grid[2, hy, hx] = 1.0
        return grid

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.done:
            raise RuntimeError("Call reset() before step() after episode end")

        # Disallow 180° turns (ignore invalid reverse)
        if len(self.snake) > 1 and action == OPPOSITE[self.dir]:
            action = self.dir
        self.dir = action

        # move head
        head = self.snake[-1].copy()
        head += DIRS[self.dir]
        self.steps += 1

        # check walls
        y, x = head
        out = (y < 0) or (y >= self.h) or (x < 0) or (x >= self.w)

        # check self-collision
        body = {tuple(p) for p in self.snake}
        hit_self = tuple(head) in body

        reward = -0.01
        terminated = False

        if out or hit_self or self.steps >= self.max_steps:
            reward = -1.0
            terminated = True
        else:
            # eat?
            if (head == self.food).all():
                reward = +1.0
                self.snake.append(head)         # grow
                self._place_food()
            else:
                self.snake.append(head)
                self.snake.pop(0)               # move without growing

        self.done = terminated
        obs = self._obs()
        truncated = False  # we use max_steps as terminated for simplicity
        info = {"length": len(self.snake)}
        return obs, reward, terminated, truncated, info

    def render(self):
        # produce an RGB array (H, W, 3)
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        img[self.food[0], self.food[1]] = (255, 0, 0)
        for y, x in self.snake[:-1]:
            img[y, x] = (0, 200, 0)
        hy, hx = self.snake[-1]
        img[hy, hx] = (0, 255, 0)
        # scale up for easier viewing
        k = 20
        return np.kron(img, np.ones((k, k, 1), dtype=np.uint8))
