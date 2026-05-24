import math
import numpy as np
import torch
import torch.nn as nn


class Embedder(nn.Module):
    def __init__(self, d_input, d_model):
        super().__init__()
        self.conv1d = nn.Conv1d(d_input, d_model, 1)
        self.batch_norm = nn.BatchNorm1d(d_model)

    def forward(self, inputs):
        embeddings = self.conv1d(inputs.permute(0, 2, 1))
        embeddings = self.batch_norm(embeddings).permute(0, 2, 1)
        return embeddings


class SlideBuffer(nn.Module):
    def __init__(self, d_model, d_query):
        super().__init__()
        self.d_inner1 = 512
        self.d_inner2 = 256

        self.fc1 = nn.Linear(2 * d_model + d_query, self.d_inner1)
        self.fc2 = nn.Linear(self.d_inner1, self.d_inner2)
        self.fc3 = nn.Linear(self.d_inner2, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, q_v, q_h, query):
        combined = torch.cat((q_v, q_h, query), dim=-1)
        x = self.relu(self.fc1(combined))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


class SinBuffer(nn.Module):
    def __init__(self, d_query):
        super().__init__()
        self.fc1 = nn.Linear(d_query, d_query)
        self.tanh = nn.Tanh()
        self.h = nn.Parameter(torch.randn(d_query))
        self.sigmoid = nn.Sigmoid()

    def forward(self, query):
        x = self.tanh(self.fc1(query))
        x = torch.matmul(x, self.h)
        x = self.sigmoid(x) * (2 * math.pi)
        return x


class ObstaclePredictor(nn.Module):
    """
    step query + obstacle encoding -> 对每个 obstacle 输出一个连续参数
    输出 shape: [batch, num_obstacles]
    """
    def __init__(self, d_query, d_model, hidden_dim=256):
        super().__init__()
        self.query_proj = nn.Linear(d_query, hidden_dim, bias=False)
        self.obs_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.fuse_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 1)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, query, obstacle_encodings):
        """
        query: [B, d_query]
        obstacle_encodings: [B, N, d_model]
        return: [B, N]
        """
        if obstacle_encodings.size(1) == 0:
            return torch.empty(query.size(0), 0, device=query.device)

        q = self.query_proj(query).unsqueeze(1)   # [B, 1, H]
        o = self.obs_proj(obstacle_encodings)     # [B, N, H]

        h = self.tanh(q + o)
        h = self.tanh(self.fuse_proj(h))
        out = self.sigmoid(self.out_proj(h).squeeze(-1)) * (2 * math.pi)
        return out


class Pointer(nn.Module):
    def __init__(self, d_query, d_unit):
        super().__init__()
        self.tanh = nn.Tanh()
        self.w_l = nn.Linear(d_query, d_unit, bias=False)
        self.v = nn.Parameter(torch.FloatTensor(d_unit), requires_grad=True)
        self.v.data.uniform_(-(1. / math.sqrt(d_unit)), 1. / math.sqrt(d_unit))

    def forward(self, refs, query, mask):
        scores = torch.sum(self.v * self.tanh(refs + self.w_l(query).unsqueeze(1)), -1)
        scores = 10. * self.tanh(scores)
        with torch.no_grad():
            scores[mask] = float('-inf')
        return scores


class Glimpse(nn.Module):
    def __init__(self, d_model, d_unit):
        super().__init__()
        self.tanh = nn.Tanh()
        self.conv1d = nn.Conv1d(d_model, d_unit, 1)
        self.v = nn.Parameter(torch.FloatTensor(d_unit), requires_grad=True)
        self.v.data.uniform_(-(1. / math.sqrt(d_unit)), 1. / math.sqrt(d_unit))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, encs):
        encoded = self.conv1d(encs.permute(0, 2, 1)).permute(0, 2, 1)
        scores = torch.sum(self.v * self.tanh(encoded), -1)
        attention = self.softmax(scores)
        glimpse = attention.unsqueeze(-1) * encs
        glimpse = torch.sum(glimpse, 1)
        return glimpse


# -----------------------------------------------------------------------------
# 旧的 CreateSinks 不再使用。
# 避免原文件里的语法错误影响整个 utils.py 导入。
# -----------------------------------------------------------------------------
"""
class CreateSinks(obstacles, batch_size, degree):
    def __init__(self):
        self.obstacles = obstacles
        self.batch_size = batch_size
        self.degree = degree

    def is_in_obstacle(self, pin):
        for i in range(len(self.obstacles)):
            x1, x2, y1, y2 = obstacles[i]
            x_min = min(x1, x2)
            x_max = max(x1, x2)
            y_min = min(y1, y2)
            y_max = max(y1, y2)
            if pin[0] >= x_min and pin[0] =< lx_max and pin[1] >= y_min and pin[1] =< y_max:
                return True
        return False

    def create_pins(self):
        x_coords = np.zeros((self.batch_size, self.degree, 2))
        for i in range(self.batch_size):
            for j in range(self.degree):
                xcor = random.uniform(0, 1)
                ycor = random.uniform(0, 1)
                while is_in_obstacle((xcor, ycor), self.obstacles):
                    xcor = random.uniform(0, 1)
                    ycor = random.uniform(0, 1)
                x_coords[i][j][0] = xcor
                x_coords[i][j][1] = ycor
        return x_coords
"""


class ObstacleAwareCreateSinks1:
    """
    新的可执行版本：
    1. 在 [0,1] x [0,1] 内生成 pin
    2. pin 不落在 obstacle 内部
    3. 支持生成 obstacle batch，供 actor / critic 使用
    """
    def __init__(self, obstacles, batch_size, degree, seed=None):
        self.obstacles = [tuple(map(float, ob)) for ob in obstacles]
        self.batch_size = int(batch_size)
        self.degree = int(degree)
        self.rng = np.random.default_rng(seed)

    def is_in_obstacle(self, pin):
        x, y = float(pin[0]), float(pin[1])
        for x1, y1, x2, y2 in self.obstacles:
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return True
        return False

    def create_pins(self):
        x_coords = np.zeros((self.batch_size, self.degree, 2), dtype=np.float32)
        for i in range(self.batch_size):
            for j in range(self.degree):
                while True:
                    pin = self.rng.random(2).astype(np.float32)
                    if not self.is_in_obstacle(pin):
                        x_coords[i, j] = pin
                        break
        return x_coords

    def obstacle_tensor(self):
        """
        return:
            [batch_size, num_obstacles, 4]
        """
        if len(self.obstacles) == 0:
            return np.zeros((self.batch_size, 0, 4), dtype=np.float32)

        obs = np.asarray(self.obstacles, dtype=np.float32)
        obs = np.broadcast_to(obs[None, :, :], (self.batch_size, obs.shape[0], obs.shape[1]))
        return obs.copy()
    

def generate_obstacles(rng, num_obstacles=None, max_attempts=1000):
    """
    在 (0,0)-(1,1) 区域内随机生成不重叠的矩形障碍物

    Args:
        rng: np.random.Generator
        num_obstacles: int, 障碍物数量，None 则随机 3-10 个
        max_attempts: int, 每个障碍物的最大尝试次数

    Returns:
        list of tuples: [(x1, y1, x2, y2), ...]
    """
    def overlaps_with_existing(x1, y1, x2, y2, existing):
        for ex1, ey1, ex2, ey2 in existing:
            if not (x2 <= ex1 or x1 >= ex2 or y2 <= ey1 or y1 >= ey2):
                return True
        return False

    count = num_obstacles if num_obstacles is not None else int(rng.integers(3, 11))
    obstacles = []
    attempts = 0
    while len(obstacles) < count and attempts < max_attempts:
        w, h = rng.uniform(0.05, 0.2, size=2)
        x1 = rng.uniform(0.0, 1.0 - w)
        y1 = rng.uniform(0.0, 1.0 - h)
        x2 = round(x1 + w, 4)
        y2 = round(y1 + h, 4)
        x1, y1 = round(x1, 4), round(y1, 4)

        if not overlaps_with_existing(x1, y1, x2, y2, obstacles):
            obstacles.append((x1, y1, x2, y2))

        attempts += 1

    if len(obstacles) < count:
        raise RuntimeError(
            f"仅生成了 {len(obstacles)}/{count} 个障碍物，"
            f"空间不足，请减少 num_obstacles 或增大 max_attempts"
        )
    return obstacles


class ObstacleAwareCreateSinks:
    def __init__(self, batch_size, degree, seed=None, num_obstacles=None):
        self.batch_size = int(batch_size)
        self.degree = int(degree)
        self.num_obstacles = num_obstacles
        self.rng = np.random.default_rng(seed)
        # 每个 batch 独立一套障碍物
        self.obstacles_batch = [
            generate_obstacles(self.rng, num_obstacles)
            for _ in range(self.batch_size)
        ]

    def is_in_obstacle(self, pin, obstacles):
        x, y = float(pin[0]), float(pin[1])
        for x1, y1, x2, y2 in obstacles:
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return True
        return False

    def create_pins(self):
        x_coords = np.zeros((self.batch_size, self.degree, 2), dtype=np.float32)
        for i in range(self.batch_size):
            for j in range(self.degree):
                while True:
                    pin = self.rng.random(2).astype(np.float32)
                    if not self.is_in_obstacle(pin, self.obstacles_batch[i]):
                        x_coords[i, j] = pin
                        break
        return x_coords

    def obstacle_tensor(self):
        """
        return:
            [batch_size, max_num_obstacles, 4]
        不足 max_num_obstacles 的位置用 0 padding
        """
        if not any(self.obstacles_batch):
            return np.zeros((self.batch_size, 0, 4), dtype=np.float32)

        max_n = max(len(obs) for obs in self.obstacles_batch)
        tensor = np.full((self.batch_size, max_n, 4), fill_value=0, dtype=np.float32)
        for i, obstacles in enumerate(self.obstacles_batch):
            tensor[i, :len(obstacles)] = np.asarray(obstacles, dtype=np.float32)
        return tensor.copy()