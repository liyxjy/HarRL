import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

try:
    from models2.utils import (
        Embedder,
        Pointer,
        Glimpse,
        SinBuffer,
        ObstaclePredictor
    )
    from models2.self_attn import Encoder
except ImportError:
    from utils import (
        Embedder,
        Pointer,
        Glimpse,
        SinBuffer,
        ObstaclePredictor
    )
    from self_attn import Encoder


class Actor(nn.Module):
    def __init__(self, input_dim, device, obstacle_dim=4):
        super().__init__()
        self.device = device

        self.node_input_dim = input_dim
        self.obstacle_input_dim = obstacle_dim

        self.d_model = 128
        self.d_unit = 256
        self.d_query = 360

        # node / obstacle 分开 embed
        self.node_embedder = Embedder(self.node_input_dim, self.d_model)
        self.obstacle_embedder = Embedder(self.obstacle_input_dim, self.d_model)

        # encoder
        self.num_stacks = 3
        self.num_heads = 16
        self.d_k = 16
        self.d_v = 16
        self.d_inner = 512

        self.encoder = Encoder(
            self.num_stacks,
            self.num_heads,
            self.d_k,
            self.d_v,
            self.d_model,
            self.d_inner
        )

        # decoder
        self.conv1d_r = nn.Conv1d(self.d_model, self.d_unit, 1)
        self.conv1d_x = nn.Conv1d(self.d_model, self.d_unit, 1)
        self.conv1d_y = nn.Conv1d(self.d_model, self.d_unit, 1)

        self.start_ptr = Pointer(self.d_query, self.d_unit)
        self.ptr1 = Pointer(self.d_query, self.d_unit)
        self.ptr2 = Pointer(self.d_query, self.d_unit)

        self.q_l1 = nn.Linear(self.d_model, self.d_query, bias=False)
        self.q_l2 = nn.Linear(self.d_model, self.d_query, bias=False)
        self.q_l3 = nn.Linear(self.d_model, self.d_query, bias=False)
        self.q_l4 = nn.Linear(self.d_model, self.d_query, bias=False)
        self.q_lx = nn.Linear(self.d_model, self.d_query, bias=False)
        self.q_ly = nn.Linear(self.d_model, self.d_query, bias=False)

        self.relu = nn.ReLU()

        # original contexts
        self.ctx_linear_T = nn.Linear(self.d_query, self.d_query, bias=False)
        self.ctx_linear_B = nn.Linear(self.d_query, self.d_query, bias=False)

        # new obstacle context
        self.ctx_linear_O = nn.Linear(self.d_query, self.d_query, bias=False)

        # buffer phase
        self.bufferloc = SinBuffer(self.d_query)

        # obstacle param predictor
        self.obstacle_predictor = ObstaclePredictor(
            d_query=self.d_query,
            d_model=self.d_model,
            hidden_dim=256
        )

        # map obstacle parameters Ψ_t ∈ R^N to d_query-dim context feature
        # 用固定上限做线性层会不灵活，因此这里采用 pooled summary:
        # pooled obs param -> scalar -> project to d_query
        self.obs_summary_proj = nn.Linear(1, self.d_query, bias=False)

        self.to(device)
        self.train()

    def _to_tensor(self, x, expected_last_dim=None):
        if torch.is_tensor(x):
            out = x.to(self.device, dtype=torch.float)
        else:
            out = torch.tensor(x, dtype=torch.float, device=self.device)

        if expected_last_dim is not None and out.size(-1) != expected_last_dim:
            raise ValueError(
                f"Expected last dim = {expected_last_dim}, got {out.size(-1)}"
            )
        return out

    def forward(self, obstacles, inputs, deterministic=False):
        """
        inputs:
            [batch, degree, 3]

        obstacles:
            [num_obs, 4] or [batch, num_obs, 4]

        return:
            edge_outputs:     [batch, 3 * (degree - 1)]
            obstacle_outputs: [batch, degree - 1, num_obs]
            log_probs:        [batch]
        """
        node_inputs = self._to_tensor(inputs, expected_last_dim=self.node_input_dim)
        obstacle_inputs = self._to_tensor(obstacles)

        if obstacle_inputs.dim() == 2:
            obstacle_inputs = obstacle_inputs.unsqueeze(0).expand(node_inputs.size(0), -1, -1)
        elif obstacle_inputs.dim() != 3:
            raise ValueError("obstacles must have shape [num_obs, 4] or [batch, num_obs, 4]")

        if obstacle_inputs.size(0) != node_inputs.size(0):
            raise ValueError("Batch size mismatch between obstacles and inputs")

        if obstacle_inputs.size(-1) != self.obstacle_input_dim:
            raise ValueError(
                f"Each obstacle must have {self.obstacle_input_dim} dims, got {obstacle_inputs.size(-1)}"
            )

        batch_size, degree, _ = node_inputs.shape
        num_obs = obstacle_inputs.size(1)

        # separate embeddings
        node_embeddings = self.node_embedder(node_inputs)                 # [B, degree, d_model]
        obstacle_embeddings = self.obstacle_embedder(obstacle_inputs)     # [B, N, d_model]

        # joint encoder
        joint_embeddings = torch.cat([node_embeddings, obstacle_embeddings], dim=1)
        joint_encodings = self.encoder(joint_embeddings, None)

        node_encodings = joint_encodings[:, :degree, :]       # [B, degree, d_model]
        obstacle_encodings = joint_encodings[:, degree:, :]   # [B, N, d_model]

        # pointer only over nodes
        node_encodings_t = node_encodings.permute(0, 2, 1)
        enc_r = self.conv1d_r(node_encodings_t).permute(0, 2, 1)
        enc_x = self.conv1d_x(node_encodings_t).permute(0, 2, 1)
        enc_y = self.conv1d_y(node_encodings_t).permute(0, 2, 1)
        enc_xy = torch.cat([enc_x, enc_y], dim=1)

        visited = torch.zeros([batch_size, degree], dtype=torch.bool, device=self.device)

        edge_outputs = []
        obstacle_outputs = []
        log_probs = []

        batch_arange = torch.arange(batch_size, device=self.device)

        # start node
        start_query = torch.zeros([batch_size, self.d_query], dtype=torch.float, device=self.device)
        start_logits = self.start_ptr(enc_r, start_query, visited)
        start_dist = Categorical(logits=start_logits)

        # 保持原始实现：起点固定为 0
        start_idx = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        visited.scatter_(1, start_idx.unsqueeze(-1), True)
        log_probs.append(start_dist.log_prob(start_idx))

        q1 = node_encodings[batch_arange, start_idx]
        q2 = q1
        qx = q1
        qy = q1

        # contexts
        context_T = torch.zeros([batch_size, self.d_query], device=self.device)
        context_B = torch.zeros([batch_size, self.d_query], device=self.device)
        context_O = torch.zeros([batch_size, self.d_query], device=self.device)

        can_pos = torch.zeros([batch_size], device=self.device)

        for _ in range(degree - 1):
            # implicit edge_{t-1} encoding
            residual = self.q_l1(q1) + self.q_l2(q2) + self.q_lx(qx) + self.q_ly(qy)

            # update topology and buffer contexts
            context_T = torch.max(
                context_T,
                self.ctx_linear_T(self.relu(residual))
            )

            context_B = torch.max(
                context_B,
                self.ctx_linear_B(self.relu(residual + can_pos.unsqueeze(1) / (2 * torch.pi)))
            )

            # obstacle-aware first query:
            # q_t^u = ReLU(edge_{t-1} + T_t + B_t + O_t)
            first_q = residual + context_T + context_B + context_O
            first_query = self.relu(first_q)

            # pick an unvisited node u_t
            first_logits = self.ptr1(enc_r, first_query, visited)
            first_dist = Categorical(logits=first_logits)
            if deterministic:
                _, first_idx = torch.max(first_logits, dim=-1)
            else:
                first_idx = first_dist.sample()
            log_probs.append(first_dist.log_prob(first_idx))

            # build second query q_t^w
            q3 = node_encodings[batch_arange, first_idx]
            second_query = self.relu(first_q + self.q_l3(q3))

            # pick the connected visited node + direction
            unvisited = ~visited
            unvisited_2way = torch.cat([unvisited, unvisited], dim=-1)

            second_logits = self.ptr2(enc_xy, second_query, unvisited_2way)
            second_dist = Categorical(logits=second_logits)
            if deterministic:
                _, idxs = torch.max(second_logits, dim=-1)
            else:
                idxs = second_dist.sample()
            log_probs.append(second_dist.log_prob(idxs))

            second_idx = idxs % degree
            sec_dir = torch.div(idxs, degree, rounding_mode='floor')
            fir_dir = 1 - sec_dir

            x_idx = first_idx * sec_dir + second_idx * fir_dir
            y_idx = first_idx * fir_dir + second_idx * sec_dir

            edge_outputs.append(x_idx)
            edge_outputs.append(y_idx)

            visited.scatter_(1, first_idx.unsqueeze(-1), True)

            q1 = q3
            q2 = node_encodings[batch_arange, second_idx]
            qx = node_encodings[batch_arange, x_idx]
            qy = node_encodings[batch_arange, y_idx]

            # current step edge context for continuous heads
            query_b = self.relu(second_query + self.q_l4(q2))

            # buffer phase φ_t
            can_pos = self.bufferloc(query_b)
            edge_outputs.append(can_pos)

            # obstacle params Ψ_t for all obstacles at current step
            obs_param = self.obstacle_predictor(query_b, obstacle_encodings)  # [B, N]
            obstacle_outputs.append(obs_param)

            # simplest obstacle context update:
            # O_{t+1} = max(O_t, W_O(g(Ψ_t)))
            # use mean pooling over obstacles as g(Ψ_t)
            if num_obs > 0:
                obs_summary = obs_param.mean(dim=1, keepdim=True)  # [B, 1]
                obs_context_candidate = self.ctx_linear_O(
                    self.relu(self.obs_summary_proj(obs_summary))
                )
                context_O = torch.max(context_O, obs_context_candidate)

        edge_outputs = torch.stack(edge_outputs, dim=-1)  # [B, 3*(degree-1)]

        if num_obs > 0:
            obstacle_outputs = torch.stack(obstacle_outputs, dim=1)  # [B, degree-1, N]
        else:
            obstacle_outputs = torch.empty(batch_size, degree - 1, 0, device=self.device)

        log_probs = torch.stack(log_probs, dim=0).sum(dim=0)

        return edge_outputs, obstacle_outputs, log_probs


class Critic(nn.Module):
    def __init__(self, input_dim, device, obstacle_dim=4):
        super().__init__()
        self.device = device

        self.node_input_dim = input_dim
        self.obstacle_input_dim = obstacle_dim

        self.d_model = 128
        self.d_unit = 256

        self.node_embedder = Embedder(self.node_input_dim, self.d_model)
        self.obstacle_embedder = Embedder(self.obstacle_input_dim, self.d_model)

        self.num_stacks = 3
        self.num_heads = 16
        self.d_k = 16
        self.d_v = 16
        self.d_inner = 512

        self.crit_encoder = Encoder(
            self.num_stacks,
            self.num_heads,
            self.d_k,
            self.d_v,
            self.d_model,
            self.d_inner
        )

        self.glimpse = Glimpse(self.d_model, self.d_unit)
        self.critic_l1 = nn.Linear(self.d_model, self.d_unit)
        self.critic_l2 = nn.Linear(self.d_unit, 1)
        self.relu = nn.ReLU()

        self.to(device)
        self.train()

    def _to_tensor(self, x, expected_last_dim=None):
        if torch.is_tensor(x):
            out = x.to(self.device, dtype=torch.float)
        else:
            out = torch.tensor(x, dtype=torch.float, device=self.device)

        if expected_last_dim is not None and out.size(-1) != expected_last_dim:
            raise ValueError(
                f"Expected last dim = {expected_last_dim}, got {out.size(-1)}"
            )
        return out

    def forward(self, obstacles, inputs):
        """
        inputs:
            [batch, degree, 3]

        obstacles:
            [num_obs, 4] or [batch, num_obs, 4]
        """
        node_inputs = self._to_tensor(inputs, expected_last_dim=self.node_input_dim)
        obstacle_inputs = self._to_tensor(obstacles)

        if obstacle_inputs.dim() == 2:
            obstacle_inputs = obstacle_inputs.unsqueeze(0).expand(node_inputs.size(0), -1, -1)
        elif obstacle_inputs.dim() != 3:
            raise ValueError("obstacles must have shape [num_obs, 4] or [batch, num_obs, 4]")

        if obstacle_inputs.size(0) != node_inputs.size(0):
            raise ValueError("Batch size mismatch between obstacles and inputs")

        if obstacle_inputs.size(-1) != self.obstacle_input_dim:
            raise ValueError(
                f"Each obstacle must have {self.obstacle_input_dim} dims, got {obstacle_inputs.size(-1)}"
            )

        node_embeddings = self.node_embedder(node_inputs)
        obstacle_embeddings = self.obstacle_embedder(obstacle_inputs)

        joint_embeddings = torch.cat([node_embeddings, obstacle_embeddings], dim=1)
        critic_encode = self.crit_encoder(joint_embeddings, None)

        glimpse = self.glimpse(critic_encode)
        critic_inner = self.relu(self.critic_l1(glimpse))
        predictions = self.relu(self.critic_l2(critic_inner)).squeeze(-1)

        return predictions