import torch
import torch.nn as nn

from torch.distributions.categorical import Categorical
from models2.utils import Embedder, Pointer, Glimpse, SlideBuffer, SinBuffer
from models2.self_attn import Encoder

class Actor(nn.Module):
    def __init__(self, input_dim, device):
        super(Actor, self).__init__()
        self.device = device

        # embedder args
        self.d_input = input_dim
        self.d_model = 128
        self.embedder = Embedder(self.d_input, self.d_model)

        # encoder args
        self.num_stacks = 3
        self.num_heads = 16
        self.d_k = 16
        self.d_v = 16
        
        # feedforward layer inner
        self.d_inner = 512

        self.encoder = Encoder(self.num_stacks, self.num_heads, self.d_k, self.d_v, self.d_model, self.d_inner)

        # decoder args
        self.d_unit = 256
        self.d_query = 360
        self.conv1d_r = nn.Conv1d(self.d_model, self.d_unit, 1)
        self.conv1d_x = nn.Conv1d(self.d_model, self.d_unit, 1)
        self.conv1d_y = nn.Conv1d(self.d_model, self.d_unit, 1)

        self.start_ptr = Pointer(self.d_query, self.d_unit)
        self.q_l1 = nn.Linear(self.d_model, self.d_query, bias=False)
        self.q_l2 = nn.Linear(self.d_model, self.d_query, bias=False)
        self.q_l3 = nn.Linear(self.d_model, self.d_query, bias=False)
        self.q_l4 = nn.Linear(self.d_model, self.d_query, bias=False)
        self.q_lx = nn.Linear(self.d_model, self.d_query, bias=False)
        self.q_ly = nn.Linear(self.d_model, self.d_query, bias=False)
        self.relu = nn.ReLU()
        self.ctx_linear_T = nn.Linear(self.d_query, self.d_query, bias=False)
        self.ctx_linear_B = nn.Linear(self.d_query, self.d_query, bias=False)

        self.ptr1 = Pointer(self.d_query, self.d_unit)
        self.ptr2 = Pointer(self.d_query, self.d_unit)

        self.bufferloc = SinBuffer(self.d_query)

        self.to(device)
        self.train()

    def forward(self, inputs, deterministic=False):
        # encode encode encode
        degree = inputs.shape[1]
        inputs_tensor = torch.tensor(inputs, dtype=torch.float).to(self.device)
        embedings = self.embedder(inputs_tensor)
        encodings = self.encoder(embedings, None).permute(0, 2, 1)
        enc_r = self.conv1d_r(encodings).permute(0, 2, 1)
        enc_x = self.conv1d_x(encodings).permute(0, 2, 1)
        enc_y = self.conv1d_y(encodings).permute(0, 2, 1)
        encodings = encodings.permute(0, 2, 1)
        enc_xy = torch.cat([enc_x, enc_y], 1)

        batch_size = encodings.size()[0]

        visited = torch.zeros([batch_size, degree], dtype=torch.bool).to(self.device)

        indexes, log_probs = [], []

        # initial_idx = torch.zeros(batch_size, dtype=torch.long).to(self.device)
        start_logits = self.start_ptr(enc_r,
            torch.zeros([batch_size, self.d_query], dtype=torch.float).to(self.device), visited)
        distr = Categorical(logits=start_logits)
        if deterministic:
            # _, start_idx = torch.max(start_logits, -1)  # 注释掉原来的代码
            start_idx = torch.zeros(batch_size, dtype=torch.long).to(self.device)  # 确保 start_idx 都是 0
        else:
            # start_idx = distr.sample()  # 注释掉原来的代码
            start_idx = torch.zeros(batch_size, dtype=torch.long).to(self.device)  # 确保 start_idx 都是 0
        visited.scatter_(1, start_idx.unsqueeze(-1), True)
        log_probs.append(distr.log_prob(start_idx))

        q1 = encodings[torch.arange(batch_size), start_idx]
        q2 = q1
        qx = q1
        qy = q1

        context_T = torch.zeros([batch_size, self.d_query]).to(self.device)
        context_B = torch.zeros([batch_size, self.d_query]).to(self.device)
        can_pos = torch.zeros([batch_size]).to(self.device)

        for step in range(degree - 1):
            residual = self.q_l1(q1) + self.q_l2(q2) + self.q_lx(qx) + self.q_ly(qy)
            context_T = torch.max(context_T, self.ctx_linear_T(self.relu(residual)))
            context_B = torch.max(context_B, self.ctx_linear_B(self.relu(residual +  can_pos.unsqueeze(1) / (2 * torch.pi))))
            first_q = residual + context_T + context_B
            first_query = self.relu(first_q)
            # first_idx
            logits = self.ptr1(enc_r, first_query, visited)
            distr = Categorical(logits=logits)
            if deterministic:
                _, first_idx = torch.max(logits, -1)
            else:
                first_idx = distr.sample()
            log_probs.append(distr.log_prob(first_idx))

            # second_idx
            q3 = encodings[torch.arange(encodings.size(0)), first_idx]
            second_query = self.relu(first_q + self.q_l3(q3))

            unvisited = ~visited
            unvisited = torch.cat([unvisited, unvisited], -1)
            logits = self.ptr2(enc_xy, second_query, unvisited)
            distr = Categorical(logits=logits)
            if deterministic:
                _, idxs = torch.max(logits, -1)
            else:
                idxs = distr.sample()
            log_probs.append(distr.log_prob(idxs))

            second_idx = idxs % degree
            sec_dir = torch.div(idxs, degree, rounding_mode='floor')
            fir_dir = 1 - sec_dir

            with torch.no_grad():
                x_idx = first_idx * sec_dir + second_idx * fir_dir
                y_idx = first_idx * fir_dir + second_idx * sec_dir

            indexes.append(x_idx)
            indexes.append(y_idx)

            # update visited
            visited.scatter_(1, first_idx.unsqueeze(-1), True)

            # update query
            q1 = q3
            q2 = encodings[torch.arange(encodings.size(0)), second_idx]
            qx = encodings[torch.arange(encodings.size(0)), x_idx]
            qy = encodings[torch.arange(encodings.size(0)), y_idx]


            query_b = self.relu(second_query + self.q_l4(q2))

            can_pos = self.bufferloc(query_b).squeeze(-1)
            indexes.append(can_pos)

        indexes = torch.stack(indexes, -1)
        log_probs = sum(log_probs)

        return indexes, log_probs

class Critic(nn.Module):
    def __init__(self, input_dim, device):
        super(Critic, self).__init__()
        self.device = device

        # embedder args
        self.d_input = input_dim
        self.d_model = 128

        # encoder args
        self.num_stacks = 3
        self.num_heads = 16
        self.d_k = 16
        self.d_v = 16
        self.d_inner = 512
        self.d_unit = 256

        self.crit_embedder = Embedder( self.d_input, self.d_model)
        self.crit_encoder = Encoder(self.num_stacks, self.num_heads, self.d_k, self.d_v, self.d_model, self.d_inner)
        self.glimpse = Glimpse(self.d_model, self.d_unit)
        self.critic_l1 = nn.Linear(self.d_model, self.d_unit)
        self.critic_l2 = nn.Linear(self.d_unit, 1)
        self.relu = nn.ReLU()

        self.to(device)
        self.train()

    def forward(self, inputs, deterministic=False):
        inputs_tensor = torch.tensor(inputs, dtype=torch.float).to(self.device)
        critic_encode = self.crit_encoder(self.crit_embedder(inputs_tensor), None)
        glimpse = self.glimpse(critic_encode)
        critic_inner = self.relu(self.critic_l1(glimpse))
        predictions = self.relu(self.critic_l2(critic_inner)).squeeze(-1)

        return predictions