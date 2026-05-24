import numpy as np
import torch
import torch.nn as nn
import os
import math

class Embedder(nn.Module):
    def __init__(self, d_input, d_model):
        super(Embedder, self).__init__()
        self.conv1d = nn.Conv1d(d_input, d_model, 1)
        self.batch_norm = nn.BatchNorm1d(d_model)

    def forward(self, inputs):
        embeddings = self.conv1d(inputs.permute(0, 2, 1))
        
        embeddings = self.batch_norm(embeddings).permute(0, 2, 1)
        return embeddings


class SlideBuffer(nn.Module):
    def __init__(self, d_model, d_query):
        super(SlideBuffer, self).__init__()
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


# class SinBuffer(nn.Module):
#     def __init__(self, d_model, d_query):
#         super(SinBuffer, self).__init__()
#         self.d_inner1 = 512
#         self.d_inner2 = 256
        
#         self.fc1 = nn.Linear(2 * d_model + d_query, self.d_inner1) 
#         self.fc2 = nn.Linear(self.d_inner1, self.d_inner2)
#         self.fc3 = nn.Linear(self.d_inner2, 1) 
        
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, q_v, q_h, query):
#         combined = torch.cat((q_v, q_h, query), dim=-1)  
#         x = self.relu(self.fc1(combined))
#         x = self.relu(self.fc2(x))
#         x = self.sigmoid(self.fc3(x))
#         x = x * (2 * math.pi) 
#         return x


class SinBuffer(nn.Module):
    def __init__(self, d_query):
        super(SinBuffer, self).__init__()
        self.fc1 = nn.Linear(d_query, d_query)
        self.tanh = nn.Tanh()
        self.h = nn.Parameter(torch.randn(d_query)) 
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, query):
        x = self.tanh(self.fc1(query))  
        x = torch.matmul(x, self.h)     
        x = self.sigmoid(x) * (2 * math.pi)
        return x
    

class Pointer(nn.Module):
    def __init__(self, d_query, d_unit):
        super(Pointer, self).__init__()
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
        super(Glimpse, self).__init__()
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
