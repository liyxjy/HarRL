import numpy as np
import torch
import os
import time
import math
from models2.actor_critic import Actor, Critic
from utils2.rsmt_utils import Evaluator
from utils2.log_utils import *
import argparse
from utils.plot_curves import plot_curve
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--degree', type=int, default=10, help='maximum degree of nets')
parser.add_argument('--capacitance_range', type=list, default=[0.1, 0.9], help='range of capacitance')
parser.add_argument('--dimension', type=int, default=3, help='terminal representation dimension')
parser.add_argument('--batch_size', type=int, default=1024, help='test batch size')
parser.add_argument('--start_batch', type=int, default=1, help='start_batch')
parser.add_argument('--base_degree', type=int, default=10, help='train on')
parser.add_argument('--eval_size', type=int, default=2000, help='eval set size')
parser.add_argument('--num_batches', type=int, default=50000, help='number of batches')
parser.add_argument('--seed', type=int, default=9, help='random seed')
# Optimizer
parser.add_argument('--learning_rate', type=float, default=0.00005)

args = parser.parse_args()

start_batch = args.start_batch

model_method = 'sinModel'
file_dir = 'saved_model/' + model_method
model_trained_on = file_dir + '/' + str(args.base_degree) + 'b.pt'

log_intvl = 100
device = torch.device("cuda:0")
start_time = time.time()

log_dir = 'logs/' + model_method + '/' + str(args.degree)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)

ckp_dir = file_dir + '/' + str(args.degree) + '.pt'
best_ckp_dir = file_dir + '/' + str(args.degree) + 'b.pt'
if not os.path.exists(file_dir):
    os.makedirs(file_dir)

best_eval = 10.
best_kept = 0

actor = Actor(args.dimension, device)
critic = Critic(args.dimension, device)
mse_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.learning_rate, eps=1e-5)
evaluator = Evaluator(args.degree)

np.random.seed(args.seed)
torch.manual_seed(args.seed)


# Evaluate on cases generated in the same way as the training data
eval_x_coords = np.random.rand(args.eval_size, args.degree, 2)  # coordinates
eval_capacitance = (np.random.rand(args.eval_size, args.degree, 1) *
                    (args.capacitance_range[1] - args.capacitance_range[0]) + args.capacitance_range[0])  # capacitance
eval_cases = np.concatenate((eval_x_coords, eval_capacitance), axis=-1)


if os.path.exists(model_trained_on):
    checkpoint = torch.load(model_trained_on)
    print("Checkpoint exists. Loading model ", model_trained_on, ".")
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])


"Trainging Loop"
for batch_idx in tqdm(range(start_batch, start_batch + args.num_batches), desc="Training"):
    actor.train()
    critic.train()

    x_coords = np.random.rand(args.batch_size, args.degree, 2)  # coordinates
    capacitance = (np.random.rand(args.batch_size, args.degree, 1) *
                   (args.capacitance_range[1] - args.capacitance_range[0]) + args.capacitance_range[0])  # capacitance
    input_batch = np.concatenate((x_coords, capacitance), axis=-1)

    outputs, log_probs = actor(input_batch)
    predictions = critic(input_batch)

    lengths = evaluator.eval_batch(input_batch, outputs.cpu().detach().numpy(), args.degree)
    length_tensor = torch.tensor(lengths, dtype=torch.float).to(device)

    with torch.no_grad():
        disadvantage = length_tensor - predictions
    actor_loss = torch.mean(disadvantage * log_probs)
    critic_loss = mse_loss(predictions, length_tensor)
    loss = actor_loss + critic_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.)
    optimizer.step()

    if batch_idx % log_intvl == 0:
        #print('[batch', str(batch_idx) + ',', 'time', str(int(time.time() - start_time)) + 's]')
        print('loss', lengths.mean())
        actor.eval()
        eval_lengths = []
        for eval_idx in range(math.ceil(args.eval_size / args.batch_size)):
            eval_batch = eval_cases[args.batch_size * eval_idx: args.batch_size * (eval_idx + 1)]
            with torch.no_grad():
                outputs, _ = actor(eval_batch, True)
            eval_lengths.append(evaluator.eval_batch(eval_batch, outputs.cpu().detach().numpy(), args.degree))
        eval_mean = np.concatenate(eval_lengths, -1).mean()
        
        if eval_mean < best_eval:
            best_eval = eval_mean
            best_kept = 0
            torch.save({
                'batch_idx': batch_idx,
                'best_eval': best_eval,
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, best_ckp_dir)
            print('ckpt saved at', best_ckp_dir)
        else:
            best_kept += 1
            
        print('eval', eval_mean)
        print('best', best_eval, '(' + str(best_kept) + ')')

        writer.add_scalar('Loss/total', loss.item(), batch_idx)
        writer.add_scalar('Loss/actor', actor_loss.item(), batch_idx)
        writer.add_scalar('Loss/critic', critic_loss.item(), batch_idx)
        writer.add_scalar('Eval/mean', eval_mean, batch_idx)
        writer.add_scalar('Eval/best', best_eval, batch_idx)

torch.save({
    'batch_idx': batch_idx,
    'best_eval': best_eval,
    'actor_state_dict': actor.state_dict(),
    'critic_state_dict': critic.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, ckp_dir)
print('ckpt saved at', ckp_dir)

writer.close()
