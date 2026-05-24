## Layout
- models2/
  - actor_critic(_v1).py — actor and critic architectures
  - self_attn.py — encoder / attention modules
  - utils.py — model helper utilities
- utils2/
  - rsmt_utils(_ob).py — evaluation utilities (interfaces to Steiner evaluation)
  - log_utils.py — logging helpers

## Requirements
- Python 3.8+
- PyTorch
- NumPy
- tqdm
- TensorBoard

## Quick start
```bash
python train.py --degree 10 --batch_size 1024 --num_batches 50000 --learning_rate 5e-5
```
