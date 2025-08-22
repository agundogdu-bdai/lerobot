## Benchmarks: How to Run

### Prerequisites
- Ensure `WANDB_API_KEY` is set or you are logged in.

### Setup (local)
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git fetch --all --tags
git checkout release-0.3.3

python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install "ray[default]" wandb
```

Note: Workers on the Ray cluster will install `lerobot==0.3.3` via the `--lerobot-req` flag in the commands below.

### Run Experiments
Only Ray submission flags before `--`. All training/policy config after `--`.

#### exp-1: PushT with SmolVLA

With default config
```bash
python ray_submit_remote.py --ray-url http://tarikmimic.lam-248.ray.clusters.corp.theaiinstitute.com --cpus 32 --gpus 1 --lerobot-req 'lerobot==0.3.3' -- --policy.type=smolvla --dataset.repo_id=lerobot/pusht --env.type=pusht --env.task=PushT-v0 --batch_size=64 --steps=50000 --seed=1000 --wandb.enable=true --wandb.entity=bdaii --wandb.project=lerobot-vpl-benchmarks-v2 --eval_freq=2000 --save_freq=2000
```

Updated config
```bash
python ray_submit_remote.py --ray-url http://tarikmimic.lam-248.ray.clusters.corp.theaiinstitute.com --cpus 32 --gpus 1 --lerobot-req 'lerobot==0.3.3' -- --policy.type=smolvla --dataset.repo_id=lerobot/pusht --env.type=pusht --env.task=PushT-v0 --batch_size=64 --steps=50000 --seed=1000 --wandb.enable=true --wandb.entity=bdaii --wandb.project=lerobot-vpl-benchmarks-v2 --eval_freq=2000 --save_freq=2000 --policy.n_action_steps=8 --policy.chunk_size=8 --policy.max_action_dim=2 --policy.freeze_vision_encoder=false --policy.load_vlm_weights=true --policy.train_expert_only=false --policy.scheduler_decay_steps=200000 --policy.push_to_hub=false --dataset.use_imagenet_stats=false --dataset.image_transforms.enable=false --wandb.disable_artifact=false
```

#### exp-2: Aloha Insertion with SmolVLA

With default config
```bash
python ray_submit_remote.py --ray-url http://tarikmimic.lam-248.ray.clusters.corp.theaiinstitute.com --cpus 32 --gpus 1 --lerobot-req 'lerobot==0.3.3' -- --policy.type=smolvla --dataset.repo_id=lerobot/aloha_sim_insertion_human_image --env.type=aloha --env.task=AlohaInsertion-v0 --batch_size=64 --steps=50000 --seed=1000 --wandb.enable=true --wandb.entity=bdaii --wandb.project=lerobot-vpl-benchmarks-v2 --policy.push_to_hub=false --eval_freq=2000 --save_freq=2000
```

Updated config
```bash
python ray_submit_remote.py --ray-url http://tarikmimic.lam-248.ray.clusters.corp.theaiinstitute.com --cpus 32 --gpus 1 --lerobot-req 'lerobot==0.3.3' -- --policy.type=smolvla --dataset.repo_id=lerobot/aloha_sim_insertion_human_image --env.type=aloha --env.task=AlohaInsertion-v0 --batch_size=64 --steps=50000 --seed=1000 --wandb.enable=true --wandb.entity=bdaii --wandb.project=lerobot-vpl-benchmarks-v2 --policy.push_to_hub=false --eval_freq=2000 --save_freq=2000 --policy.scheduler_decay_steps=200000 --policy.n_action_steps=100 --policy.chunk_size=100 --policy.max_action_dim=14 --policy.max_state_dim=14 --policy.freeze_vision_encoder=false --policy.load_vlm_weights=true --policy.train_expert_only=false '--policy.normalization_mapping={"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MIN_MAX"}' --dataset.use_imagenet_stats=true --dataset.image_transforms.enable=false
```

#### exp-3: Aloha Transfer Cube with SmolVLA

With default config
```bash
python ray_submit_remote.py --ray-url http://tarikmimic.lam-248.ray.clusters.corp.theaiinstitute.com --cpus 32 --gpus 1 --lerobot-req 'lerobot==0.3.3' -- --policy.type=smolvla --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human_image --env.type=aloha --env.task=AlohaTransferCube-v0 --batch_size=64 --steps=50000 --seed=1000 --wandb.enable=true --wandb.entity=bdaii --wandb.project=lerobot-vpl-benchmarks-v2 --policy.push_to_hub=false --eval_freq=2000 --save_freq=2000
```

Updated config
```bash
python ray_submit_remote.py --ray-url http://tarikmimic.lam-248.ray.clusters.corp.theaiinstitute.com --cpus 32 --gpus 1 --lerobot-req 'lerobot==0.3.3' -- --policy.type=smolvla --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human_image --env.type=aloha --env.task=AlohaTransferCube-v0 --batch_size=64 --steps=50000 --seed=1000 --wandb.enable=true --wandb.entity=bdaii --wandb.project=lerobot-vpl-benchmarks-v2 --policy.push_to_hub=false --eval_freq=2000 --save_freq=2000 --policy.scheduler_decay_steps=200000 --policy.n_action_steps=100 --policy.chunk_size=100 --policy.max_action_dim=14 --policy.max_state_dim=14 --policy.freeze_vision_encoder=false --policy.load_vlm_weights=true --policy.train_expert_only=false '--policy.normalization_mapping={"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MIN_MAX"}' --dataset.use_imagenet_stats=true --dataset.image_transforms.enable=false
```

#### exp-4: Aloha Transfer Cube with Diffusion
Using defaults
```bash
python ray_submit_remote.py --ray-url http://tarikmimic.lam-248.ray.clusters.corp.theaiinstitute.com --cpus 32 --gpus 1 --lerobot-req 'lerobot==0.3.3' -- --policy.type=diffusion --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human_image --env.type=aloha --env.task=AlohaTransferCube-v0 --batch_size=64 --steps=50000 --seed=1000 --wandb.enable=true --wandb.entity=bdaii --wandb.project=lerobot-vpl-benchmarks-v2 --policy.push_to_hub=false
```

#### exp-5: Aloha Transfer Cube with Pi0
Using defaults
```bash
python ray_submit_remote.py --ray-url http://tarikmimic.lam-248.ray.clusters.corp.theaiinstitute.com --cpus 32 --gpus 1 --lerobot-req 'lerobot==0.3.3' -- --policy.path=lerobot/pi0 --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human_image --env.type=aloha --env.task=AlohaTransferCube-v0 --batch_size=16 --steps=50000 --seed=1000 --wandb.enable=true --wandb.entity=bdaii --wandb.project=lerobot-vpl-benchmarks-v2 --policy.push_to_hub=false
```
