#!/usr/bin/env python

"""
Submit LeRobot training to a remote Ray cluster.
- Supports submitting multiple concurrent jobs safely.
- Lets Ray assign GPUs per job (no hardcoded CUDA device).
- Parameterize Ray URL, resources, and run specifics via CLI args.
- Use an official PyPI release of lerobot by default (no local code upload), overridable via --lerobot-req.
- Supports passing arbitrary lerobot train.py args after a "--" separator.
"""

import argparse
import hashlib
import os
from pathlib import Path
from typing import Tuple

from ray import job_submission


def _short_hash(*parts: str, n: int = 6) -> str:
    """Deprecated: previously used for adding suffixes to run names."""
    h = hashlib.md5("_".join(parts).encode()).hexdigest()
    return h[:n]


def build_training_cmd(
    extra_train_args: list[str] | None,
) -> str:
    args = [
        "python",
        "-u",  # unbuffered for real-time logs
        "-m",
        "lerobot.scripts.train",
    ]

    # Ensure we don't accidentally try to push to the Hub without a repo id
    train_args = list(extra_train_args or [])
    if not any(a.startswith("--policy.push_to_hub=") for a in train_args):
        train_args.insert(0, "--policy.push_to_hub=false")

    if train_args:
        # Pass through arbitrary train.py arguments verbatim
        # Handle JSON arguments that need proper shell escaping
        for arg in train_args:
            if '={' in arg and '}' in arg:
                # This looks like a JSON argument, quote it properly
                key, value = arg.split('=', 1)
                args.append(f"{key}=\'{value}\'")
            else:
                args.append(arg)

    return " ".join(args)


# Removed version prelude; run training directly


def submit_training_job(
    ray_dashboard_url: str,
    cpus: int,
    gpus: float,
    lerobot_req: str,
    run_base_override: str | None,
    extra_train_args: list[str] | None,
) -> Tuple[str, job_submission.JobSubmissionClient]:
    """Submit a single training job to the Ray cluster and return (job_id, client)."""

    client = job_submission.JobSubmissionClient(ray_dashboard_url)
    print(f"Connected to Ray cluster at: {ray_dashboard_url}")

    def _extract_arg(prefixes: list[str]) -> str | None:
        if not extra_train_args:
            return None
        for arg in extra_train_args:
            for p in prefixes:
                if arg.startswith(p):
                    return arg.split("=", 1)[1].strip("'\"")
        return None

    if run_base_override:
        derived_run_base = run_base_override
    else:
        policy_id = _extract_arg(["--policy.type=", "--policy.path="])
        dataset_id = _extract_arg(["--dataset.repo_id="])
        if not policy_id or not dataset_id:
            raise ValueError(
                "Missing required training args after --. Expected both --policy.type/--policy.path and --dataset.repo_id."
            )
        derived_run_base = f"{Path(policy_id).name}_{Path(dataset_id).name}"
    run_name = derived_run_base + "alldefaults"
    print(f"WandB run name: {run_name}")

    # Ensure a job name is present in training args for reproducible naming
    train_args = list(extra_train_args or [])
    if not any(a.startswith("--job_name=") for a in train_args):
        train_args.insert(0, f"--job_name={run_name}")

    training_cmd = build_training_cmd(extra_train_args=train_args)
    entrypoint_cmd = training_cmd

    pip_packages = [
        lerobot_req,  # Expect lerobot==0.3.3 by default (base package only)
        # Pin a transformers version compatible with PI0 pretrained weights API surface (embed_tokens present)
        "transformers==4.52.0",
        # Direct env deps
        "dm_control==1.0.14",
        "mujoco==2.3.7",
        "num2words",
        # PushT env specific dep
        "pymunk==6.11.0",
        "gym_pusht",
        "gym_aloha",
    ]

    job_id = client.submit_job(
        entrypoint=entrypoint_cmd,
        runtime_env={
            "pip": pip_packages,
            "env_vars": {
                # Important: DO NOT set CUDA_VISIBLE_DEVICES. Ray sets it per allocation so concurrent jobs each see GPU 0 within their namespace.
                # Avoid setting PYTHONPATH to prevent local path shadowing the pip-installed lerobot package.
                "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
                "HF_TOKEN": os.getenv("HF_TOKEN", ""),
                "TOKENIZERS_PARALLELISM": "false",
                "MUJOCO_GL": "egl",
                "DISPLAY": "",
            },
        },
        entrypoint_num_cpus=cpus,
        entrypoint_num_gpus=gpus,
        metadata={
            "job_name": f"{run_name}",
            "description": "LeRobot training job",
        },
    )

    print("Job submitted successfully!")
    print(f"Job ID: {job_id}")
    print(f"Monitor at: {ray_dashboard_url}/jobs/{job_id}")
    return job_id, client


def monitor_job(job_id: str, client: job_submission.JobSubmissionClient) -> None:
    """Deprecated: waiting/monitoring is no longer supported via --wait."""
    pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit LeRobot training jobs to a Ray cluster")

    # Ray
    p.add_argument(
        "--ray-url",
        default="http://tarikmimic.lam-248.ray.clusters.corp.theaiinstitute.com",
        help="Ray Dashboard/Jobs API URL (protocol + host).",
    )

    # Resources
    p.add_argument("--cpus", type=int, default=32, help="CPUs per job")
    p.add_argument("--gpus", type=float, default=1.0, help="GPUs per job (can be fractional)")

    # Naming
    p.add_argument("--name", default=None, help="Optional base name override for job/run")

    # Batch submission is removed; one job per invocation
    # LeRobot requirement (version or VCS)
    p.add_argument(
        "--lerobot-req",
        default="lerobot==0.3.3",
        help="pip requirement string for lerobot (e.g., 'lerobot==0.3.3' or a VCS URL)",
    )

    # Monitoring removed (--wait)

    # Remainder: any args after "--" are forwarded verbatim to train.py
    p.add_argument("train_args", nargs=argparse.REMAINDER, help="Optional: args after -- are passed to train.py verbatim")

    args = p.parse_args()

    # Strip leading "--" in remainder if present
    if args.train_args and len(args.train_args) > 0 and args.train_args[0] == "--":
        args.train_args = args.train_args[1:]

    return args


if __name__ == "__main__":
    args = parse_args()

    job_id, _client = submit_training_job(
        ray_dashboard_url=args.ray_url.rstrip("/"),
        cpus=args.cpus,
        gpus=args.gpus,
        lerobot_req=args.lerobot_req,
        run_base_override=args.name,
        extra_train_args=args.train_args,
    )

    print(f"\nYou can monitor the job at: {args.ray_url.rstrip('/')}/jobs/{job_id}")
    print(f"ray job logs {job_id} -f")
