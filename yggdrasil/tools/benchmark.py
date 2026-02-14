"""Benchmark tool -- compare solver/schedule combinations.

Usage:
    python -m yggdrasil.tools.benchmark
"""
from __future__ import annotations

import time
import torch
import argparse
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    solver: str
    schedule: str
    num_steps: int
    time_seconds: float
    peak_memory_mb: float
    samples_per_second: float
    
    def __repr__(self):
        return (
            f"{self.solver:20s} | {self.schedule:15s} | "
            f"{self.num_steps:4d} steps | "
            f"{self.time_seconds:6.2f}s | "
            f"{self.peak_memory_mb:8.1f} MB | "
            f"{self.samples_per_second:6.2f} samples/s"
        )


def benchmark_sampling(
    model=None,
    solvers: List[str] = None,
    schedules: List[str] = None,
    step_counts: List[int] = None,
    shape: tuple = (1, 4, 64, 64),
    device: str = "cpu",
    num_runs: int = 3,
) -> List[BenchmarkResult]:
    """Benchmark different solver/schedule combinations.
    
    Args:
        model: Optional model (uses dummy if None)
        solvers: List of solver type keys
        schedules: List of schedule type keys  
        step_counts: List of step counts to test
        shape: Latent tensor shape
        device: Device to run on
        num_runs: Number of runs to average
        
    Returns:
        List of BenchmarkResult
    """
    from yggdrasil.core.block.builder import BlockBuilder
    from yggdrasil.core.block.registry import auto_discover
    
    auto_discover()
    
    solvers = solvers or ["ddim", "heun"]
    schedules = schedules or ["linear", "cosine"]
    step_counts = step_counts or [20, 50]
    
    results = []
    
    for solver_name in solvers:
        for sched_name in schedules:
            for num_steps in step_counts:
                # Build components
                try:
                    solver = BlockBuilder.build({"type": f"diffusion/solver/{solver_name}"})
                    schedule = BlockBuilder.build({"type": f"noise/schedule/{sched_name}"})
                except Exception as e:
                    print(f"Skip {solver_name}/{sched_name}: {e}")
                    continue
                
                # Time the sampling loop simulation
                times = []
                for _ in range(num_runs):
                    latents = torch.randn(shape, device=device)
                    timesteps = schedule.get_timesteps(num_steps)
                    
                    # Track memory
                    if device == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    
                    start = time.perf_counter()
                    
                    for i in range(len(timesteps)):
                        t = timesteps[i]
                        next_t = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0)
                        
                        # Simulate model output (random)
                        model_output = torch.randn_like(latents)
                        
                        process = BlockBuilder.build({"type": "diffusion/process/ddpm"})
                        latents = solver.step(
                            model_output=model_output,
                            current_latents=latents,
                            timestep=t.unsqueeze(0) if t.dim() == 0 else t,
                            process=process,
                            next_timestep=next_t.unsqueeze(0) if next_t.dim() == 0 else next_t,
                        )
                    
                    elapsed = time.perf_counter() - start
                    times.append(elapsed)
                
                avg_time = sum(times) / len(times)
                
                peak_mem = 0.0
                if device == "cuda":
                    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
                
                result = BenchmarkResult(
                    solver=solver_name,
                    schedule=sched_name,
                    num_steps=num_steps,
                    time_seconds=avg_time,
                    peak_memory_mb=peak_mem,
                    samples_per_second=1.0 / avg_time if avg_time > 0 else 0,
                )
                results.append(result)
                print(f"  {result}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="YggDrasil Benchmark")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda/mps)")
    parser.add_argument("--steps", nargs="+", type=int, default=[20, 50])
    parser.add_argument("--runs", type=int, default=3)
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"  YggDrasil Sampling Benchmark (device={args.device})")
    print(f"{'='*80}\n")
    
    results = benchmark_sampling(
        device=args.device,
        step_counts=args.steps,
        num_runs=args.runs,
    )
    
    print(f"\n{'='*80}")
    print(f"  Done! {len(results)} configurations tested.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
