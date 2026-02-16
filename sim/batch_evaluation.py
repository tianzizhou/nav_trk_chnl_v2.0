#!/usr/bin/env python3
# =============================================================================
# Batch evaluation across all scenarios + Monte Carlo support
# Usage:
#   python sim/batch_evaluation.py --all
#   python sim/batch_evaluation.py --all --monte-carlo 100
# =============================================================================

import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(
  os.path.join(os.path.dirname(__file__), '..')))

from sim.run_simulation import (
  SCENARIOS, run_single, load_scenario, save_results,
)
from src.pipeline import DetectionPipeline
from src.utils import load_config

ALL_SCENARIOS = list(SCENARIOS.keys())


def batch_evaluate(scenario_names, config_path=None,
                   duration=None, frame_rate=None,
                   output_dir='results', verbose=True):
  """Run batch evaluation across multiple scenarios.

  Args:
    scenario_names: List of scenario name strings.
    config_path:    Config file path.
    duration:       Override duration.
    frame_rate:     Override frame rate.
    output_dir:     Output directory.
    verbose:        Print progress.

  Returns:
    dict: {scenario_name: SequenceResult}
  """
  results = {}
  total_start = time.perf_counter()

  for i, name in enumerate(scenario_names):
    if verbose:
      print(f"\n[{i+1}/{len(scenario_names)}] "
            f"Running: {name}")

    sc_output = os.path.join(output_dir, name)
    result = run_single(
      name,
      config_path=config_path,
      duration=duration,
      frame_rate=frame_rate,
      output_dir=sc_output,
      verbose=verbose,
    )
    results[name] = result

  total_time = time.perf_counter() - total_start

  _print_summary(results, total_time, verbose)
  _save_summary(results, total_time, output_dir)

  return results


def monte_carlo_evaluate(scenario_names, num_runs=100,
                         config_path=None, duration=None,
                         frame_rate=None, output_dir='results',
                         verbose=True):
  """Run Monte Carlo evaluation with varying random seeds.

  Args:
    scenario_names: List of scenario name strings.
    num_runs:       Number of MC runs per scenario.
    config_path:    Config file path.
    duration:       Override duration.
    frame_rate:     Override frame rate.
    output_dir:     Output directory.
    verbose:        Print progress.

  Returns:
    dict: {scenario_name: {metric_name: [values_per_run]}}
  """
  config_path = config_path or "config/default_config.yaml"
  config = load_config(config_path)

  mc_results = {}
  total_start = time.perf_counter()

  for sc_name in scenario_names:
    if verbose:
      print(f"\n{'='*60}")
      print(f"  Monte Carlo: {sc_name} ({num_runs} runs)")
      print(f"{'='*60}")

    scenario = load_scenario(sc_name, duration, frame_rate)
    run_metrics = []

    for run_idx in range(num_runs):
      # Vary the random seed for each run
      run_seed = 42 + run_idx
      config['simulation']['random_seed'] = run_seed

      pipeline = DetectionPipeline(
        config=config, skip_image_render=True
      )
      result = pipeline.run_sequence(scenario, verbose=False)
      run_metrics.append(result.metrics)

      if verbose and (run_idx + 1) % 10 == 0:
        dr = result.metrics.get('detection_rate', 0)
        print(f"  Run {run_idx+1}/{num_runs}: "
              f"Pd={dr:.1%}")

    # Aggregate statistics
    mc_results[sc_name] = _aggregate_mc(run_metrics)

    if verbose:
      agg = mc_results[sc_name]
      print(f"\n  {sc_name} MC Summary ({num_runs} runs):")
      for metric, stats in agg.items():
        if 'mean' in stats:
          print(f"    {metric}: "
                f"mean={stats['mean']:.4f}, "
                f"std={stats['std']:.4f}, "
                f"CI95=[{stats['ci95_lo']:.4f}, "
                f"{stats['ci95_hi']:.4f}]")

  total_time = time.perf_counter() - total_start

  # Save MC results
  mc_output_path = os.path.join(output_dir, 'monte_carlo.json')
  os.makedirs(output_dir, exist_ok=True)

  mc_save = {
    'num_runs': num_runs,
    'total_time_sec': total_time,
    'scenarios': {}
  }
  for sc_name, agg in mc_results.items():
    mc_save['scenarios'][sc_name] = {}
    for metric, stats in agg.items():
      clean = {}
      for k, v in stats.items():
        if isinstance(v, float) and (v != v):
          clean[k] = 'nan'
        else:
          clean[k] = v
      mc_save['scenarios'][sc_name][metric] = clean

  with open(mc_output_path, 'w') as f:
    json.dump(mc_save, f, indent=2)

  if verbose:
    print(f"\n  MC results saved to: {mc_output_path}")
    print(f"  Total MC time: {total_time:.1f}s")

  return mc_results


def _aggregate_mc(run_metrics_list):
  """Aggregate metrics across MC runs.

  Args:
    run_metrics_list: List of metric dicts (one per run).

  Returns:
    dict: {metric_name: {mean, std, min, max, ci95_lo, ci95_hi}}
  """
  if not run_metrics_list:
    return {}

  # Collect all metric keys
  all_keys = set()
  for m in run_metrics_list:
    all_keys.update(m.keys())

  aggregated = {}
  for key in all_keys:
    values = []
    for m in run_metrics_list:
      v = m.get(key)
      if v is not None and isinstance(v, (int, float)):
        if v == v and v != float('inf'):  # not NaN/Inf
          values.append(float(v))

    if len(values) >= 2:
      arr = np.array(values)
      n = len(arr)
      mean = float(np.mean(arr))
      std = float(np.std(arr, ddof=1))
      ci_half = 1.96 * std / np.sqrt(n)
      aggregated[key] = {
        'mean': mean,
        'std': std,
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'ci95_lo': mean - ci_half,
        'ci95_hi': mean + ci_half,
        'n_valid': n,
      }
    elif len(values) == 1:
      aggregated[key] = {
        'mean': values[0],
        'std': 0.0,
        'min': values[0],
        'max': values[0],
        'ci95_lo': values[0],
        'ci95_hi': values[0],
        'n_valid': 1,
      }

  return aggregated


def _print_summary(results, total_time, verbose):
  """Print batch evaluation summary table."""
  if not verbose:
    return

  print(f"\n{'='*60}")
  print(f"  BATCH EVALUATION SUMMARY")
  print(f"{'='*60}")
  print(f"  Scenarios run: {len(results)}")
  print(f"  Total time: {total_time:.1f}s")
  print()

  print(f"  {'Scenario':<25} {'DetRate':>8} "
        f"{'FAR':>8} {'RMSE(m)':>9} "
        f"{'AvgTime':>9}")
  print(f"  {'-'*25} {'-'*8} {'-'*8} "
        f"{'-'*9} {'-'*9}")

  for name, result in results.items():
    m = result.metrics
    dr = m.get('detection_rate', 0)
    far = m.get('false_alarm_rate', 0)
    rmse = m.get('position_rmse_m', float('nan'))
    avg_t = m.get('avg_frame_time_ms', 0)

    rmse_str = (
      f"{rmse:>9.2f}" if rmse == rmse else "     N/A"
    )

    print(f"  {name:<25} {dr:>7.1%} "
          f"{far:>8.3f} {rmse_str} "
          f"{avg_t:>8.1f}ms")


def _save_summary(results, total_time, output_dir):
  """Save batch summary JSON."""
  summary_path = os.path.join(output_dir, 'batch_summary.json')
  os.makedirs(output_dir, exist_ok=True)
  summary = {
    'total_time_sec': total_time,
    'num_scenarios': len(results),
    'scenarios': {}
  }
  for name, result in results.items():
    m = result.metrics
    clean_metrics = {}
    for k, v in m.items():
      if isinstance(v, float) and (
          v != v or v == float('inf')
      ):
        clean_metrics[k] = str(v)
      else:
        clean_metrics[k] = v
    summary['scenarios'][name] = {
      'num_frames': result.num_frames,
      'total_time_sec': result.total_time_sec,
      'metrics': clean_metrics,
    }

  with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)


def main():
  parser = argparse.ArgumentParser(
    description='Batch evaluation of UAV detection scenarios'
  )
  parser.add_argument(
    '--all', action='store_true',
    help='Run all 8 scenarios'
  )
  parser.add_argument(
    '--scenarios', '-s', nargs='+',
    choices=ALL_SCENARIOS,
    help='Specific scenarios to run'
  )
  parser.add_argument(
    '--monte-carlo', '-m', type=int, default=0,
    help='Number of Monte Carlo runs (0=disabled)'
  )
  parser.add_argument(
    '--config', '-c', default='config/default_config.yaml',
    help='Configuration file path'
  )
  parser.add_argument(
    '--duration', '-d', type=float, default=None,
    help='Override duration (seconds)'
  )
  parser.add_argument(
    '--frame-rate', '-f', type=int, default=None,
    help='Override frame rate (Hz)'
  )
  parser.add_argument(
    '--output', '-o', default='results',
    help='Output directory'
  )
  parser.add_argument(
    '--quiet', '-q', action='store_true',
    help='Suppress verbose output'
  )

  args = parser.parse_args()

  if args.all:
    scenarios = ALL_SCENARIOS
  elif args.scenarios:
    scenarios = args.scenarios
  else:
    parser.error("Specify --all or --scenarios")

  if args.monte_carlo > 0:
    monte_carlo_evaluate(
      scenarios,
      num_runs=args.monte_carlo,
      config_path=args.config,
      duration=args.duration,
      frame_rate=args.frame_rate,
      output_dir=args.output,
      verbose=not args.quiet,
    )
  else:
    batch_evaluate(
      scenarios,
      config_path=args.config,
      duration=args.duration,
      frame_rate=args.frame_rate,
      output_dir=args.output,
      verbose=not args.quiet,
    )


if __name__ == '__main__':
  main()
