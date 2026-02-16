#!/usr/bin/env python3
# =============================================================================
# Batch evaluation across all scenarios
# Usage: python sim/batch_evaluation.py --all
# =============================================================================

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(
  os.path.join(os.path.dirname(__file__), '..')))

from sim.run_simulation import (
  SCENARIOS, run_single, load_scenario,
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

  if verbose:
    print(f"\n{'='*60}")
    print(f"  BATCH EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Scenarios run: {len(results)}")
    print(f"  Total time: {total_time:.1f}s")
    print()

    # Print summary table
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

  # Save batch summary
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

  if verbose:
    print(f"\n  Summary saved to: {summary_path}")

  return results


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
