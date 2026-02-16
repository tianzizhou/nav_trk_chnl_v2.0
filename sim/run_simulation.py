#!/usr/bin/env python3
# =============================================================================
# Run single scenario simulation
# Usage: python sim/run_simulation.py --scenario s1_head_on
# =============================================================================

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(
  os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import DetectionPipeline
from src.utils import load_config

# Scenario registry
SCENARIOS = {
  's1_head_on': ('scenarios.s1_head_on', 'HeadOnScenario'),
  's2_high_speed_cross': (
    'scenarios.s2_high_speed_cross', 'HighSpeedCrossScenario'
  ),
  's3_tail_chase': (
    'scenarios.s3_tail_chase', 'TailChaseScenario'
  ),
  's4_multi_target': (
    'scenarios.s4_multi_target', 'MultiTargetScenario'
  ),
  's5_maneuvering': (
    'scenarios.s5_maneuvering', 'ManeuveringScenario'
  ),
  's6_long_range': (
    'scenarios.s6_long_range', 'LongRangeScenario'
  ),
  's7_clutter': ('scenarios.s7_clutter', 'ClutterScenario'),
  's8_occlusion': (
    'scenarios.s8_occlusion', 'OcclusionScenario'
  ),
}


def load_scenario(name, duration=None, frame_rate=None):
  """Load a scenario by name."""
  if name not in SCENARIOS:
    raise ValueError(
      f"Unknown scenario '{name}'. "
      f"Available: {list(SCENARIOS.keys())}"
    )

  module_name, class_name = SCENARIOS[name]
  import importlib
  mod = importlib.import_module(module_name)
  cls = getattr(mod, class_name)

  kwargs = {}
  if duration is not None:
    kwargs['duration'] = duration
  if frame_rate is not None:
    kwargs['frame_rate'] = frame_rate

  return cls(**kwargs)


def run_single(scenario_name, config_path=None,
               duration=None, frame_rate=None,
               output_dir=None, verbose=True):
  """Run a single scenario simulation.

  Args:
    scenario_name: Scenario name string.
    config_path:   Path to config YAML.
    duration:      Override duration (seconds).
    frame_rate:    Override frame rate (Hz).
    output_dir:    Output directory path.
    verbose:       Print progress.

  Returns:
    SequenceResult: Simulation result.
  """
  if verbose:
    print(f"\n{'='*60}")
    print(f"  Running scenario: {scenario_name}")
    print(f"{'='*60}")

  scenario = load_scenario(
    scenario_name, duration, frame_rate
  )
  pipeline = DetectionPipeline(config_path=config_path)

  if verbose:
    print(f"  Duration: {scenario.duration}s, "
          f"Frames: {scenario.num_frames}, "
          f"FPS: {scenario.frame_rate}")
    print(f"  Targets: "
          f"{len(scenario.get_target_trajectories())}")
    print()

  result = pipeline.run_sequence(scenario, verbose=verbose)

  if verbose:
    print(f"\n  Results:")
    print(f"  Total time: {result.total_time_sec:.1f}s")
    print(f"  Avg frame: "
          f"{result.avg_frame_time_ms:.1f}ms")
    for key, val in result.metrics.items():
      if isinstance(val, float):
        print(f"  {key}: {val:.4f}")
      else:
        print(f"  {key}: {val}")

  # Save results if output directory specified
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
    save_results(result, output_dir)
    if verbose:
      print(f"\n  Results saved to: {output_dir}/")

  return result


def save_results(result, output_dir):
  """Save simulation results to files.

  Args:
    result:     SequenceResult.
    output_dir: Directory to save to.
  """
  # Save metrics summary
  metrics_path = os.path.join(output_dir, 'metrics.json')
  metrics_data = {
    'scenario_name': result.scenario_name,
    'num_frames': result.num_frames,
    'total_time_sec': result.total_time_sec,
    'avg_frame_time_ms': result.avg_frame_time_ms,
    'metrics': {}
  }
  for k, v in result.metrics.items():
    if isinstance(v, float) and (
        v != v or v == float('inf') or v == float('-inf')
    ):
      metrics_data['metrics'][k] = str(v)
    else:
      metrics_data['metrics'][k] = v

  with open(metrics_path, 'w') as f:
    json.dump(metrics_data, f, indent=2)

  # Save per-frame tracking data
  frames_path = os.path.join(output_dir, 'frame_data.jsonl')
  with open(frames_path, 'w') as f:
    for fr in result.frame_results:
      frame_record = {
        'frame_idx': fr.frame_idx,
        'timestamp': fr.timestamp,
        'num_detections': fr.num_detections,
        'num_confirmed_tracks': fr.num_confirmed_tracks,
        'timing_ms': fr.timing_ms,
        'reports': []
      }
      for r in fr.target_reports:
        frame_record['reports'].append({
          'track_id': r.track_id,
          'azimuth_deg': r.azimuth_deg,
          'elevation_deg': r.elevation_deg,
          'slant_range_m': r.slant_range_m,
          'velocity_mps': r.velocity_mps,
          'ttc_sec': (
            r.ttc_sec if r.ttc_sec != float('inf')
            else 'inf'
          ),
          'threat_level': r.threat_level,
        })
      f.write(json.dumps(frame_record) + '\n')


def main():
  parser = argparse.ArgumentParser(
    description='Run UAV detection simulation'
  )
  parser.add_argument(
    '--scenario', '-s', required=True,
    choices=list(SCENARIOS.keys()),
    help='Scenario to run'
  )
  parser.add_argument(
    '--config', '-c', default='config/default_config.yaml',
    help='Configuration file path'
  )
  parser.add_argument(
    '--duration', '-d', type=float, default=None,
    help='Override simulation duration (seconds)'
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

  output_dir = os.path.join(args.output, args.scenario)
  run_single(
    args.scenario,
    config_path=args.config,
    duration=args.duration,
    frame_rate=args.frame_rate,
    output_dir=output_dir,
    verbose=not args.quiet,
  )


if __name__ == '__main__':
  main()
