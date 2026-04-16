#!/usr/bin/env python3
"""
experiment_logger.py
Experiment case logging and data management for tank test.

Records metadata for each experimental case:
  - Case ID, wave parameters, LiDAR settings
  - File paths for PCD data and wave probe CSV
  - Start/end times, notes
  - Results (when processed)

Usage:
    # Log a new case
    python scripts/experiment_logger.py log \
        --case_id "R1_01" \
        --Hs 0.05 --Tp 1.0 \
        --description "Regular wave, T=1.0s, H=0.05m" \
        --data_dir "experiment/R1_01/" \
        --probe_file "experiment/R1_01/wave_probe.csv"

    # List all cases
    python scripts/experiment_logger.py list

    # Export summary
    python scripts/experiment_logger.py summary --output results/experiment_summary.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional


DEFAULT_LOG_FILE = 'results/experiment_log.json'


def load_log(filepath: str) -> Dict:
    """Load experiment log from JSON."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {'cases': [], 'metadata': {'created': datetime.now().isoformat()}}


def save_log(log: Dict, filepath: str):
    """Save experiment log to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    log['metadata']['last_modified'] = datetime.now().isoformat()
    with open(filepath, 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def log_case(filepath: str,
             case_id: str,
             Hs: float,
             Tp: float,
             description: str = '',
             data_dir: str = '',
             probe_file: str = '',
             wave_type: str = 'regular',
             depth: float = 3.5,
             notes: str = ''):
    """Log a new experimental case."""
    log = load_log(filepath)

    # Check for duplicate
    for case in log['cases']:
        if case['case_id'] == case_id:
            print(f"Warning: case_id '{case_id}' already exists, updating.")
            case.update({
                'Hs': Hs, 'Tp': Tp,
                'description': description,
                'data_dir': data_dir,
                'probe_file': probe_file,
                'wave_type': wave_type,
                'depth': depth,
                'notes': notes,
                'updated': datetime.now().isoformat(),
            })
            save_log(log, filepath)
            return

    case = {
        'case_id': case_id,
        'Hs': Hs,
        'Tp': Tp,
        'wave_type': wave_type,
        'depth': depth,
        'description': description,
        'data_dir': data_dir,
        'probe_file': probe_file,
        'notes': notes,
        'logged': datetime.now().isoformat(),
        'status': 'recorded',
        'results': None,
    }

    log['cases'].append(case)
    save_log(log, filepath)
    print(f"Logged case: {case_id}")


def list_cases(filepath: str):
    """Print summary of all logged cases."""
    log = load_log(filepath)
    cases = log['cases']

    if not cases:
        print("No cases logged yet.")
        return

    print(f"{'ID':<10} {'Type':<12} {'Hs[m]':>7} {'Tp[s]':>7} {'Status':<12} {'Description'}")
    print('-' * 80)
    for c in cases:
        print(f"{c['case_id']:<10} {c.get('wave_type','?'):<12} "
              f"{c['Hs']:>7.3f} {c['Tp']:>7.2f} "
              f"{c.get('status','?'):<12} {c.get('description','')}")


def update_results(filepath: str, case_id: str, results: Dict):
    """Update results for a case after processing."""
    log = load_log(filepath)
    for case in log['cases']:
        if case['case_id'] == case_id:
            case['results'] = results
            case['status'] = 'processed'
            case['processed'] = datetime.now().isoformat()
            save_log(log, filepath)
            print(f"Updated results for {case_id}")
            return
    print(f"Case {case_id} not found.")


def export_summary(filepath: str, output: str):
    """Export summary with results to JSON."""
    log = load_log(filepath)
    summary = {
        'n_cases': len(log['cases']),
        'n_processed': sum(1 for c in log['cases'] if c.get('status') == 'processed'),
        'cases': log['cases'],
    }
    with open(output, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary exported to {output}")


def main():
    parser = argparse.ArgumentParser(description='Experiment case logger')
    parser.add_argument('--log_file', default=DEFAULT_LOG_FILE,
                        help='Path to experiment log JSON')
    sub = parser.add_subparsers(dest='command')

    # log
    p_log = sub.add_parser('log', help='Log a new case')
    p_log.add_argument('--case_id', required=True)
    p_log.add_argument('--Hs', type=float, required=True)
    p_log.add_argument('--Tp', type=float, required=True)
    p_log.add_argument('--description', default='')
    p_log.add_argument('--data_dir', default='')
    p_log.add_argument('--probe_file', default='')
    p_log.add_argument('--wave_type', default='regular')
    p_log.add_argument('--depth', type=float, default=3.5)
    p_log.add_argument('--notes', default='')

    # list
    sub.add_parser('list', help='List all cases')

    # summary
    p_sum = sub.add_parser('summary', help='Export summary')
    p_sum.add_argument('--output', default='results/experiment_summary.json')

    args = parser.parse_args()

    if args.command == 'log':
        log_case(args.log_file, args.case_id, args.Hs, args.Tp,
                 args.description, args.data_dir, args.probe_file,
                 args.wave_type, args.depth, args.notes)
    elif args.command == 'list':
        list_cases(args.log_file)
    elif args.command == 'summary':
        export_summary(args.log_file, args.output)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
