# CLI helper for demo: optional synthetic data generation + Flask server
# Usage: python scripts/run_demo.py --generate --items 500 --users 200 --events 5000
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GEN = ROOT / 'scripts' / 'generate_synthetic_sample.py'
APP = ROOT / 'app.py'

parser = argparse.ArgumentParser()
parser.add_argument('--generate', action='store_true', help='Generate synthetic sample files before running')
parser.add_argument('--items', type=int, default=500)
parser.add_argument('--users', type=int, default=200)
parser.add_argument('--events', type=int, default=5000)
args = parser.parse_args()

if args.generate:
    cmd = [sys.executable, str(GEN)]
    print('Generating synthetic data...')
    subprocess.check_call(cmd)

print('Starting Flask demo (app.py).')
subprocess.check_call([sys.executable, str(APP)])
