"""The single entry point (one config: configs/config.yaml).

    python run.py                      # real run: the optimize loop on the train split (your optimizer class)
    python run.py --mock               # no GPU: mock model presets + bundled sample data, same config
    python run.py --set max_iterations=10 --set benchmarks.gsm8k.train=100
    python run.py --set optimizer.name=opro                # the worked-example optimizer
    python run.py --eval-only --skills cot runs/<run>/rounds/001/skills/opro-i1-c0.json --split test
"""
import sys

from skillopt.cli import main

if __name__ == "__main__":
    main(sys.argv[1:])
