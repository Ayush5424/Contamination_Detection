"""Sensor feeder agent as its own process (simulated plant gateway, no CSV file).

    python sensor_publisher.py                    # one fresh reading (5 plant minutes) every second
    python sensor_publisher.py --interval 300     # true real time: one reading every 5 minutes
    python sensor_publisher.py --seed 7           # repeatable plant run

The API starts the same agent by itself (LIVE_AGENT=1, the default), so this script is only
needed when the parts run as separate processes (Start Live System.bat, Docker Compose).
See wq_system/feeder.py for the message format and the control commands.
"""
import argparse

from wq_system import config
from wq_system.feeder import FeederAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=1.0, help="real seconds between readings (300 = real time)")
    ap.add_argument("--seed", type=int, default=None, help="random seed of the simulated plant (default: random)")
    ap.add_argument("--device", default=config.DEVICE)
    args = ap.parse_args()
    FeederAgent(args.interval, args.device, args.seed, log=lambda m: print(m, flush=True)).run()


if __name__ == "__main__":
    main()
