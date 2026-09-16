import argparse
from src.config_loader import load_config
from src.sim_main import run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="simulation_output")
    parser.add_argument("--years", type=int, default=None)
    parser.add_argument("--budget", type=int, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.years:
        overrides.setdefault("simulation", {})["years"] = args.years
    if args.budget:
        overrides.setdefault("site", {})["annual_build_budget"] = args.budget

    config = load_config(args.config, overrides)
    run(config, output_dir=args.output)

if __name__ == "__main__":
    main()