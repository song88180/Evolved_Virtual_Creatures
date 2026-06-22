"""Generate and view a MuJoCo creature from a genotype JSON file."""

import argparse
from pathlib import Path

from evol_virtual_creature.genotype_io import load_genotype_from_json
from evol_virtual_creature.graph_analysis import PhenotypeBuildAbort
from evol_virtual_creature.phenotype import PhenotypeBuilder
from evol_virtual_creature.viewer import launch_viewer


DEFAULT_GENOTYPE_PATH = Path(__file__).with_name("example_genotype.json")
OUTPUT_XML_PATH = Path(__file__).with_name("generated_creature.xml")
MAX_N_NODES = 500


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show default values for command-line options."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and view a mutated MuJoCo creature.",
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "--genotype",
        type=Path,
        default=DEFAULT_GENOTYPE_PATH,
        help="Source genotype JSON path. (default: example_genotype.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_XML_PATH,
        help="Generated MuJoCo XML output path. (default: generated_creature.xml)",
    )
    parser.add_argument(
        "--mutations",
        type=int,
        default=50,
        help="Number of random mutations to apply.",
    )
    parser.add_argument(
        "--max-node",
        type=int,
        default=MAX_N_NODES,
        help="Maximum phenotype nodes allowed during construction.",
    )
    args = parser.parse_args()
    if args.mutations < 0:
        parser.error("--mutations must be non-negative")
    if args.max_node < 1:
        parser.error("--max-node must be at least 1")
    return args


def main():
    args = parse_args()
    genotype = load_genotype_from_json(args.genotype)

    for _ in range(args.mutations):
        genotype.mutation(num_mutations=1)

    print("Building MuJoCo organism from mutated genotype.")

    builder = PhenotypeBuilder(genotype, max_node=args.max_node)
    try:
        mjcf = builder.build()
    except PhenotypeBuildAbort as error:
        print(f"Aborting phenotype construction: {error}")
        raise SystemExit(1) from error

    args.output.write_text(mjcf)
    print(f"Generated MuJoCo model saved to {args.output}")

    launch_viewer(mjcf, builder.actuator_controllers)


if __name__ == "__main__":
    main()
