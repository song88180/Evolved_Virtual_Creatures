"""Generate and view a MuJoCo creature from a genotype JSON file."""

from pathlib import Path

from evol_virtual_creature.genotype_io import load_genotype_from_json
from evol_virtual_creature.graph_analysis import PhenotypeBuildAbort
from evol_virtual_creature.phenotype import PhenotypeBuilder
from evol_virtual_creature.viewer import launch_viewer


DEFAULT_GENOTYPE_PATH = Path(__file__).with_name("example_genotype.json")
OUTPUT_XML_PATH = Path(__file__).with_name("generated_creature.xml")
MAX_N_NODES = 500


def main():
    genotype = load_genotype_from_json(DEFAULT_GENOTYPE_PATH)
    
    for i in range(50):
        genotype.mutation(num_mutations=1)
    
    print("Building MuJoCo organism from mutated genotype.")

    builder = PhenotypeBuilder(genotype, max_node=MAX_N_NODES)
    try:
        mjcf = builder.build()
    except PhenotypeBuildAbort as error:
        print(f"Aborting phenotype construction: {error}")
        raise SystemExit(1) from error

    OUTPUT_XML_PATH.write_text(mjcf)
    print(f"Generated MuJoCo model saved to {OUTPUT_XML_PATH.name}")

    launch_viewer(mjcf, builder.actuator_controllers)


if __name__ == "__main__":
    main()
