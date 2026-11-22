"""Tests for the graph brain: neuron functions, genes, compilation, runtime, mutation, and I/O."""

from __future__ import annotations

import copy
import json
import math
import random

import mujoco
import numpy as np
import pytest

from evc3d.brain_functions import (
    NEURON_ARITY,
    NEURON_FUNCTIONS,
    SIGNAL_LIMIT,
    evaluate_inputs,
    initial_state,
)
from evc3d.brain_genes import (
    BrainEffector,
    BrainGene,
    BrainInput,
    BrainNeuron,
    format_source,
    parse_source,
)
from evc3d.control import BrainRuntime, apply_controller, create_brain_runtime
from evc3d.evaluation import evaluate_task
from evc3d.genes import CONTROL_MODES, ConnectionGene, NodeGene
from evc3d.genotype_io import (
    convert_control_mode,
    genotype_to_dict,
    load_genotype_from_json,
    save_genotype_to_json,
)
from evc3d.phenotype import PhenotypeBuilder
from evc3d.simulation import start_episode
from helpers import make_definition, tree_genotype, tree_node

GRAPH_EXAMPLE = "examples/example_genotype_graph.json"
DT = 0.01


def run(function, *inputs, steps=1, state=None, dt=DT):
    """Step one neuron ``steps`` times and return the outputs."""
    state = initial_state(function) if state is None else state
    return [evaluate_inputs(function, inputs, state, dt) for _ in range(steps)]


# --- neuron functions ------------------------------------------------------


def test_function_set_matches_the_papers():
    assert len(NEURON_FUNCTIONS) == 23
    assert set(NEURON_FUNCTIONS) == {
        "sum", "product", "divide", "sum-threshold", "greater-than", "sign-of",
        "min", "max", "abs", "if", "interpolate", "sin", "cos", "atan", "log",
        "expt", "sigmoid", "integrate", "differentiate", "smooth", "memory",
        "oscillate-wave", "oscillate-saw",
    }
    assert all(1 <= arity <= 3 for arity in NEURON_ARITY.values())


@pytest.mark.parametrize(
    "function, inputs, expected",
    [
        ("sum", (1.0, 2.0, 3.0), 6.0),
        ("product", (3.0, -2.0, 99.0), -6.0),
        ("divide", (6.0, 3.0, 0.0), 2.0),
        ("divide", (6.0, 0.0, 0.0), 0.0),
        ("sum-threshold", (1.0, 1.0, 1.5), 1.0),
        ("sum-threshold", (1.0, 1.0, 2.5), 0.0),
        ("greater-than", (2.0, 1.0, 0.0), 1.0),
        ("greater-than", (1.0, 2.0, 0.0), 0.0),
        ("sign-of", (-0.3, 0.0, 0.0), -1.0),
        ("sign-of", (0.0, 0.0, 0.0), 0.0),
        ("min", (2.0, -1.0, 0.0), -1.0),
        ("max", (2.0, -1.0, 0.0), 2.0),
        ("abs", (-4.0, 0.0, 0.0), 4.0),
        ("if", (1.0, 7.0, 9.0), 7.0),
        ("if", (-1.0, 7.0, 9.0), 9.0),
        ("interpolate", (0.0, 10.0, 0.25), 2.5),
        ("interpolate", (0.0, 10.0, 5.0), 10.0),
        ("sin", (math.pi / 2, 0.0, 0.0), 1.0),
        ("cos", (0.0, 0.0, 0.0), 1.0),
        ("atan", (1.0, 0.0, 0.0), math.pi / 4),
        ("log", (math.e, 0.0, 0.0), 1.0),
        ("expt", (3.0, 2.0, 0.0), 9.0),
        ("sigmoid", (0.0, 0.0, 0.0), 0.5),
    ],
)
def test_stateless_functions(function, inputs, expected):
    assert run(function, *inputs)[0] == pytest.approx(expected)


def test_edge_cases_stay_finite_and_bounded():
    for function in NEURON_FUNCTIONS:
        for inputs in [(0.0,) * 3, (1e9,) * 3, (-1e9,) * 3, (1e-12,) * 3]:
            outputs = run(function, *inputs, steps=3)
            assert all(math.isfinite(v) and abs(v) <= SIGNAL_LIMIT for v in outputs)
    assert run("log", 0.0)[0] == pytest.approx(math.log(1e-6))


def test_integrate_accumulates_and_differentiate_measures_change():
    assert run("integrate", 2.0, steps=3) == pytest.approx([0.02, 0.04, 0.06])
    state = initial_state("differentiate")
    first = evaluate_inputs("differentiate", (1.0,), state, DT)
    second = evaluate_inputs("differentiate", (1.5,), state, DT)
    assert first == 0.0  # no previous sample yet
    assert second == pytest.approx(50.0)


def test_smooth_memory_and_oscillators():
    smooth = run("smooth", 1.0, 0.5, steps=3)
    assert smooth == pytest.approx([0.5, 0.75, 0.875])

    state = initial_state("memory")
    stored = evaluate_inputs("memory", (3.0, 1.0), state, DT)
    held = evaluate_inputs("memory", (9.0, 0.0), state, DT)
    assert (stored, held) == (3.0, 3.0)

    # Angular frequency 2*pi rad/s sampled at 100 Hz: a quarter wave at step 25.
    wave = run("oscillate-wave", 2 * math.pi, 0.0, steps=25)
    assert wave[-1] == pytest.approx(1.0, abs=1e-6)
    shifted = run("oscillate-wave", 0.0, math.pi / 2)
    assert shifted[0] == pytest.approx(1.0)

    saw = run("oscillate-saw", 2 * math.pi, 0.0, steps=100)
    assert min(saw) >= -1.0 and max(saw) < 1.0
    assert saw[49] == pytest.approx(0.0, abs=0.05)


def test_unknown_function_is_rejected():
    with pytest.raises(ValueError):
        evaluate_inputs("nope", (1.0,), [], DT)


# --- genes -------------------------------------------------------------------


def test_parse_source_accepts_all_scopes_and_items():
    assert parse_source("self:n0").kind == "neuron"
    assert parse_source("parent:angle2").item == "2"
    contact = parse_source("child:left_leg:contact-z")
    assert contact.scope_arg == "left_leg" and contact.item == "-z"
    assert parse_source("node:Torso:n3").scope_arg == "Torso"
    assert parse_source("global:clock").scope == "global"
    assert format_source("child", "n1", "conn") == "child:conn:n1"


@pytest.mark.parametrize(
    "bad",
    ["n0", "bogus:n0", "self:angle3", "self:contact+w", "child:n0", "self:x:n0", "self:0bad"],
)
def test_parse_source_rejects_malformed_sources(bad):
    with pytest.raises(ValueError):
        parse_source(bad)


def test_neuron_validation_and_padding():
    neuron = BrainNeuron("n0", "sum", [{"source": "self:angle0", "weight": 2.0}])
    assert len(neuron.inputs) == 3 and neuron.inputs[0].weight == 2.0
    assert neuron.inputs[1].source is None
    assert len(BrainNeuron("n1", "abs").used_inputs) == 1
    with pytest.raises(ValueError):
        BrainNeuron("n0", "frobnicate")
    with pytest.raises(ValueError):
        BrainNeuron("angle0", "sum")
    with pytest.raises(ValueError):
        BrainNeuron("n0", "sum", [BrainInput()] * 4)
    with pytest.raises(ValueError):
        BrainInput(None, float("nan"))
    with pytest.raises(ValueError):
        BrainEffector("bad", 1.0)


def test_brain_gene_rejects_duplicate_neuron_names_and_names_new_ones():
    with pytest.raises(ValueError):
        BrainGene(neurons=[BrainNeuron("n0"), BrainNeuron("n0")])
    brain = BrainGene(neurons=[BrainNeuron("n0"), BrainNeuron("n2")])
    assert brain.new_neuron_name() == "n1"


def test_node_gene_accepts_brain_dict():
    node = NodeGene(
        name="leg",
        size=(0.1, 0.1, 0.1),
        brain={
            "neurons": [{"name": "n0", "function": "sin", "inputs": []}],
            "effectors": [{"source": "self:n0", "weight": 0.5}],
        },
    )
    assert isinstance(node.brain, BrainGene)
    assert node.brain.effectors[0].weight == 0.5


# --- I/O -----------------------------------------------------------------------


def test_graph_is_a_control_mode():
    assert "graph" in CONTROL_MODES


def test_graph_genotype_round_trips_through_json(tmp_path):
    genotype = load_genotype_from_json(GRAPH_EXAMPLE)
    path = tmp_path / "graph.json"
    save_genotype_to_json(genotype, path)
    reloaded = load_genotype_from_json(path)
    assert json.dumps(genotype_to_dict(reloaded), sort_keys=True) == json.dumps(
        genotype_to_dict(genotype), sort_keys=True
    )
    assert reloaded.global_neurons[0].name == "clock"
    assert reloaded.nodes["leg"].brain.neurons[0].inputs[0].source == "global:clock"


def test_genotypes_without_brains_still_load():
    genotype = load_genotype_from_json("examples/example_genotype.json")
    assert genotype.global_neurons == []
    assert all(not node.brain.neurons for node in genotype.nodes.values())


def test_convert_to_graph_empties_brains():
    graph = load_genotype_from_json(GRAPH_EXAMPLE)
    converted = convert_control_mode(graph, "sine")
    assert converted.control_mode == "sine"
    back = convert_control_mode(graph, "neural")
    back = convert_control_mode(back, "graph")
    assert back.global_neurons == []
    assert all(not node.brain.neurons for node in back.nodes.values())
    assert graph.global_neurons  # the input genotype is untouched


# --- compilation ----------------------------------------------------------------


def build(genotype, task="swimming_x"):
    definition = make_definition(task)
    builder = PhenotypeBuilder(genotype, definition)
    xml = builder.build()
    return builder, definition, xml


def test_example_compiles_one_brain_per_instance():
    genotype = load_genotype_from_json(GRAPH_EXAMPLE)
    builder, _, _ = build(genotype)
    spec = builder.brain_spec
    # one global clock + one neuron in each of the 4 symmetric legs
    assert len(spec.neurons) == 5
    assert len(spec.effectors) == 4
    assert len(spec.sensors) == 4 and all(s.kind == "angle" for s in spec.sensors)
    leg_neuron = spec.neurons[1]
    assert leg_neuron.function == "sum"
    assert leg_neuron.inputs[0] == (0, 1.0)  # the shared global clock
    sensor_slots = {slot for slot, _ in (leg_neuron.inputs[1],)}
    assert min(sensor_slots) >= 5
    # each leg reads its own angle sensor
    sensor_slots = [spec.neurons[i].inputs[1][0] for i in range(1, 5)]
    assert len(set(sensor_slots)) == 4
    assert [e.slot for e in spec.effectors] == [1, 2, 3, 4]


def test_non_graph_modes_have_no_brain_spec():
    genotype = load_genotype_from_json("examples/example_genotype_neural.json")
    builder, _, _ = build(genotype)
    assert builder.brain_spec is None


def _parent_child_genotype():
    return tree_genotype(
        "body",
        {
            "body": tree_node(
                "body",
                (0.2, 0.1, 0.1),
                joint_type="free",
                children=[
                    ConnectionGene(
                        name="to_leg",
                        child="leg",
                        axis=(0.0, 1.0, 0.0),
                        parent_face="-z",
                        orientation=(0.0, 90.0, 0.0),
                    )
                ],
            ),
            "leg": tree_node("leg", (0.2, 0.04, 0.04), joint_type="hinge"),
        },
        control_mode="graph",
    )


def test_parent_and_child_scopes_resolve_between_part_instances():
    genotype = _parent_child_genotype()
    genotype.nodes["body"].brain = BrainGene(
        neurons=[
            BrainNeuron(
                "n0",
                "sum",
                [BrainInput("child:to_leg:angle0", 1.0), BrainInput("child:to_leg:n0", 1.0)],
            )
        ]
    )
    genotype.nodes["leg"].brain = BrainGene(
        neurons=[BrainNeuron("n0", "abs", [BrainInput("parent:n0", 1.0)])],
        effectors=[BrainEffector("self:n0", 1.0)],
    )
    builder, _, _ = build(genotype)
    spec = builder.brain_spec
    assert len(spec.neurons) == 2
    body_inputs = spec.neurons[0].inputs
    assert body_inputs[1] == (1, 1.0)  # body reads the leg neuron
    assert spec.neurons[1].inputs[0] == (0, 1.0)  # leg reads the body neuron
    assert body_inputs[0][0] >= 2  # the angle sensor slot follows the neurons


def test_unresolvable_sources_read_as_zero():
    genotype = _parent_child_genotype()
    genotype.nodes["body"].brain = BrainGene(
        neurons=[
            BrainNeuron(
                "n0",
                "sum",
                [
                    BrainInput("parent:n0", 3.0),  # the root has no parent
                    BrainInput("child:missing:n0", 3.0),  # no such connection
                    BrainInput("self:n99", 3.0),  # no such neuron
                ],
            )
        ]
    )
    builder, _, _ = build(genotype)
    assert builder.brain_spec.neurons[0].inputs == ((-1, 0.0),) * 3


def test_constant_input_equals_its_weight():
    genotype = _parent_child_genotype()
    genotype.nodes["leg"].brain = BrainGene(
        neurons=[BrainNeuron("n0", "sum", [BrainInput(None, 0.25)])],
        effectors=[BrainEffector(None, 0.5)],
    )
    builder, _, _ = build(genotype)
    assert builder.brain_spec.neurons[0].inputs[0] == (-1, 0.25)
    assert builder.brain_spec.effectors[0].slot == -1
    assert builder.brain_spec.effectors[0].weight == 0.5


def test_missing_effectors_default_to_zero_output():
    genotype = _parent_child_genotype()
    builder, _, _ = build(genotype)
    effectors = builder.brain_spec.effectors
    assert len(effectors) == 1 and effectors[0].slot == -1 and effectors[0].weight == 0.0


def test_ball_joints_get_three_effectors_and_sensors():
    genotype = _parent_child_genotype()
    genotype.nodes["leg"].joint_type = "ball"
    genotype.nodes["leg"].brain = BrainGene(
        neurons=[
            BrainNeuron("n0", "sum", [BrainInput(f"self:angle{i}", 1.0) for i in range(3)])
        ],
        effectors=[BrainEffector("self:n0", 1.0)] * 3,
    )
    builder, _, _ = build(genotype)
    assert len(builder.brain_spec.effectors) == 3
    assert [s.index for s in builder.brain_spec.sensors] == [0, 1, 2]


# --- runtime ---------------------------------------------------------------------


def make_runtime(genotype, task="swimming_x"):
    builder, definition, xml = build(genotype, task)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return builder, definition, model, data


def test_runtime_drives_actuators_and_is_deterministic():
    genotype = load_genotype_from_json(GRAPH_EXAMPLE)
    builder, definition, model, data = make_runtime(genotype)

    def rollout():
        local = mujoco.MjData(model)
        brain = create_brain_runtime(model, builder.brain_spec)
        controls = []
        for _ in range(30):
            apply_controller(model, local, builder.actuator_controllers, brain)
            controls.append(local.ctrl.copy())
            mujoco.mj_step(model, local)
        return np.array(controls)

    first, second = rollout(), rollout()
    assert np.abs(first).max() > 0.1
    np.testing.assert_array_equal(first, second)  # state resets with the runtime


def test_effector_output_is_clamped_and_scaled_to_ctrlrange():
    genotype = _parent_child_genotype()
    genotype.connections["to_leg"].ctrlrange = (-0.5, 0.25)
    genotype.nodes["leg"].brain = BrainGene(effectors=[BrainEffector(None, 7.0)])
    builder, _, model, data = make_runtime(genotype)
    brain = create_brain_runtime(model, builder.brain_spec)
    brain.step(model, data)
    assert data.ctrl[0] == pytest.approx(0.25)
    genotype.nodes["leg"].brain = BrainGene(effectors=[BrainEffector(None, -7.0)])
    builder, _, model, data = make_runtime(genotype)
    create_brain_runtime(model, builder.brain_spec).step(model, data)
    assert data.ctrl[0] == pytest.approx(-0.5)


def test_brain_runs_two_steps_per_control_step():
    genotype = _parent_child_genotype()
    # n1 reads n0 which reads a constant: with 2 brain steps the constant has
    # propagated through both neurons after a single control step.
    genotype.nodes["leg"].brain = BrainGene(
        neurons=[
            BrainNeuron("n0", "sum", [BrainInput(None, 0.4)]),
            BrainNeuron("n1", "sum", [BrainInput("self:n0", 1.0)]),
        ],
        effectors=[BrainEffector("self:n1", 1.0)],
    )
    builder, _, model, data = make_runtime(genotype)
    create_brain_runtime(model, builder.brain_spec).step(model, data)
    assert data.ctrl[0] == pytest.approx(0.4)


def test_contact_sensors_report_the_touching_face():
    genotype = tree_genotype(
        "body",
        {"body": tree_node("body", (0.2, 0.15, 0.1), joint_type="free")},
        control_mode="graph",
    )
    genotype.nodes["body"].brain = BrainGene(
        neurons=[
            BrainNeuron("down", "sum", [BrainInput("self:contact-z", 1.0)]),
            BrainNeuron("up", "sum", [BrainInput("self:contact+z", 1.0)]),
            BrainNeuron("side", "sum", [BrainInput("self:contact+x", 1.0)]),
        ]
    )
    definition = make_definition("walking_x")
    builder = PhenotypeBuilder(genotype, definition)
    model = mujoco.MjModel.from_xml_string(builder.build())
    data = mujoco.MjData(model)
    assert start_episode(model, data, definition.config, definition.environment) is None
    brain = BrainRuntime(model, builder.brain_spec, model.opt.timestep)
    for _ in range(20):
        mujoco.mj_step(model, data)
    brain.step(model, data)
    down, up, side = brain.signals[:3]
    assert (down, up, side) == (1.0, -1.0, -1.0)


def test_angle_sensor_tracks_the_joint():
    genotype = _parent_child_genotype()
    genotype.nodes["leg"].brain = BrainGene(
        neurons=[BrainNeuron("n0", "sum", [BrainInput("self:angle0", 1.0)])],
        effectors=[BrainEffector("self:n0", 0.0)],
    )
    builder, _, model, data = make_runtime(genotype)
    brain = create_brain_runtime(model, builder.brain_spec)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, builder.brain_spec.sensors[0].joint_name)
    data.qpos[model.jnt_qposadr[joint_id]] = 0.3
    brain.step(model, data)
    assert brain.signals[0] == pytest.approx(0.3)


def test_evaluate_task_runs_a_graph_creature():
    genotype = load_genotype_from_json(GRAPH_EXAMPLE)
    result = evaluate_task(genotype, make_definition("swimming_x"))
    assert not result.build_failed
    assert result.control_energy > 0.0
    assert result.actuator_count == 4


# --- mutation --------------------------------------------------------------------


def seeded_graph(seed=0):
    genotype = convert_control_mode(
        load_genotype_from_json("examples/example_genotype_neural.json"), "graph"
    )
    return genotype, random.Random(seed)


def all_brain_state(genotype):
    return json.dumps(genotype_to_dict(genotype), sort_keys=True)


def test_first_mutation_seeds_a_moving_brain():
    genotype, rng = seeded_graph()
    assert all(not node.brain.neurons for node in genotype.nodes.values())
    genotype.mutation(num_mutations=1, rng=rng, allow_topology_mutations=False)
    for node in genotype.nodes.values():
        if node.joint_type in ("hinge", "slide"):
            assert len(node.brain.effectors) == 1
        elif node.joint_type == "ball":
            assert len(node.brain.effectors) == 3
        else:
            assert node.brain.effectors == []
    assert any(
        n.function == "oscillate-wave"
        for node in genotype.nodes.values()
        for n in node.brain.neurons
    ) or all(node.joint_type in ("free", "fixed") for node in genotype.nodes.values())


def check_brain_invariants(genotype):
    for name, node in genotype.nodes.items():
        names = [n.name for n in node.brain.neurons]
        assert len(names) == len(set(names))
        for neuron in node.brain.neurons:
            assert neuron.function in NEURON_ARITY
            assert len(neuron.inputs) == 3
            assert all(math.isfinite(i.weight) for i in neuron.inputs)
        expected = {"hinge": 1, "slide": 1, "ball": 3}.get(node.joint_type, 0)
        assert len(node.brain.effectors) == expected
    names = [n.name for n in genotype.global_neurons]
    assert len(names) == len(set(names))
    # The brain must always compile and build.
    builder, _, _ = build(genotype)
    assert builder.brain_spec is not None


def test_repeated_mutation_keeps_brains_valid_and_evaluable():
    genotype, rng = seeded_graph(3)
    definition = make_definition("swimming_x")
    for step in range(120):
        genotype.mutation(num_mutations=4, rng=rng)
        check_brain_invariants(genotype)
        if step % 30 == 29:
            result = evaluate_task(genotype, definition)
            assert math.isfinite(result.fitness)


def test_mutation_is_reproducible_with_a_seed():
    first, rng_a = seeded_graph(5)
    second, rng_b = seeded_graph(5)
    for _ in range(15):
        first.mutation(num_mutations=3, rng=rng_a)
        second.mutation(num_mutations=3, rng=rng_b)
    assert all_brain_state(first) == all_brain_state(second)


def test_brain_mutation_runs_after_body_mutation_in_a_batch():
    genotype = load_genotype_from_json(GRAPH_EXAMPLE)
    pool = genotype._collect_mutable_parameters()
    kinds = {path[0] for path in pool}
    assert {"brain_function", "brain_input_weight", "brain_input_source",
            "brain_effector_weight", "brain_effector_source",
            "brain_neuron_addition"} <= kinds
    # Sine fields stay out of the pool; body fields stay in.
    assert ("genotype", "global_control_freq") not in pool
    assert any(path[0] == "connection" and path[2] == "motor_gear" for path in pool)


def test_each_brain_mutation_kind_changes_the_genotype():
    genotype = load_genotype_from_json(GRAPH_EXAMPLE)
    rng = random.Random(0)
    leg_context = ("node", "leg")
    paths = [
        ("brain_function", leg_context, "n0"),
        ("brain_input_weight", leg_context, "n0", 0),
        ("brain_input_source", leg_context, "n0", 0),
        ("brain_effector_weight", "leg", 0),
        ("brain_effector_source", "leg", 0),
        ("brain_neuron_addition", leg_context),
        ("brain_neuron_addition", ("global",)),
        ("brain_function", ("global",), "clock"),
    ]
    for path in paths:
        before = all_brain_state(genotype)
        description = genotype._mutate_parameter_path(path, rng)
        assert "stale" not in description, path
        assert all_brain_state(genotype) != before, path


def test_stale_brain_paths_are_ignored():
    genotype = load_genotype_from_json(GRAPH_EXAMPLE)
    before = all_brain_state(genotype)
    rng = random.Random(0)
    for path in [
        ("brain_function", ("node", "gone"), "n0"),
        ("brain_function", ("node", "leg"), "n99"),
        ("brain_effector_weight", "leg", 9),
        ("brain_neuron_addition", ("node", "gone")),
    ]:
        assert "stale" in genotype._mutate_parameter_path(path, rng)
    assert all_brain_state(genotype) == before


def test_new_neuron_is_wired_to_a_consumer():
    genotype = load_genotype_from_json(GRAPH_EXAMPLE)
    genotype._mutate_parameter_path(
        ("brain_neuron_addition", ("node", "leg")), random.Random(1)
    )
    brain = genotype.nodes["leg"].brain
    assert len(brain.neurons) == 2
    new_name = brain.neurons[1].name
    consumers = [i.source for n in brain.neurons[:1] for i in n.inputs] + [
        e.source for e in brain.effectors
    ]
    assert f"self:{new_name}" in consumers


def test_garbage_collection_removes_only_unreachable_neurons():
    genotype = load_genotype_from_json(GRAPH_EXAMPLE)
    leg = genotype.nodes["leg"].brain
    leg.neurons.append(BrainNeuron("dead", "sum", [BrainInput(None, 1.0)]))
    leg.neurons.append(BrainNeuron("dead2", "sum", [BrainInput("self:dead", 1.0)]))
    genotype.global_neurons.append(BrainNeuron("lonely", "sum"))
    description = genotype._collect_brain_garbage()
    assert "leg.dead" in description and "global.lonely" in description
    assert [n.name for n in leg.neurons] == ["n0"]
    assert [n.name for n in genotype.global_neurons] == ["clock"]


def test_garbage_collection_follows_chains_and_cross_part_references():
    genotype = _parent_child_genotype()
    genotype.global_neurons = [
        BrainNeuron("g0", "sum", [BrainInput("self:g1", 1.0)]),
        BrainNeuron("g1", "sum", [BrainInput(None, 1.0)]),
    ]
    genotype.nodes["body"].brain = BrainGene(
        neurons=[BrainNeuron("b0", "sum", [BrainInput("child:to_leg:l0", 1.0)])]
    )
    genotype.nodes["leg"].brain = BrainGene(
        neurons=[
            BrainNeuron("l0", "sum", [BrainInput("parent:b1", 1.0)]),
            BrainNeuron("l_dead", "sum"),
        ],
        effectors=[BrainEffector("global:g0", 1.0)],
    )
    genotype.nodes["body"].brain.neurons.append(BrainNeuron("b1", "sum"))
    genotype._collect_brain_garbage()
    # g0 -> (global self is meaningless: g0 reads "self:g1", which does not resolve
    # for global neurons), so only g0 is reachable from the effector.
    assert [n.name for n in genotype.global_neurons] == ["g0"]
    assert {n.name for n in genotype.nodes["leg"].brain.neurons} == set()
    assert genotype.nodes["body"].brain.neurons == []


def test_joint_type_change_repairs_effectors():
    genotype = load_genotype_from_json(GRAPH_EXAMPLE)
    rng = random.Random(0)
    genotype.nodes["leg"].joint_type = "ball"
    genotype._seed_graph_brains(rng)
    assert len(genotype.nodes["leg"].brain.effectors) == 3
    assert all(e.source == "self:n0" for e in genotype.nodes["leg"].brain.effectors)
    genotype.nodes["leg"].joint_type = "hinge"
    genotype._seed_graph_brains(rng)
    assert len(genotype.nodes["leg"].brain.effectors) == 1
    genotype.nodes["leg"].joint_type = "fixed"
    genotype._seed_graph_brains(rng)
    assert genotype.nodes["leg"].brain.effectors == []


def test_source_candidates_cover_scopes():
    genotype = load_genotype_from_json(GRAPH_EXAMPLE)
    leg = set(genotype._source_candidates(("node", "leg")))
    assert {"self:n0", "self:angle0", "self:contact+x", "global:clock"} <= leg
    body = set(genotype._source_candidates(("node", "body")))
    assert "child:leg_connection:angle0" in body and "child:leg_connection:n0" in body
    assert any(s.startswith("parent:") for s in leg)
    glob = set(genotype._source_candidates(("global",)))
    assert "global:clock" in glob and "node:leg:angle0" in glob


def test_deep_copy_keeps_brains_independent():
    genotype = load_genotype_from_json(GRAPH_EXAMPLE)
    clone = copy.deepcopy(genotype)
    clone.nodes["leg"].brain.neurons[0].inputs[0].weight = 99.0
    assert genotype.nodes["leg"].brain.neurons[0].inputs[0].weight == 1.0
