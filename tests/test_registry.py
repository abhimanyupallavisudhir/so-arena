"""The component registry: YAML-buildable reward rules, verifiers, scorers and mechanisms."""

from pathlib import Path

import pytest

from so_arena.cli import main
from so_arena.core.rewards import JudgeScore, MonitorPenalty, Scaled, Sum, ZeroSum
from so_arena.core.verification import PythonExecVerifier, VerificationPolicy
from so_arena.core.store import RunStore
from so_arena.registry import KINDS, RegistryError, build, names, register, resolve, spec_errors
from so_arena.spec import SpecError, build_mechanism, load_spec, parse_spec, run_spec, spec_hash

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_every_registered_component_resolves():
    for kind in KINDS:
        for name in names(kind):
            try:
                assert callable(resolve(kind, name))
            except RegistryError as e:  # a domain whose optional dependency is missing
                assert "unavailable" in str(e)


def test_shipped_configs_load_and_build_as_before():
    for path in sorted(CONFIGS.glob("*.yaml")):
        spec = load_spec(path)
        mechs = [build_mechanism(m) for m in spec.mechanisms]
        assert len(mechs) == len(spec.mechanisms)
    # the older `name:` form of a reward and the registry's `type:` form build the same mechanism
    old = build_mechanism({"name": "debate", "rounds": 2, "reward": {"name": "zero_sum", "transform": "prob"}})
    new = build_mechanism({"type": "debate", "rounds": 2, "reward": {"type": "zero_sum", "transform": "prob"}})
    assert isinstance(old.reward_rule, ZeroSum) and old.config_hash() == new.config_hash()


def test_nested_components_are_built_recursively():
    m = build_mechanism({
        "name": "debate", "label": "debate+checked", "rounds": 1,
        "reward": {"type": "sum", "rules": ["judge_score", {"type": "scaled", "c": 0.5,
                                                            "rule": {"type": "monitor_penalty", "lam": 2.0}}]},
        "verification": {"verifiers": ["fact", {"type": "python", "timeout": 2}], "budget_per_role": 1}})
    assert m.name == "debate+checked"
    rule = m.reward_rule
    assert isinstance(rule, Sum) and isinstance(rule.rules[0], JudgeScore) and isinstance(rule.rules[1], Scaled)
    assert isinstance(rule.rules[1].rule, MonitorPenalty) and rule.rules[1].rule.lam == 2.0
    vp = m.verification
    assert isinstance(vp, VerificationPolicy) and vp.budget_per_role == 1
    assert vp.verifiers[0] == "fact" and isinstance(vp.verifiers[1], PythonExecVerifier)  # names stay names
    assert vp.verifiers[1].timeout == 2
    # a component outside a known argument is found by its type, when only one kind registers it
    assert isinstance(build("reward", {"type": "zero_sum", "inner": {"type": "judge_score", "transform": "brier"}}).inner,
                      JudgeScore)


def test_spec_errors_name_the_problem_before_anything_is_built():
    errs = spec_errors("mechanism", {"type": "debate", "reward": {"type": "sum", "rules": [
        {"type": "judge_scor"}, {"type": "random_audit", "inner": "judge_score", "p": 0.1}]},
        "verification": {"verifiers": [{"type": "pyton"}], "budgett": 3}}, "m")
    text = "\n".join(errs)
    assert "did you mean 'judge_score'" in text
    assert "needs 'oracle'" in text  # a required argument a YAML spec cannot give is reported, not a TypeError later
    assert "did you mean 'python'" in text and "did you mean 'budget'" in text
    assert spec_errors("reward", {"type": "judge_score", "transfrom": "log"}, "r")[0].count("did you mean 'transform'")
    with pytest.raises(SpecError, match="did you mean 'stance_value'"):
        parse_spec({**SPEC, "ground_truth": ["domain", "stance_valu"]})


def test_register_adds_a_component_to_specs():
    @register("reward", "test_constant_one")
    class One(JudgeScore):
        pass

    try:
        m = build_mechanism({"name": "direct", "reward": {"type": "test_constant_one", "transform": "brier"}})
        assert isinstance(m.reward_rule, One) and "test_constant_one" in names("reward")
    finally:
        from so_arena import registry

        registry._REGISTRY["reward"].pop("test_constant_one")


SPEC = {"name": "reg", "domain": {"name": "synthetic", "n_items": 3},
        "policies": {"expert": {"synthetic": "arguer"}, "judge": {"synthetic": "judge"}},
        "mechanisms": [{"name": "direct"}], "experiment": {"type": "asd", "agent": "expert", "fixtures": {"judge": "judge"}},
        "report": False}


def test_spec_ground_truth_replaces_the_domains_scorers(tmp_path):
    base = parse_spec(SPEC)
    assert base.ground_truth is None
    # specs written before the key existed keep their hash (a resumable run stays resumable)
    assert spec_hash(base) == spec_hash({k: v for k, v in base.model_dump(mode="json").items() if k != "ground_truth"})
    only_stance = parse_spec({**SPEC, "ground_truth": ["stance_value"]})
    assert spec_hash(only_stance) != spec_hash(base)
    run_spec(only_stance, out=tmp_path / "a")
    eps = RunStore(tmp_path / "a").episodes()
    assert eps and all("judge_correct" not in e.ground_truth and "role_values" in e.ground_truth for e in eps)
    run_spec(parse_spec({**SPEC, "ground_truth": ["domain"]}), out=tmp_path / "b")
    assert all("judge_correct" in e.ground_truth for e in RunStore(tmp_path / "b").episodes())


def test_cli_lists_rewards_verifiers_and_scorers(capsys):
    for what in ("rewards", "verifiers", "scorers"):
        assert main(["list", what]) == 0
    out = capsys.readouterr().out
    assert "judge_score" in out and "quote" in out and "stance_value" in out


# Components that take a Python callable as a required argument cannot come from YAML; they are built in Python.
PYTHON_ONLY = {"so_arena.core.rewards:FunctionReward", "so_arena.core.rewards:ZeroSum",  # ZeroSum: via "zero_sum"
               "so_arena.core.ground_truth:FunctionScorer", "so_arena.core.verification:CallableVerifier"}


def test_every_component_class_is_registered():
    """A new reward rule, verifier or ground-truth scorer must be registered (or be deliberately Python-only),
    so that specs can name it."""
    import importlib
    import inspect
    import pkgutil

    import so_arena
    from so_arena import registry
    from so_arena.core.ground_truth import GroundTruthScorer
    from so_arena.core.rewards import RewardRule
    from so_arena.core.verification import Verifier

    # entries are "module:attribute" strings until first resolved, then the objects themselves
    targets = {kind: {v if isinstance(v, str) else f"{getattr(v, '__module__', '')}:{getattr(v, '__qualname__', '')}"
                      for v in table.values()} for kind, table in registry._REGISTRY.items()}
    missing = []
    for info in pkgutil.walk_packages(so_arena.__path__, "so_arena."):
        try:
            mod = importlib.import_module(info.name)
        except ImportError:  # optional dependencies
            continue
        for name, obj in vars(mod).items():
            if not inspect.isclass(obj) or obj.__module__ != mod.__name__ or inspect.isabstract(obj):
                continue
            for kind, base in (("reward", RewardRule), ("verifier", Verifier), ("scorer", GroundTruthScorer)):
                ref = f"{mod.__name__}:{name}"
                if issubclass(obj, base) and obj is not base and ref not in targets[kind] and ref not in PYTHON_ONLY:
                    missing.append(f"{kind} {ref}")
    assert not missing, f"unregistered components: {missing}"


def test_existing_configurations_keep_their_config_hashes():
    """Options added later stay out of config hashes at their defaults: configurations that existed before keep
    their hashes, so run stores resume and old and new runs group together. (``MonitoredWork`` changed on
    purpose: missing monitor scores now fail closed, so episodes stored before are not reused.)"""
    from so_arena.core.rewards import Whistleblower
    from so_arena.mechanisms import get_mechanism

    pinned = {  # the hashes of the configurations as first released
        "debate+verification": (lambda: get_mechanism("debate", verification=VerificationPolicy(verifiers=["quote"], budget_per_role=2)),
                                "f5cad58faa"),
        "debate": (lambda: get_mechanism("debate"), "17f5edbc74"),
        "team+whistleblower": (lambda: get_mechanism("team", reward=Whistleblower(bounty=0.5)), "8e80ad892d"),
        "team": (lambda: get_mechanism("team"), "69ed34a15c"),
        "consultancy": (lambda: get_mechanism("consultancy"), "7ebd5af6f1"),
        "peer_prediction": (lambda: get_mechanism("peer_prediction"), "e3db2938df"),
    }
    for name, (make, h) in pinned.items():
        assert make().config_hash() == h, name
    # and a new option, once set, does change the hash
    assert get_mechanism("debate", verification=VerificationPolicy(verifiers=["quote"], budget_per_role=2, noise=0.1)).config_hash() \
        != "f5cad58faa"
