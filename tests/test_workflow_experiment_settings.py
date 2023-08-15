import filecmp
import shutil
import sys

import pytest

from tiger.workflow import ExperimentExistsError, ExperimentSettings


@pytest.fixture
def expdir(tmp_path):
    return tmp_path / "experiments"


class ExperimentFactory:
    def __init__(self, expdir):
        self.expdir = expdir

    def __del__(self):
        if self.expdir.exists():
            shutil.rmtree(self.expdir)

    def make_experiment(self):
        return ExperimentSettings(self.expdir)


@pytest.fixture
def experiment(expdir):
    factory = ExperimentFactory(expdir)
    yield factory.make_experiment()
    del factory


@pytest.fixture
def experiment_factory(expdir):
    factory = ExperimentFactory(expdir)
    yield factory
    del factory


def test_exp_name_required(experiment):
    """First parameter is always the name of the experiment, fail when not provided"""
    with pytest.raises(SystemExit):
        experiment.parse([])


def test_exp_name_accessible(experiment):
    """First parameter is always the name of the experiment, later available as name"""
    assert not experiment.parsed
    with pytest.raises(RuntimeError):
        _ = experiment.name

    expname = "foobar"
    experiment.parse([expname])
    assert experiment.name == expname


def test_status(experiment):
    assert experiment.parsed is False
    experiment.parse(["expname"])
    assert experiment.parsed is True


def test_required_param(experiment):
    """Not specifying a parameter without default value will fail"""
    experiment.add_param("test_param")
    with pytest.raises(SystemExit):
        experiment.parse(["expname"])


def test_optional_param(experiment):
    test_str = "foobar"
    experiment.add_param("test_param", default=test_str)
    experiment.parse(["expname"])
    assert experiment["test_param"] == test_str


def test_unknown_param(experiment):
    with pytest.raises(ValueError):
        experiment.parse(["expname", "--another_param"])
    with pytest.raises(ValueError):
        experiment.parse_and_preserve(["expname", "--another_param"], preserve_code=False)

    experiment.parse_and_preserve(["expname"])
    with pytest.raises(ValueError):
        experiment.parse_and_restore(["expname", "--another_param"], execute_preserved_code=False)

    # Tolerate unknown arguments
    args = experiment.parse(["expname", "--another_param"], tolerate_unknown_args=True)
    assert args == ["--another_param"]


def test_contains_param(experiment):
    experiment.add_param("test_param", default="foobar")
    experiment.parse(["expname"])
    assert "test_param" in experiment
    assert "unknown_test_param" not in experiment


def test_cannot_overwrite_internal_param(experiment):
    for pname in experiment.internal_args:
        with pytest.raises(ValueError):
            experiment.add_param(pname)


@pytest.mark.parametrize("ptype, pval", [("str", "foobar"), ("int", 42), ("float", 3.14)])
def test_typed_param(experiment, ptype, pval):
    experiment.add_param("test_param", type=ptype)
    experiment.parse(["expname", "--test_param", str(pval)])
    assert experiment["test_param"] == pval


def test_typed_param_unsupported_type(experiment):
    with pytest.raises(ValueError):
        experiment.add_param("test_param", type="list")
    with pytest.raises(ValueError):
        experiment.add_param("test_param", type=list)


@pytest.mark.parametrize(
    "ptype, pval", [("str", "foobar"), ("int", 42), ("float", 3.14), ("bool", False)]
)
def test_typed_param_defaults(experiment, ptype, pval):
    experiment.add_param("test_param", type=ptype, default=pval)
    experiment.parse(["expname"])
    assert experiment["test_param"] == pval


def test_non_cmd_arg(experiment):
    # Default value is required for these kind of parameters
    with pytest.raises(ValueError):
        experiment.add_param("test_param", cmdarg=False)

    # Otherwise, these args work the same
    pval = "foobar"
    experiment.add_param("test_param", cmdarg=False, default=pval)
    experiment.parse(["expname"])
    assert experiment["test_param"] == pval

    pval2 = "barfoo"
    experiment.parse(["expname", "--test_param", pval2])
    assert experiment["test_param"] == pval2


def test_bool_param(experiment):
    experiment.add_param("test_param", type="bool")

    experiment.parse(["expname"])
    assert experiment["test_param"] is False

    experiment.parse(["expname", "--test_param"])
    assert experiment["test_param"] is True


def test_bool_default_true(experiment):
    with pytest.raises(ValueError):
        experiment.add_param("test_param", type="bool", default=True)


def test_bool_list(experiment):
    with pytest.raises(ValueError):
        experiment.add_param("test_param", type="bool", list=True)


@pytest.mark.parametrize(
    "ptype, pvals", [("str", ["foo", "bar"]), ("int", [4, 2]), ("float", [3.1, 4.8])]
)
def test_list_param(experiment, ptype, pvals):
    experiment.add_param("test_param", type=ptype, list=True)
    experiment.parse(["expname", "--test_param"] + [str(p) for p in pvals])
    assert experiment["test_param"] == pvals


@pytest.mark.parametrize(
    "ptype, pvals", [("str", ["foo", "bar"]), ("int", [4, 2]), ("float", [3.1, 4.8])]
)
def test_list_param_defaults(experiment, ptype, pvals):
    experiment.add_param("test_param", type=ptype, default=pvals, list=True)
    experiment.parse(["expname"])
    assert experiment["test_param"] == pvals


def test_folder(experiment, expdir):
    # Experiment settings not parsed yet so folder name not available, resulting in a RuntimeError
    with pytest.raises(RuntimeError):
        _ = experiment.folder

    expname = "expname"
    experiment.parse([expname])
    assert experiment.folder == (expdir / expname)


def test_parse_persist_restore(experiment_factory):
    expname = "expname"
    pval = "foobar"

    # Build stack of parameters and parse and persist
    exp1 = experiment_factory.make_experiment()
    exp1.add_param("test_param")
    exp1.parse_and_preserve([expname, "--test_param", pval], preserve_code=False)
    assert exp1["test_param"] == pval

    # Attempt to restore settings
    pval2 = 42
    exp2 = experiment_factory.make_experiment()
    exp2.add_param("test_param2", type=int)
    exp2.parse_and_restore([expname, "--test_param2", str(pval2)], execute_preserved_code=False)
    assert exp2["test_param"] == pval
    assert exp2["test_param2"] == pval2


def test_preserve_code(experiment, dummy_codebase):
    # Preserve code
    experiment.parse_and_preserve(["expname"], preserve_code=True)
    cmp = filecmp.dircmp(experiment.code_folder, str(dummy_codebase))
    assert len(cmp.diff_files) == 0 and len(cmp.left_only) == 0 and len(cmp.right_only) == 0

    # Excplitily not preserve the code
    experiment.parse_and_preserve(["expname2"], preserve_code=False)
    assert not experiment.code_folder.exists()


def test_execute_preserved_code(experiment_factory, dummy_codebase, capsys):
    exp1 = experiment_factory.make_experiment()
    exp1.parse_and_preserve(["expname"], preserve_code=True)
    preserved_script_file = str(exp1.code_folder / "PrintArgvFile.py")

    # Update command line arguments
    sys.argv += ["expname"]

    # Restore arguments and execute preserved code
    exp2 = experiment_factory.make_experiment()
    with pytest.raises(SystemExit):
        exp2.parse_and_restore(sys.argv[1:], execute_preserved_code=True)

    output = capsys.readouterr().out
    assert output.strip().endswith(preserved_script_file)  # FooBar dummy script prints out __file__


def test_execute_preserved_code_with_removed_arg(experiment_factory, dummy_codebase, capsys):
    # Create experiment with argument and preserve code
    exp1 = experiment_factory.make_experiment()
    exp1.add_param("test_param")
    exp1.parse_and_preserve(["expname", "--test_param", "foo"], preserve_code=True)

    # Resume from preserved codebase after removing argument
    exp2 = experiment_factory.make_experiment()
    exp2.add_param("new_param")

    sys.argv += ["expname", "--test_param", "bar", "--new_param", "foobar", "--resume"]
    with pytest.raises(SystemExit):
        # Should exit while executing external entrypoint
        exp2.parse_and_preserve(sys.argv[1:], preserve_code=True)

    output = capsys.readouterr().out
    assert "--test_param" in output


def test_resume_with_preserved_code(experiment_factory, dummy_codebase, capsys):
    exp1 = experiment_factory.make_experiment()
    exp1.parse_and_preserve(["expname"], preserve_code=True)
    preserved_script_file = str(exp1.code_folder / "PrintArgvFile.py")

    # Update command line arguments
    sys.argv += ["expname", "--resume"]

    # Resume experiments and restore code
    exp2 = experiment_factory.make_experiment()
    with pytest.raises(SystemExit):
        exp2.parse_and_preserve(sys.argv[1:], execute_preserved_code=True)

    output = capsys.readouterr().out.strip()
    assert output.strip().endswith(preserved_script_file)
    assert exp2["resume"]


def test_resume(experiment_factory):
    # Resuming does not delete the folder and renames the previous arguments dump
    exp1 = experiment_factory.make_experiment()
    exp1.parse_and_preserve(["expname"], preserve_code=False)

    with open(str(exp1.folder / "weights.pkl"), "w"):
        pass

    # Check if the created file and the arguments file are still there
    exp2 = experiment_factory.make_experiment()
    exp2.parse_and_preserve(["expname", "--resume"])

    assert (exp2.folder / "weights.pkl").exists()
    assert (exp2.folder / "arguments.json").exists()
    assert (exp2.folder / "arguments_0.json").exists()


def test_overwrite(experiment_factory):
    exp1 = experiment_factory.make_experiment()
    exp1.add_param("test_param")
    exp1.parse_and_preserve(["expname", "--test_param", "foobar"])

    # Try to overwrite (expected to fail)
    exp2 = experiment_factory.make_experiment()
    exp2.add_param("test_param")
    with pytest.raises(ExperimentExistsError):
        exp2.parse_and_preserve(["expname", "--test_param", "foobar"])

    # Force overwrite
    exp3 = experiment_factory.make_experiment()
    exp3.add_param("test_param")
    exp3.parse_and_preserve(["expname", "--test_param", "foobar", "--overwrite"])
    assert exp3.parsed


def test_overwrite_resume(experiment):
    # Cannot do both at the same time
    with pytest.raises(ValueError):
        experiment.parse(["expname", "--overwrite", "--resume"])
