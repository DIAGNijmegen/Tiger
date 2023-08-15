"""
Module aiming at supporting a standardized workflow for experiments etc, that improves the
reproducibility of results by automatically storing parameters used to obtain the results.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from . import TigerException
from .cluster import Entrypoint
from .io import PathLike, copytree

_type_lut = {"str": str, "int": int, "float": float, "bool": bool}


def _type_from_string(s: str) -> object:
    return _type_lut[s]


def _type_to_string(t: object) -> str:
    for s in _type_lut:
        if _type_lut[s] == t:
            return s
    raise KeyError(t)


def _execute_command(args: List[str]) -> str:
    output = subprocess.check_output(args, stderr=subprocess.DEVNULL)
    return str(output, "utf-8").strip()


class ExperimentExistsError(TigerException):
    """Raised when an experiment already exists in situation where this is not allowed"""


class ExperimentSettings:
    """
    Represents an experiment with all its modifiable parameters and results. An experiment can have multiple
    custom parameters, but always has a name (used to identify the experiment, so has to be unique).

    Initially, no parameters are defined and no values are available. When running a new experiment, parameters
    are added and a list of input values (typically command line arguments from sys.argv) are parsed. After parsing
    the input values, the whole set of parameters and corresponding value (e.g., which batch size was used in
    this experiment?) are stored in a JSON file. The values are then available via experiment["param_name"].

    >>> experiment = ExperimentSettings("/home/users/experiments")
    >>> experiment.add_param("batch_size", type="int", default=16)
    >>> experiment.parse_and_preserve()  # command line arguments: FinalExperiment --batch_size 64
    >>> train_network(batch_size=experiment["batch_size"])

    Typically, the results of the experiment (a trained network, for example) need to be used after the
    experiment finished. This could be a test script that applies the trained network to the test data and
    therefore needs to know some of the original arguments that defined the experiment (batch size might not
    be relevant, but other parameter such as the number of filters might be). In this situtation, the name
    of the experiment is used to find the corresponding directory and to restore the original arguments
    from the JSON file. If needed, additional parameters can still be added:

    >>> experiment = ExperimentSettings("/home/users/experiments")
    >>> experiment.add_param("calculate_dice", type="bool")
    >>> experiment.parse_and_restore()  # command line arguments: FinalExperiment --calculate_dice
    >>> experiment["batch_size"]
    64
    >>> experiment["batch_size"]
    true

    Parameters
    ----------
    expdir : pathlib.Path or str
        Path to a directory in which all experiment-releated data is stored. Each individual experiment
        has a name and a corresponding subdirectory with that name in this folder.
    """

    def __init__(self, expdir: PathLike):
        self.expdir = Path(expdir)
        self.args: Dict[str, Any] = {}
        self.argparser = argparse.ArgumentParser()
        self.argparser.add_argument("experiment_name")
        self.argparser.add_argument("--overwrite", action="store_true")
        self.argparser.add_argument("--resume", action="store_true")
        self.argparser.add_argument("--debug", action="store_true")
        self.internal_args = ("experiment_name", "overwrite", "resume", "debug")

    def add_param(
        self,
        name: str,
        type: Union[str, object] = "str",
        default: Optional[Any] = None,
        list: bool = False,
        cmdarg: bool = True,
        help: Optional[str] = None,
    ):
        """
        Register a parameter of the experiment

        Parameters
        ----------
        name
            Parameter name. Corresponds to a (command line) argument "--name", i.e., adding a parameter
            with name "batch_size" activates a command line parameter "--batch_size".

        type
            Data type of the parameter (supported are str, int, float, bool)

        default
            Default value of the parameter if no value is specified. If the default value is set to None,
            the parameter becomes mandatory.

        list
            If set to true, multiple values are accepted and the parameter will be a list with items of
            the parameter type (so on the command line: "--batch_sizes 8 16 32" is parsed as "[8, 16, 32]")

        cmdarg
            Whether the parameter is a command line argument or an internal argument. Internal arguments
            require a sensible default value.

        help
            When incorrect arguments are supplied on the command line, an overview of all non-internal
            required and optional parameters is displayed. To display a more detailed message with a
            parameter, specify a help text.
        """
        if default is None and not cmdarg:
            raise ValueError(
                "Parameters that are not command line arguments must have a default value"
            )

        if name in self.internal_args:
            raise ValueError("This name is a reserved name and cannot be used for a parameter")

        try:
            if isinstance(type, str):
                type_str = type
                type = _type_from_string(type)
            else:
                type_str = _type_to_string(type)
        except KeyError:
            raise ValueError("Unknown or not supported type for parameter")

        # Add parameter to argument parser
        parser_args = {}
        if not cmdarg:
            parser_args["help"] = argparse.SUPPRESS
        elif help is not None:
            parser_args["help"] = help

        if type_str == "bool":
            if default:
                raise ValueError("Boolean parameters cannot have True as default value")
            if list:
                raise ValueError("Boolean parameters cannot be lists")
            parser_args["action"] = "store_true"
        else:
            if default is None:
                parser_args["required"] = True
            else:
                parser_args["default"] = default

            if list:
                parser_args["nargs"] = "*"

            parser_args["type"] = type

        self.argparser.add_argument(f"--{name}", **parser_args)

    def parse(
        self,
        argv: Optional[Iterable[str]] = None,
        *,
        tolerate_unknown_args: bool = False,
    ) -> List[str]:
        """Only parses the arguments, does not write or read JSON file with arguments"""
        known_args, unknown_args = self.argparser.parse_known_args(argv)

        # Cannot overwrite and resume at the same time
        if known_args.overwrite and known_args.resume:
            raise ValueError(
                "Overwriting and resuming experiments are not compatible, "
                "specify either --overwrite or --resume, but not both"
            )

        # Unknown arguments?
        if not tolerate_unknown_args and len(unknown_args) > 0:
            raise ValueError("Unknown arguments: " + str(unknown_args))

        self.args = vars(known_args)
        return unknown_args

    def parse_and_preserve(
        self,
        argv: Optional[Iterable[str]] = None,
        *,
        preserve_code: bool = True,
        execute_preserved_code: bool = True,
    ):
        """Parses arguments (by default from sys.argv) and stores all parameters and their values in a JSON file"""
        # Parse command line arguments
        unknown_args = self.parse(argv, tolerate_unknown_args=True)

        # Unknown arguments are only okay when not resuming, i.e., executing preserved code - they
        # have to be tolerated when resuming because the preserved code might require different
        # arguments (this can be happen when the main code base was changed, e.g., a parameter was
        # removed that is required by the preserved code).
        if len(unknown_args) > 0 and not (self.args["resume"] and execute_preserved_code):
            raise ValueError("Unknown arguments: " + str(unknown_args))

        # Check if the experiment folder already exists
        if self.folder.exists():
            if self.args["overwrite"]:
                # Delete entire folder to make sure that we don't mix new with old data
                shutil.rmtree(self.folder)
            elif self.args["resume"]:
                # Did we preserve the original code and should we run this instead?
                if execute_preserved_code:
                    try:
                        self.execute_preserved_code()
                    except (FileNotFoundError, RecursionError):
                        # Could not resume because there was no preserved code or we are already
                        # running the preserved code, which are both fine situations - only that
                        # in those cases unknown arguments should not be tolerated
                        if len(unknown_args) > 0:
                            raise ValueError("Unknown arguments: " + str(unknown_args))
            else:
                raise ExperimentExistsError(
                    "Experiment already exists, "
                    "choose different name or specify --overwrite or --resume"
                )

        # Store parameters in JSON file (minus the internal ones)
        outfile = self.folder / "arguments.json"

        if self.args["resume"]:
            # When resuming, the arguments file can already exist and we need to rename it
            i = 0
            while outfile.exists():
                newname = self.folder / f"arguments_{i}.json"
                if not newname.exists():
                    outfile.rename(newname)
                else:
                    i += 1

        self.dump_arguments(outfile)

        # Preserve the original codebase?
        if preserve_code:
            self.preserve_code()

    def parse_and_restore(
        self,
        argv: Optional[Iterable[str]] = None,
        *,
        execute_preserved_code: bool = True,
        tolerate_unknown_args: bool = False,
    ):
        """Parses arguments (by default from sys.argv) and restores previously used arguments"""
        # Parse command line arguments (important, then we know the name of the experiment)
        unknown_args = self.parse(argv, tolerate_unknown_args=True)
        if not execute_preserved_code and not tolerate_unknown_args and len(unknown_args) > 0:
            raise ValueError("Unknown arguments: " + str(unknown_args))

        # Read parameters from JSON file and add them to the args dict
        infile = self.folder / "arguments.json"
        with open(str(infile)) as fp:
            restored_args = json.load(fp)

        for name, value in restored_args.items():
            if name not in self.args:
                self.args[name] = value

        del self.args["overwrite"]  # this internal flag is not needed when restoring parameters

        # Did we preserve the original code and should we run this instead?
        if execute_preserved_code:
            try:
                self.execute_preserved_code()
            except (FileNotFoundError, RecursionError):
                # Executing preserved code failed, so check now whether we have to complain about
                # unknown arguments that we silently tolerated before because the preserved code
                # might know about them
                if not tolerate_unknown_args and len(unknown_args) > 0:
                    raise ValueError("Unknown arguments: " + str(unknown_args))

    def dump_arguments(
        self,
        filename: PathLike,
        *,
        include_internal_args: bool = False,
        include_git_commit: bool = True,
        include_env: bool = True,
    ):
        # Remove internal arguments from argument list
        persistent_args = self.args.copy()
        if not include_internal_args:
            for arg in self.internal_args:
                try:
                    del persistent_args[arg]
                except KeyError:
                    pass

        # Check if current directory is a git repository and include commit ID
        if include_git_commit:
            try:
                log = _execute_command(["git", "log", "-1", "--format=%H"])
                match = re.match(r"[a-f0-9]+", log)
                if match:
                    commit_id = match.group(0)
                    commit_clean = _execute_command(["git", "status", "-s"]) == ""

                    persistent_args["_git"] = {
                        "commit": commit_id,
                        "clean": commit_clean,
                    }
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass

        # Add environment variables
        if include_env:
            persistent_args["_environ"] = dict(os.environ)

        self.dump_into_json_file(persistent_args, filename)

    @staticmethod
    def active_codebase() -> Path:
        return Path(sys.argv[0]).resolve().parent

    def preserve_code(self, srcdir: Optional[PathLike] = None):
        """Preserve a copy of the code run in this experiment in the experiment's folder"""
        if srcdir is None:
            srcdir = self.active_codebase()
        else:
            srcdir = Path(srcdir)

        dstdir = self.code_folder
        if dstdir.exists():
            if srcdir.samefile(dstdir):
                return
            else:
                shutil.rmtree(dstdir)

        copytree(srcdir, dstdir, ignore=shutil.ignore_patterns("__pycache__", ".git"))

    def execute_preserved_code(self):
        """Attempts to switch to the preserved codebase"""
        newdir = self.code_folder
        if not newdir.exists():
            raise FileNotFoundError("Code has not been preserved and can therefore not be executed")

        curdir = self.active_codebase()
        if newdir.samefile(curdir):
            raise RecursionError(
                "Preserved codebase is already being executed, refusing to execute again"
            )

        entrypoint = Entrypoint(newdir)
        entrypoint.execute_external_entrypoint(recurse=False)

    @staticmethod
    def dump_into_json_file(obj: Any, filename: PathLike):
        # Ensure parent folder exists
        json_file = Path(filename)
        json_file.parent.mkdir(parents=True, exist_ok=True)

        # Dump object json-encoded into file with some pretty printing
        with json_file.open("w") as fp:
            json.dump(obj, fp, indent=2, separators=(",", ": "))

    def __getitem__(self, item: str) -> Any:
        return self.args[item]

    def __contains__(self, item: str) -> bool:
        return item in self.args

    @property
    def name(self) -> str:
        """Name of the experiment, available afters parsing input data"""
        try:
            return self.args["experiment_name"]
        except KeyError:
            raise RuntimeError("Arguments not parsed yet, experiment name still unknown")

    @property
    def parsed(self) -> bool:
        """Indicates whether input data has been parsed so that parameter values have been populated"""
        return "experiment_name" in self.args

    @property
    def folder(self) -> Path:
        """The experiment's directory in which all results should be stored"""
        if not self.parsed:
            raise RuntimeError("Arguments not parsed yet, experiment folder still unknown")
        return self.expdir / self.name

    @property
    def code_folder(self) -> Path:
        """Subdirectory of the experiment's directory in which the source code is backed up"""
        return self.folder / "code"
