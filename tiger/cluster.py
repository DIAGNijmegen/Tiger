import importlib
import os
import sys
from pathlib import Path
from typing import Optional

from . import TigerException
from .io import PathLike


class EntrypointError(TigerException):
    EXTERNAL_ENTRYPOINT_NOT_SPECIFIED = 1
    RECURSION = 2
    ENTRYPOINT_NOT_FOUND = 3
    MODULE_NOT_SPECIFIED = 4
    MODULE_IMPORT_ERROR = 5


class Entrypoint:
    """
    The entrypoint module enables dynamic execution of different scripts in a docker-based
    environment. A single entrypoint can be defined and the actual script that is executed
    is determined by that entrypoint script based on the first command line argument. Also,
    external codebases can be used so that the docker image does not have to be rebuild for
    every little change to the code.
    """

    def __init__(
        self,
        external_source_dir: Optional[PathLike] = None,
        module_class: str = "Module",
        module_function: str = "main",
        verbose: bool = True,
    ):
        if external_source_dir is None:
            self.external_source_dir = None
        else:
            self.external_source_dir = Path(external_source_dir)

        self.module_class = module_class
        self.module_function = module_function
        self.verbose = verbose

    def execute_external_entrypoint(self, recurse: bool = True):
        # Check if file that needs to be executed exists
        if self.external_source_dir is None:
            raise EntrypointError(
                "Cannot execute external entrypoint, no external source directory specified",
                EntrypointError.EXTERNAL_ENTRYPOINT_NOT_SPECIFIED,
            )

        # Are we recursing and need to stop?
        try:
            if globals()["__entrypoint_stop__"]:
                raise EntrypointError(
                    "Cannot execute external entrypoint, this is not the first external entrypoint and recusion has been disabled",
                    EntrypointError.RECURSION,
                )
        except KeyError:
            pass

        try:
            srcdir = self.external_source_dir.resolve()
            internal_source_dir = Path(sys.argv[0]).resolve().parent
            if srcdir.samefile(internal_source_dir):
                raise EntrypointError(
                    "Cannot execute external entrypoint, external source directory is the active working directory",
                    EntrypointError.RECURSION,
                )
        except FileNotFoundError:
            raise EntrypointError(
                "External entrypoint not available",
                EntrypointError.ENTRYPOINT_NOT_FOUND,
            )

        script = Path(sys.argv[0]).name
        try:
            entrypoint_name = globals()["__entrypoint__"]
        except KeyError:
            pass
        else:
            module_name = script[:-3]
            script = entrypoint_name
            sys.argv.insert(1, module_name)

        external_entrypoint = srcdir / script
        if not external_entrypoint.exists():
            raise EntrypointError(
                "External entrypoint not available",
                EntrypointError.ENTRYPOINT_NOT_FOUND,
            )
        elif self.verbose:
            print(f'Executing external entrypoint "{external_entrypoint}"')

        # Determine which modules were loaded from the current source directory and therefore
        # need to be reloaded from the external source directory
        modules_to_reload = []
        for module in sys.modules.values():
            try:
                if module.__name__ == "__main__":
                    continue

                module_dir = Path(module.__file__).resolve().parent
                if module_dir.samefile(internal_source_dir):
                    modules_to_reload.append(module)
            except (AttributeError, TypeError, FileNotFoundError):
                continue

        # Modify global variables to contain updated file path
        global_variables = globals()
        global_variables["__file__"] = str(external_entrypoint)
        global_variables["__entrypoint__"] = script
        global_variables["__entrypoint_stop__"] = not recurse

        # Update argv list to contain correct script name
        sys.argv[0] = script

        # Switch into new working directory
        srcdir = str(srcdir)
        os.chdir(srcdir)
        sys.path[0] = srcdir

        # Reload modules so that now the external module is loaded
        for module in modules_to_reload:
            importlib.reload(module)

        # Execute external entrypoint
        with open(script, "rb") as pyfile:
            exec(compile(pyfile.read(), script, "exec"), global_variables)

        # Remove global variables (not really needed, only for PyTest)
        del global_variables["__entrypoint__"]
        del global_variables["__entrypoint_stop__"]
        sys.exit(0)

    def execute(self):
        # Execute external entrypoint?
        try:
            self.execute_external_entrypoint()
        except EntrypointError as e:
            if e.code not in (
                EntrypointError.EXTERNAL_ENTRYPOINT_NOT_SPECIFIED,
                EntrypointError.RECURSION,
                EntrypointError.ENTRYPOINT_NOT_FOUND,
            ):
                raise

        # Import and execute module based on module name
        if not (len(sys.argv) > 1):
            raise EntrypointError("No module specified", EntrypointError.MODULE_NOT_SPECIFIED)

        # At this point, we know the name of the entrypoint script, so preserve it
        global __entrypoint__
        __entrypoint__ = Path(sys.argv[0]).name

        # Use first command line argument to deduce module name
        module_name = sys.argv[1]
        if module_name.endswith(".py"):
            module_name = module_name[:-3]

        sys.argv[0] = f"{module_name}.py"
        del sys.argv[1]

        if self.verbose:
            print(f'Executing module "{module_name}"')

        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            raise EntrypointError(
                f'Error loading module "{module_name}": {error}',
                EntrypointError.MODULE_IMPORT_ERROR,
            )
        else:
            if hasattr(module, self.module_class):
                Module = getattr(module, self.module_class)
                Module(sys.argv[1:]).execute()
            elif hasattr(module, self.module_function):
                main = getattr(module, self.module_function)
                main(sys.argv[1:])
