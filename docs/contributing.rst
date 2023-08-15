Contributing to Tiger
=====================

Bug reports and feature requests
--------------------------------

If you find any bugs or strange behavior while using Tiger, please report this on GitHub_ with a clear description of your
setup, the experienced and the expected behavior.

Feature requests can also be submitted on GitHub_.

.. _GitHub: https://github.com/DIAGNijmegen/msk-tiger/issues

Writing code yourself
---------------------

Next to reporting bugs and requesting new features, direct contributions are also very welcome. For example, you could fix
a bug, extend the functionality of an existing functions or class, or add one of your frequently used functions to the library.

We follow a few important principles:

* **Pull requests:** The master branch of the Tiger repository is protected, even users with write access to the repository can
  not directly push to the master branch. All contributions need to be implemented in branches, either in the Tiger repository
  or in your personal fork. These changes are then merged into the master branch via pull requests, which are first reviewed and
  tested.
* **Releases:** New releases of the Tiger library are first assembled in the *master* branch. When a new version is released, the
  master branch is merged into the *stable* branch.
* **Tests:** New code should be covered by unit tests. We use the `PyTest` framework for writing tests and GitHub Actions to run
  them automatically for any pull request. So before a pull request can be merged into the master branch, all tests need to pass.
  Check the ``tests/`` folder in the repository for examples. Additionally, please also make sure to test your changes locally.
* **Documentation:** We use sphinx to automatically generate this documentation from the source code. New functions or classes
  therefore need to be documented with block comments in the code. The style that we use is the numpy style, see numpydoc_. When
  adding new functions or classes, please make sure to add them to the documentation (in the ``docs/modules/`` folder).
* **Code formatting:** We use black_ to automatically format Python code. Install pre-commit with ``pip install pre-commit`` and
  run in the project folder ``pre-commit install`` to automatically run the black formatter on every commit.
* **Type hints:** Provide type hints as much as possible, but move them to the docstring if they would be complex to implement
  (e.g., functions that accept either one or multiple images, and optionally one or multiple masks, and return either one or
  multiple values depending on which parameters were supplied).
* **Private objects:** Use one underscore ("_") at the beginning of function and variable names to indicate that they are not part of
  the public API of the module (this is mostly done for module-wide definitions, not within classes). Do not use two underscores
  for this purpose.

Releases
--------

Steps to make a new release:

* Remove "dev" from the version number on the master branch in `tiger/__init__.py`.
* Tag the commit with the version number.
* Push the changes to GitHub, including the new tag.
* GitHub will automatically create a release. Update the description of this release with a list of changes (see `docs/changelog.rst`).
* Rebase the *stable* branch onto the new release: `git checkout stable` and `git rebase X.Y.Z` (use tag of the release for X.Y.Z).
* Add a new section to changelog.rst for future changes and add "dev0" to the version number on the master branch (`X.Y.dev0`).
* Check if there was a milestone on GitHub for this release and close it.

.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html
.. _black: https://github.com/psf/black
