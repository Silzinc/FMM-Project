# Numerical physics class project

Many body simulation (FMM method)

---

## Project organisation

- [src](./src): production source code.
- [test](./test): tests and data analysis.
- [cpp](./cpp): C++ implementation of the FMM algorithm.

## Installation (python part)

### With `uv`

[`uv`](https://docs.astral.sh/uv/#highlights) is recommended to have the right python version installed, as it manages both python versions and packages.

```bash
uv sync
```

Then you can activate the virtual environment created in `.venv` to use python, or prefix all your python commands with `uv run` :

```bash
uv run some_script.py
uv run pytest test/test_fmm.py -s
```

## Using `uv`

### Handle python packages

- Install a library: `uv add <library>`
- Remove a library: `uv remove <library>`
- Install libraries from `pyproject.toml`: `uv sync`

### Run tests

After running `uv sync`, you can run the tests in the `test` directory with the following command:

```bash
uv run pytest test/*.py -s
```
