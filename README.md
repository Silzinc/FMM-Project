# Numerical physics class project
Many body simulation (FMM method)

***

## Project organisation

- [src](./src): production source code
- [design](./design): algorithm design scripts/notebooks
- [doc](./doc): documentation
- [test](./test): tests and data analysis

## Handling jupyter notebooks

Notebooks may be pushed in the `test` and `design` folders **without the output** which can be very heavy. 

## Using `uv` 

### Handle python packages

- Install a library: `uv add <library>`
- Remove a library: `uv remove <library>`
- Install libraries from `pyproject.toml`: `uv sync`
 
### Run python scripts

- In an editor supporting it (VSCode, PyCharm, etc...), activate the virtual environement of the project after installing it with `uv sync`.
- Run `uv run <script>`, or just `python <script>` with the environment of the project activated.