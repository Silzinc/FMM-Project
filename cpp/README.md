# C++ code for the FMM Solver

## Prerequisites

- A C++ compiler (e.g. [GCC](https://gcc.gnu.org/), [Clang](https://clang.llvm.org/)), supporting the C++20 standard.
- [CMake](https://cmake.org/)
- [vcpkg](https://learn.microsoft.com/fr-fr/vcpkg/get_started/get-started?pivots=shell-bash)
- [Ninja](https://ninja-build.org/)
- [Gnuplot](http://www.gnuplot.info/) (for the plotting test case)

## Installation

Add the following `CMakeUserPresets.json` (user specific) to the project directory:

```json
{
  "version": 2,
  "configurePresets": [
    {
      "name": "default",
      "inherits": "vcpkg",
      "environment": {
        "VCPKG_ROOT": "<path to vcpkg>"
      }
    }
  ]
}
```

Then run

```bash
cmake --preset=default
# or
make sync
```

Note : the first time you run this command will take a while to finish, as vcpkg downloads, configures and compiles the source code of each package instead of downloading a precompiled dynamic library.

## Building

```bash
cmake --build build
# or
make
```

The executables (`fmm-solver` built from `src/main.cpp` and `fmm-solver-test` built from `test/test.cpp`) will be in `build`.

## Adding dependencies

After finding a package on [vcpkg](https://vcpkg.io/en/packages.html), run

```bash
vcpkg add port <package>
```

Then, rerun the cmake preset command. vcpkg will indicate the lines to add to `CMakeLists.txt`. Add it, and rebuild.

## Dependencies used

- [Boost.MultiArray](https://www.boost.org/doc/libs/1_87_0/libs/multi_array/doc/index.html) is used to represent the 3D arrays in the `FMMTree` struct.
- [Boost.QVM](https://www.boost.org/doc/libs/1_87_0/libs/qvm/doc/html/index.html) provides an implementation for the arithmetic on 3x3 matrices and size 3 vectors.
- [fmt](https://fmt.dev/latest/index.html) is used for string formatting and printing.
- [doctest](https://github.com/doctest/doctest) allows to write test cases in a convenient way.
- [Matplot++](https://alandefreitas.github.io/matplotplusplus/) is a simple interface to Gnuplot for plotting.
