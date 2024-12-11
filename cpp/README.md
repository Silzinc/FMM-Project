# C++ code for the FMM Solver

## Prerequisites

- A C++ compiler (e.g. [GCC](https://gcc.gnu.org/), [Clang](https://clang.llvm.org/)), supporting the C++20 standard.
- [CMake](https://cmake.org/)
- [vcpkg](https://vcpkg.io/en/getting-started.html)
- [Ninja](https://ninja-build.org/)
- [Gnuplot](http://www.gnuplot.info/) (for the plotting test case)

## Getting started

Add the following `CMakeUserPresets.json` (user specific):

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
