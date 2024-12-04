# C++ project template

## Prerequisites

- [CMake](https://cmake.org/)
- [vcpkg](https://vcpkg.io/en/getting-started.html)
- [Ninja](https://ninja-build.org/)
- [C++ compiler](https://gcc.gnu.org/)

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
```

## Building

```bash
cmake --build build
```

The executable will be in `build`, with its name set in `CMakeLists.txt`.

## Adding dependencies

After finding a package on [vcpkg](https://vcpkg.io/en/packages.html), run

```bash
vcpkg add port <package>
```

Then, rerun the cmake preset command. vcpkg will indicate the lines to add to `CMakeLists.txt`. Add it, and rebuild.
