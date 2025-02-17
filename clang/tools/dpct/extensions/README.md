# Pre-defined rules
This folder provides predefined user-defined migration rules in YAML files, targeted to extend the tool's migration capability. The `rule_templates` folder provides example rules to implement a user migration rule in YAML user can refer to, while other rule folders provide migration rules that can be loaded and applied directly by the tool's option `--rule-files`.

## cmake_rules
This folder provides migration rules for CMake script.

`cmake_script_migration_rule.yaml`: Provides the general migration rules for CMake script migration. This file is loaded by default if the tool is guided to migrate the CMake script (option `--migrate-build-script=CMake` is provided by the user).

`cmake_script_migration_rule_optional.yaml`: Provides extra migration rules for CMake script migration. It is not loaded by default, but please apply it if necessary.

## opt_rules
This folder provides optional migration rules.  The migration rules are not loaded by default, but please apply them if necessary.

`forceinline.yaml`: Provides a specific migration rule to migrate `__forceinline__` to `inline` instead of `__dpct_inline__`.

`intel_specific_math.yaml`: Provides migration rules to migrate some math API to Intel(R) hardware-specific API.

`macro_checks.yaml`: Provides a specific migration rule to migrate the error-checking macro `CUDA_CHECK` to `DPCT_CHECK_ERROR.`

## python_rules
This folder provides migration rules for Python script.  The migration rules are not loaded by default, but please apply them if necessary.

`python_build_script_migration_rule_ipex.yaml`: provides migration rules to migrate python build script for Pytorch-based projects to work with Intel(R) Extension for PyTorch (IPEX).

`python_build_script_migration_rule_pytorch.yaml`: provides migration rules to migrate Python build script for Pytorch-based projects to work with official PyTorch release with XPU support.

## pytorch_api_rule_rules
`pytorch_api.yaml`: provides migration rules to migrate Pytorch CUDA backend-specific API of Pytorch-based projects to Pytorch backend agnostic API or XPU backend-specific API.
