:orphan:

.. _opt-always-use-async-handler:

``--always-use-async-handler``

.. _desc-always-use-async-handler:

Use async exception handler when creating new ``sycl::queue`` with
``dpct::create_queue`` in addition to default ``dpct::get_default_queue``.
Default: ``off``.

.. _end-always-use-async-handler:



.. _opt-analysis-scope-path:

``--analysis-scope-path=<dir>``

.. _desc-analysis-scope-path:

The directory path for the analysis scope of the source tree that needs
to be migrated. Default: the value of ``--in-root``.

.. _end-analysis-scope-path:





.. _opt-assume-nd-range-dim:

``--assume-nd-range-dim=<value>``

.. _desc-assume-nd-range-dim:

Provides a hint to the tool on the dimensionality of nd_range to use in
generated code. The values are:

- ``=1``: Generate kernel code assuming 1D ``nd_range`` where possible, and 3D
  in other cases.
- ``=3``: Generate kernel code assuming 3D ``nd_range`` (default).

.. _end-assume-nd-range-dim:



.. _opt-build-script-file:

``--build-script-file=<file>``

.. _desc-build-script-file:

Specifies the name of generated makefile for migrated file(s). Default name:
``Makefile.dpct``.

.. _end-build-script-file:



.. _opt-change-cuda-files-extension-only:

``--change-cuda-files-extension-only``

.. _desc-change-cuda-files-extension-only:

Limit extension change to ``.cu`` and ``.cuh`` files only. Default: ``off``.

.. _end-change-cuda-files-extension-only:



.. _opt-check-unicode-security:

``--check-unicode-security``

.. _desc-check-unicode-security:

Enable detection and warnings about Unicode constructs that can be exploited by
using bi-directional formatting codes and homoglyphs in identifiers. Default: ``off``.

.. _end-check-unicode-security:



.. _opt-comments:

``--comments``

.. _desc-comments:

Insert comments explaining the generated code. Default: ``off``.

.. _end-comments:



.. _opt-cuda-include-path:

``--cuda-include-path=<dir>``

.. _desc-cuda-include-path:

The directory path of the CUDA\* header files.

.. _end-cuda-include-path:



.. _opt-custom-helper-name:

``--custom-helper-name=<name>``

.. _desc-custom-helper-name:

DEPRECATED: Specifies the helper headers folder name and main helper header file
name. Default: ``dpct``.

.. _end-custom-helper-name:



.. _opt-enable-ctad:

``--enable-ctad``

.. _desc-enable-ctad:

Use a C++17 class template argument deduction (CTAD) in your generated code.

.. _end-enable-ctad:



.. _opt-enable-profiling:

``--enable-profiling``

.. _desc-enable-profiling:

Enable SYCL\* queue profiling in helper functions. Default: ``off``.

.. _end-enable-profiling:



.. _opt-extra-arg:

``--extra-arg=<string>``

.. _desc-extra-arg:

Additional argument to append to the migration command line, example:
``--extra-arg="-I /path/to/header"``. The options that can be passed this
way can be found with the ``dpct -- -help`` command.

.. _end-extra-arg:



.. _opt-format-range:

``--format-range=<value>``

.. _desc-format-range:

Sets the range of formatting.

The values are:

- ``=all``: Formats all code.
- ``=migrated``: Only formats the migrated code (default).
- ``=none``: Do not format any code.

.. _end-format-range:



.. _opt-format-style:

``--format-style=<value>``

.. _desc-format-style:

Sets the formatting style.

The values are:

- ``=custom``: Use the coding style defined in the ``.clang-format`` file (default).
- ``=google``: Use the Google\* coding style.
- ``=llvm``: Use the LLVM coding style.


Example for the .clang-format file content:

::

  BasedOnStyle:
  LLVM IndentWidth: 4
  TabWidth: 4
  UseTab: ForIndentation

.. _end-format-style:



.. _opt-gen-build-script:

``--gen-build-script``

.. _desc-gen-build-script:

Generates makefile for migrated file(s) in ``-out-root`` directory.
Default: ``off``.

.. _end-gen-build-script:



.. _opt-help:

``--help``

.. _desc-help:

Provides a list of ``dpct`` specific options.

.. _end-help:



.. _opt-helper-func-dir:

``--helper-function-dir``

.. _desc-helper-func-dir:

Print the installation directory for helper function header files.

.. _end-helper-func-dir:



.. _opt-helper-func-pref:

``--helper-function-preference=<value>``

.. _desc-helper-func-pref:

The preference of helper function usage in migration. Value:

- ``=no-queue-device``: Call SYCL API to get queue and device instead of calling helper function.

.. _end-helper-func-pref:



.. _opt-in-root:

``--in-root=<dir>``

.. _desc-in-root:

The directory path for the root of the source tree that needs to be migrated.
Only files under this root are migrated. Default:

- The current directory, if the input source files are not provided.
- The directory of the first input source file, if the input source files are provided.

Details:

- Any source within the directory specified by ``--in-root`` (at any nesting level)
  may be migrated.
- Any header file within the directory specified by ``--in-root`` (at any nesting
  level) that is included by the source or header file which is being migrated, is also
  migrated.
- Files from outside the ``--in-root`` directory will not be migrated even if
  they are included by any of the program source files.

.. _end-in-root:



.. _opt-in-root-exclude:

``--in-root-exclude=<dir|file>``

.. _desc-in-root-exclude:

Excludes the specified directory or file from processing.

.. _end-in-root-exclude:



.. _opt-keep-original-code:

``--keep-original-code``

.. _desc-keep-original-code:

Keeps the original code in the comments of generated SYCL files. Default: ``off``.

.. _end-keep-original-code:



.. _opt-no-cl-namespace-inline:

``--no-cl-namespace-inline``

.. _desc-no-cl-namespace-inline:

DEPRECATED: Do not use ``cl::`` namespace inline. Default: ``off``. This
option will be ignored if the replacement option ``--use-explicit-namespace``
is used.

.. _end-no-cl-namespace-inline:



.. _opt-no-dpcpp-extensions:

``--no-dpcpp-extensions=<value>``

.. _desc-no-dpcpp-extensions:

A comma-separated list of extensions not to be used in migrated code.
By default, these extensions are used in migrated code.

The values are:

- ``=bfloat16``: The SYCL extensions for bfloat16.
- ``=device_info``: The Intel extensions for device information if supported
  by the compiler and the backend.
- ``=enqueued_barriers``: The enqueued barriers extension.

.. _end-no-dpcpp-extensions:



.. _opt-no-dry-pattern:

``--no-dry-pattern``

.. _desc-no-dry-pattern:

Do not use a Don't Repeat Yourself (DRY) pattern when functions from the
``dpct`` namespace are inserted. Default: ``off``.

.. _end-no-dry-pattern:



.. _opt-no-incremental-migration:

``--no-incremental-migration``

.. _desc-no-incremental-migration:

Tells the tool to not perform an incremental migration. Default: ``off``
(incremental migration happens).

.. _end-no-incremental-migration:



.. _opt-optimize-migration:

``--optimize-migration``

.. _desc-optimize-migration:

Generates SYCL code applying more aggressive assumptions that
potentially may alter the semantics of your program. Default: ``off``.

.. _end-optimize-migration:



.. _opt-out-root:

``--out-root=<dir>``

.. _desc-out-root:

The directory path for root of generated files. A directory is created if
it does not exist. Default: ``dpct_output``.

The relative paths for the generated files are maintained. By default, file
extensions are changed as follows:

- ``*.cu → *.dp.cpp``
- ``*.cpp → *.cpp.dp.cpp``
- ``*.cc → *.cc.dp.cpp``
- ``*.cxx → *.cxx.dp.cpp``
- ``*.C → *.C.dp.cpp``
- ``*.cuh → *.dp.hpp``
- ``*.h *.hpp *.hxx`` → extensions are kept the same

To limit file extension changes to ``.cu`` and ``.cuh`` files only, use the
``--change-cuda-files-extension-only`` option.

.. _end-out-root:



.. _opt-output-file:

``--output-file=<file>``

.. _desc-output-file:

Redirects the ``stdout``/``stderr`` output to ``<file>`` in the
output directory specified by the ``--out-root`` option.

.. _end-output-file:



.. _opt-output-verbosity:

``--output-verbosity=<value>``

.. _desc-output-verbosity:

Sets the output verbosity level:

- ``=detailed``: 'normal' and messages about which file is being processed.
- ``=diagnostics``: 'detailed' and information about the detected conflicts
  and crashes (default).
- ``=normal``: 'silent' and warnings, errors, and notes from |tool_name|.
- ``=silent``: Only messages from clang.

.. _end-output-verbosity:




.. _opt-p:

``-p``

.. _desc-p:

Alias for ``--compilation-database``.

.. _end-p:




.. _opt-process-all:

``--process-all``

.. _desc-process-all:

Migrates or copies all files, except hidden, from the ``--in-root``
directory to the ``--out-root`` directory. The ``--in-root`` option should
be explicitly specified. Default: ``off``.

Details:

If ``--process-all`` and ``--in-root`` options are specified, but no
input files are provided, the tool migrates or copies all files, except
hidden, from the ``--in-root`` directory to the output directory.

- If there is a compilation database:

  - Files from the compilation database are migrated with the options
    specified in the compilation database
  - Files with the ``.cu`` extension that are not listed in the compilation
    database are migrated as standalone
  - Remaining files are copied to the ``–out-root`` directory

- If there is no compilation database:

  - Files with the ``.cu`` extension are migrated as standalone
  - Remaining files are copied to the ``-out-root`` directory

``--process-all`` is ignored if input files are provided on the command line.

.. _end-process-all:




.. _opt-query-api-map:

``-query-api-mapping=<api>``

.. _desc-query-api-map:

Query functionally-compatible SYCL API to migrate CUDA API.

.. _end-query-api-map:




.. _opt-report-file-prefix:

``--report-file-prefix=<prefix>``

.. _desc-report-file-prefix:

Prefix for the report file names. The full file name will have a suffix
derived from the ``report-type`` and an extension derived from the
``report-format``. For example: ``<prefix>.apis.csv`` or ``<prefix>.stats.log``.
If this option is not specified, the report goes to ``stdout``. The report
files are created in the directory, specified by ``-out-root``.

.. _end-report-file-prefix:



.. _opt-report-format:

``--report-format=<value>``

.. _desc-report-format:

Format of the reports:

- ``=csv``: The output is lines of comma-separated values. The report name
  extension will be ``.csv`` (default).
- ``=formatted``: The output is formatted for easier human readability.
  The report file name extension is ``log``.

.. _end-report-format:



.. _opt-report-only:

``--report-only``

.. _desc-report-only:

Only reports are generated. No SYCL code is generated. Default: ``off``.

.. _end-report-only:



.. _opt-report-type:

``--report-type=<value>``

.. _desc-report-type:

Specifies the type of report. Values are:

- ``=all``: All reports.
- ``=apis``: Information about API signatures that need migration and the
  number of times they were encountered. The report file name has the
  ``.apis`` suffix added.
- ``=stats``: High level migration statistics: Lines Of Code (LOC) that
  are migrated to SYCL, LOC migrated to SYCL with helper functions,
  LOC not needing migration, LOC needing migration but are not migrated.
  The report file name has the ``.stats`` suffix added (default).

.. _end-report-type:



.. _opt-rule-file:

``--rule-file=<file>``

.. _desc-rule-file:

Specifies the rule file path that contains rules used for migration.

.. _end-rule-file:



.. _opt-stop-on-parse-err:

``--stop-on-parse-err``

.. _desc-stop-on-parse-err:

Stop migration and generation of reports if parsing errors happened. Default: ``off``.

.. _end-stop-on-parse-err:



.. _opt-suppress-warnings:

``--suppress-warnings=<value>``

.. _desc-suppress-warnings:

A comma-separated list of migration warnings to suppress. Valid warning IDs
range from 1000 to 1118. Hyphen-separated ranges are also allowed. For
example: ``-suppress-warnings=1000-1010,1011``.

.. _end-suppress-warnings:



.. _opt-suppress-warnings-all:

``--suppress-warnings-all``

.. _desc-suppress-warnings-all:

Suppresses all migration warnings. Default: ``off``.

.. _end-suppress-warnings-all:



.. _opt-sycl-named-lambda:

``--sycl-named-lambda``

.. _desc-sycl-named-lambda:

Generates kernels with the kernel name. Default: ``off``.

.. _end-sycl-named-lambda:



.. _opt-use-custom-helper:

``--use-custom-helper=<value>``

.. _desc-use-custom-helper:

DEPRECATED: Customize the helper header files for migrated code. The values are:

- ``=all``: Generate a complete set of helper header files and place them
  in the ``--out-root`` directory.
- ``=api``: Limit helper header files to only the necessary APIs for the
  migrated code and place them in the ``--out-root`` directory.
- ``=file``: Limit helper header files to only the necessary files for the
  migrated code and place them in the ``--out-root`` directory.
- ``=none``: No customization (default).

.. _end-use-custom-helper:



.. _opt-use-dpcpp-extensions:

``--use-dpcpp-extensions=<value>``

.. _desc-use-dpcpp-extensions:

A comma-separated list of extensions to be used in migrated code.
By default, these extensions are not used in migrated code.

- ``=c_cxx_standard_library``: Use std functions from the libdevice library
  (provided by |dpcpp_compiler|_) and C/C++ Standard Library to migrate functions
  which have no mapping in the SYCL standard. If this value is used together with
  ``intel_device_math``, the ``intel_device_math`` functions take precedence.
- ``=intel_device_math``: Use ``sycl::ext::intel::math`` functions from the libdevice
  library (provided by |dpcpp_compiler|) to migrate functions which have no
  mapping in the SYCL standard.

.. _end-use-dpcpp-extensions:



.. _opt-use-experimental-features:

``--use-experimental-features=<value>``

.. _desc-use-experimental-features:

A comma-separated list of experimental features to be used in migrated code.
By default, experimental features will not be used in migrated code.

The values are:

- ``=bfloat16_math_functions``: Experimental extension that allows use of bfloat16 math functions.
- ``=dpl-experimental-api``: Experimental extension that allows use of experimental
  oneDPL APIs.
- ``=free-function-queries``: Experimental extension that allows getting
  ``id``, ``item``, ``nd_item``, ``group``, and ``sub_group`` instances
  globally.
- ``=local-memory-kernel-scope-allocation``: Experimental extension that
  allows allocation of local memory objects at the kernel functor scope.
- ``=logical-group``: Experimental helper function used to logically
  group work-items.
- ``=masked-sub-group-operation``: Experimental helper function used to execute
  sub-group operation with mask.
- ``=matrix``: Experimental extension that allows use of matrix extension like class ``joint_matrix``.
- ``=nd_range_barrier``: Experimental helper function used to help cross-group synchronization during migration.
- ``=occupancy-calculation``: Experimental helper function used to calculate occupancy.
- ``=user-defined-reductions``: Experimental extension that allows user-defined
  reductions.

.. _end-use-experimental-features:



.. _opt-use-explicit-namespace:

``--use-explicit-namespace=<value>``

.. _desc-use-explicit-namespace:

Defines the namespaces to use explicitly in generated code. The value is
a comma-separated list. Default: ``dpct, sycl``.

Possible values are:

- ``=cl``: DEPRECATED. Generate code with ``cl::sycl::`` namespace. Cannot be
  used with ``sycl`` or ``sycl-math`` values.
- ``=dpct``: Generate code with ``dpct::`` namespace.
- ``=none``: Generate code without namespaces. Cannot be used with other values.
- ``=sycl``: Generate code with ``sycl::`` namespace. Cannot be used with ``cl``
  or ``sycl-math`` values.
- ``=sycl-math``: Generate code with ``sycl::`` namespace, applied only for SYCL
  math functions. Cannot be used with ``cl`` or ``sycl`` values.

.. _end-use-explicit-namespace:



.. _opt-usm-level:

``--usm-level=<value>``

.. _desc-usm-level:

Sets the Unified Shared Memory (USM) level to use in source code generation:

- ``=none``: Uses helper functions from |tool_name| header files
  for memory management migration.
- ``=restricted``: Uses USM API for memory management migration. (default).

.. _end-usm-level:

.. _opt-vcxprojfile:

``--vcxprojfile=<file>``

.. _desc-vcxprojfile:

The file path of ``vcxproj``.

.. _end-vcxprojfile:



.. _opt-version:

``--version``

.. _desc-version:

Shows the version of the tool.

.. _end-version:



.. _opt-compilation-db:

``--compilation-database=<dir>``

.. _desc-compilation-db:

The directory path for the compilation database (`compile_commands.json`). When no
path is specified, a search for `compile_commands.json` is attempted through all
parent directories of the first input source file. Same as ``-p``.

.. _end-compilation-db:


.. _opt-gen-helper-func:

``--gen-helper-function``

.. _desc-gen-helper-func:

Generates helper function files in the ``--out-root`` directory. Default: ``off``.

.. _end-gen-helper-func:




.. _opt-intercept-build-block:

intercept-build Options
-----------------------

The following table lists all current `intercept-build` tool command line options
in alphabetical order.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Option
     - Description
   * - `--append`
     - Extend existing compilation database with new entries. Duplicate entries are
       detected and not present in the final output. The output is not continuously
       updated; it's done when the build command finished. Default: disabled.
   * - `--cdb <file>`
     - The JSON compilation database. Default name: `compile_commands.json`.
   * - `--linker-entry`
     - Generate linker entry in compilation database if the `--linker-entry` option
       is present. Default: enabled.
   * - `--no-linker-entry`
     - Do not generate linker entry in compilation database if the `--no-linker-entry`
       option is present. Default: disabled.
   * - `--parse-build-log <file>`
     - Specifies the file path of the build log.
   * - `--verbose`, `-v`
     - Enable verbose output from `intercept-build`. A second, third, and fourth
       flag increases verbosity.
   * - `--work-directory <path>`
     - Specifies the working directory of the command that generates the build log
       specified by option `-parse-build-log`. Default: the directory of build log
       file specified by option `-parse-build-log`.

.. _end-intercept-build-block:


.. _report-opt-block:

Specifying any of the following options will trigger report generation:

-  ``--report-file-prefix``
-  ``--report-type``
-  ``--report-format``
-  ``--report-only``

.. _end-report-opt-block:
