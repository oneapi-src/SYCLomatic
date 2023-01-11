.. _cmd_opt_ref:

Command Line Options Reference
==============================

This topic shows the command line options with a short description, current
deprecated options, and information for working with source files.

Command Line Options
--------------------

The following table lists all current |tool_name| command line options
in alphabetical order.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Option
     - Description
   * - ``--always-use-async-handler``
     - Use async exception handler when creating new ``sycl::queue`` 
       with ``dpct::create_queue`` in addition to default 
       ``dpct::get_default_queue``. Default: ``off``.
   * - ``--analysis-scope-path=<dir>``
     - The directory path for the analysis scope of the source tree that needs
       to be migrated. Default: the value of ``--in-root``.
   * - ``--assume-nd-range-dim=<value>``
     - Provides a hint to the tool on the dimensionality of nd_range to use in
       generated code. The values are:

       - ``=1``: Generate kernel code assuming 1D ``nd_range`` where possible, 
         and 3D in other cases.
       - ``=3``: Generate kernel code assuming 3D ``nd_range`` (default).
   * - ``--build-script-file=<file>``
     - Specifies the name of generated makefile for migrated file(s). Default
       name: ``Makefile.dpct``.
   * - ``=c_cxx_standard_library``
     - A set of functions from the C and C++ standard libraries that are allowed
       to be used in SYCL device code.
   * - ``--check-unicode-security``
     - Enable detection and warnings about Unicode constructs that can be
       exploited by using bi-directional formatting codes and homoglyphs in
       identifiers. Default: ``off``.
   * - ``--comments``
     - Insert comments explaining the generated code. Default: ``off``.
   * - ``--cuda-include-path=<dir>``
     - The directory path of the CUDA\* header files.
   * - ``--custom-helper-name=<name>``
     - DEPRECATED: Specifies the helper headers folder name and main helper
       header file name. Default: ``dpct``.
   * - ``--enable-ctad``
     - Use a C++17 class template argument deduction (CTAD) in your generated code.
   * - ``--extra-arg=<string>``
     - Additional argument to append to the migration command line, example:
       ``--extra-arg="-I /path/to/header"``. The options that can be passed this
       way can be found with the ``dpct -- -help`` command.
   * - ``--format-range=<value>``
     - Sets the range of formatting.

       The values are:

       - ``=migrated``: Only formats the migrated code (default).
       - ``=all``: Formats all code.
       - ``=none``: Do not format any code.
   * - ``--format-style=<value>``
     - Sets the formatting style.

       The values are:

       - ``=llvm``: Use the LLVM coding style.
       - ``=google``: Use the Google\* coding style.
       - ``=custom``: Use the coding style defined in the ``.clang-format`` file (default).

       Example for the .clang-format file content:

       ::

          BasedOnStyle:
          LLVM IndentWidth: 4
          TabWidth: 4
          UseTab: ForIndentation
   * - ``--gen-build-script``
     - Generates makefile for migrated file(s) in ``-out-root`` directory.
       Default: ``off``.
   * - ``--help``
     - Provides a list of ``dpct`` specific options.
   * - ``--in-root=<dir>``
     - The directory path for the root of the source tree that needs to be migrated.
       Only files under this root are migrated. Default:

       - The current directory, if the input source files are not provided.
       - The directory of the first input source file, if the input source files are provided.

       Details:

       Any source within the directory specified by ``--in-root`` (at any nesting
       level) may be migrated. Any header file within the directory specified by
       ``--in-root`` (at any nesting level) included by the source or header file,
       which is being migrated, is also migrated. Files from outside the ``--in-root``
       directory are considered system files and they will not be migrated even
       if they are included by any of the program source files.
   * - ``--in-root-exclude=<dir|file>``
     - Excludes the specified directory or file from processing.
   * - ``--keep-original-code``
     - Keeps the original code in the comments of generated SYCL files. Default: ``off``.
   * - ``--no-cl-namespace-inline``
     - DEPRECATED: Do not use ``cl::`` namespace inline. Default: ``off``. This
       option will be ignored if the replacement option ``--use-explicit-namespace``
       is used.
   * - ``--no-dpcpp-extensions=<value>``
     - A comma-separated list of extensions not to be used in migrated code.
       By default, these extensions will be used in migrated code.

       - ``=enqueued_barriers``: Enqueued barriers extension.
   * - ``--no-dry-pattern``
     - Do not use a Don't Repeat Yourself (DRY) pattern when functions from the
       ``dpct`` namespace are inserted. Default: ``off``.
   * - ``--no-incremental-migration``
     - Tells the tool to not perform an incremental migration. Default: ``off``
       (incremental migration happens).
   * - ``--optimize-migration``
     - Generates SYCL code applying more aggressive assumptions that
       potentially may alter the semantics of your program. Default: ``off``.
   * - ``--out-root=<dir>``
     - The directory path for root of generated files. A directory is created if
       it does not exist. Default: ``dpct_output``.

       The relative paths for the generated files are maintained, and the
       extension is changed as follows:

       - ``*.cu → *.dp.cpp``
       - ``*.cpp → *.cpp.dp.cpp``
       - ``*.cc → *.cc.dp.cpp``
       - ``*.cxx → *.cxx.dp.cpp``
       - ``*.C → *.C.dp.cpp``
       - ``*.cuh → *.dp.hpp``
       - ``*.h *.hpp *.hxx`` → extensions are kept the same
   * - ``--output-file=<file>``
     - Redirects the ``stdout``/``stderr`` output to ``<file>`` in the
       output directory specified by the ``--out-root`` option.
   * - ``--output-verbosity=<value>``
     - Sets the output verbosity level:

       - ``=silent``: Only messages from clang.
       - ``=normal``: 'silent' and warnings, errors, and notes from |tool_name|.
       - ``=detailed``: 'normal' and messages about which file is being processed.
       - ``=diagnostics``: 'detailed' and information about the detected conflicts
         and crashes (default).
   * - ``-p=<dir>``
     - The directory path for the compilation database (``compile_commands.json``).
       When no path is specified, a search for ``compile_commands.json`` is
       attempted through all parent directories of the first input source file.
   * - ``--process-all``
     - Migrates or copies all files, except hidden, from the ``--in-root``
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
   * - ``--report-file-prefix=<prefix>``
     - Prefix for the report file names. The full file name will have a suffix
       derived from the ``report-type`` and an extension derived from the
       ``report-format``. For example: ``<prefix>.apis.csv`` or ``<prefix>.stats.log``.
       If this option is not specified, the report goes to ``stdout``. The report
       files are created in the directory, specified by ``-out-root``.
   * - ``--report-format=<value>``
     - Format of the reports:

       - ``=csv``: The output is lines of comma-separated values. The report name
         extension will be ``.csv`` (default).
       - ``=formatted``: The output is formatted for easier human readability.
         The report file name extension is ``log``.
   * - ``--report-only``
     - Only reports are generated. No SYCL code is generated. Default: ``off``.
   * - ``--report-type=<value>``
     - Specifies the type of report. Values are:

       - ``=apis``: Information about API signatures that need migration and the
         number of times they were encountered. The report file name has the
         ``.apis`` suffix added.
       - ``=stats``: High level migration statistics: Lines Of Code (LOC) that
         are migrated to SYCL, LOC migrated to SYCL with helper functions,
         LOC not needing migration, LOC needing migration but are not migrated.
         The report file name has the ``.stats`` suffix added (default).
       - ``=all``: All reports.
   * - ``--rule-file=<file>``
     - Specifies the rule file path that contains rules used for migration.
   * - ``--stop-on-parse-err``
     - Stop migration and generation of reports if parsing errors happened. Default: ``off``.
   * - ``--suppress-warnings=<value>``
     - A comma-separated list of migration warnings to suppress. Valid warning IDs
       range from 1000 to 1100. Hyphen-separated ranges are also allowed. For
       example: ``-suppress-warnings=1000-1010,1011``.
   * - ``--suppress-warnings-all``
     - Suppresses all migration warnings. Default: ``off``.
   * - ``--sycl-named-lambda``
     - Generates kernels with the kernel name. Default: ``off``.
   * - ``--use-custom-helper=<value>``
     - DEPRECATED: Customize the helper header files for migrated code. The values are:

       - ``=none``: No customization (default).
       - ``=file``: Limit helper header files to only the necessary files for the
         migrated code and place them in the ``--out-root`` directory.
       - ``=api``: Limit helper header files to only the necessary APIs for the
         migrated code and place them in the ``--out-root`` directory.
       - ``=all``: Generate a complete set of helper header files and place them
         in the ``--out-root`` directory.
   * - ``--use-dpcpp-extensions=<value>``
     - A comma-separated list of extensions to be used in migrated code. By
       default, these extensions are not used in migrated code.
   * - ``--use-experimental-features=<value>``
     - A comma-separated list of experimental features to be used in migrated code.
       By default, experimental features will not be used in migrated code.

       The values are:

       - ``=free-function-queries``: Experimental extension that allows getting
         ``id``, ``item``, ``nd_item``, ``group``, and ``sub_group`` instances
         globally.
       - ``=local-memory-kernel-scope-allocation``: Experimental extension that
         allows allocation of local memory objects at the kernel functor scope.
       - ``=logical-group``: Experimental helper function used to logically
         group work-items.
       - ``=nd_range_barrier``: Experimental helper function used to help cross
         group synchronization during migration.
   * - ``--use-explicit-namespace=<value>``
     - Defines the namespaces to use explicitly in generated code. The value is
       a comma-separated list. Default: ``dpct, sycl``.

       Possible values are:

       - ``=none``: Generate code without namespaces. Cannot be used with other
         values.
       - ``=cl``: DEPRECATED. Generate code with ``cl::sycl::`` namespace. Cannot
         be used with ``sycl`` or ``sycl-math`` values.
       - ``=dpct``: Generate code with ``dpct::`` namespace.
       - ``=sycl``: Generate code with ``sycl::`` namespace. Cannot be
         used with ``cl`` or ``sycl-math`` values.
       - ``=sycl-math``: Generate code with ``sycl::`` namespace, applied only
         for SYCL math functions. Cannot be used with ``cl`` or ``sycl`` values.
   * - ``--usm-level=<value>``
     - Sets the Unified Shared Memory (USM) level to use in source code generation:

       - ``=restricted``: Uses USM API for memory management migration. (default).
       - ``=none``: Uses helper functions from |tool_name| header files
         for memory management migration.
   * - ``--vcxprojfile=<file>``
     - The file path of ``vcxproj``.
   * - ``--version``
     - Shows the version of the tool.


.. note::

   Specifying any of these options will trigger report generation.

   -  ``--report-file-prefix``
   -  ``--report-type``
   -  ``--report-format``
   -  ``--report-only``

Deprecated Command Line Options
-------------------------------

The following table lists |tool_name| command line options that are 
currently deprecated.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Deprecated Options
     - Suggested Replacement
   * - ``--no-cl-namespace-inline``
     - ``--use-explicit-namespace``


Source Files
------------

To work with source files use ``<source0> ...`` to create paths
for your input source files. These paths can be found in the
compilation database.

Examples:

-  Migrate single source file: ``dpct source.cpp``
-  Migrate single source file with C++11 features:
   ``dpct --extra-arg="-std=c++11" source.cpp``
-  Migrate all files available in compilation database:
   ``dpct -p=<path to location of compilation database file>``
-  Migrate one file in compilation database:
   ``dpct -p=<path to location of compilation database file> source.cpp``
