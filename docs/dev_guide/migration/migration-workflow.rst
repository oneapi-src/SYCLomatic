.. _mig_workflow:

Migration Workflow Guidelines
=============================

Overview
--------

The CUDA* to SYCL* code migration workflow consists of the following high-level
stages:

* `Stage 1: Prepare for Migration`_. Prepare your project and configure the tool for
  a successful migration.
* `Stage 2: Migrate Your Code`_. Review tool options and migrate your code with the
  tool.
* `Stage 3: Review the Migrated Code`_. Review and manually convert any unmigrated code.
* `Stage 4: Build the New  SYCL Code Base`_. Build your project with the
  migrated code.
* `Stage 5: Validate the New SYCL Application`_. Validate your new SYCL application to
  check for correct functionality after migration.

This document describes the steps in each stage with general recommendations and
optional steps.

.. note::

   CUDA* API migration support is broad but not complete. If you encounter CUDA
   APIs that were not migrated due to a lack of tool support, please report it
   to the |sycl_forum|_ or
   `priority support <https://www.intel.com/content/www/us/en/developer/get-help/priority-support.html>`_.
   Alternatively, `submit an issue <https://github.com/oneapi-src/SYCLomatic/issues>`_ or
   `contribute to the SYCLomatic project <https://github.com/oneapi-src/SYCLomatic/blob/SYCLomatic/CONTRIBUTING.md>`_.
   This helps prioritize which CUDA APIs will be supported in future releases.

Prerequisites
-------------

.. include:: /_include_files/prereq_mig_flow.rst

Stage 1: Prepare for Migration
------------------------------

Before migrating your CUDA code to SYCL, prepare your CUDA source code for the
migration process.

Prepare Your CUDA Project
*************************

Before migration, it is recommended to prepare your CUDA project to minimize errors
during migration:

#. Make sure your CUDA source code has no syntax errors.
#. Make sure your CUDA source code is Clang compatible.

Fix Syntax Errors
~~~~~~~~~~~~~~~~~

If your original CUDA source code has syntax errors, it may result in unsuccessful
migration.

Before you start migration, make sure that your original CUDA source code builds
and runs correctly:

#. Compile your original source code using the compiler defined for your original
   CUDA project.
#. Run your compiled application and verify that it functions as expected.

When your code compiles with no build errors and you have verified that your
application works as expected, your CUDA project is ready for migration.

Clang Compatibility
~~~~~~~~~~~~~~~~~~~

|tool_name| uses the latest version of the Clang* parser to analyze your CUDA source
code during migration. The Clang parser isn’t always compatible with the NVIDIA*
CUDA compiler driver (nvcc). The tool will provide errors about incompatibilities
between nvcc and Clang during migration.

In some cases, additional manual edits to the CUDA source may be needed before
migration. For example:

* The Clang parser may need namespace qualification in certain usage scenarios
  where nvcc does not require them.
* The Clang parser may need additional forward class declarations where nvcc
  does not require them.

  Space within the triple brackets of kernel invocation is tolerated by nvcc but
  not Clang. For example, ``cuda_kernel<< <num_blocks, threads_per_block>> >(args…)``
  is ok for nvcc, but the Clang parser requires the spaces to be removed.

If you run the migration tool on CUDA source code that has unresolved
incompatibilities between nvcc and Clang parsers, you will get a mixture of errors
in the migration results:

* Clang errors, which must be resolved in the CUDA source code
* DPCT warnings, which must be resolved in the migrated SYCL code

For detailed information about dialect differences between Clang and nvcc, refer to llvm.org's
`Compiling CUDA with clang <https://www.llvm.org/docs/CompileCudaWithLLVM.html>`_ page.

Run CodePin to Capture Application Signature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CodePin is a feature that helps reduce the effort of debugging inconsistencies in
runtime behavior. CodePin generates reports from the CUDA and SYCL programs
that, when compared, can help identify the source of divergent runtime behavior.

Enable the CodePin tool during the migration in order to capture the project
signature.

This signature will be used later for validation after migration.

Enable CodePin with the ``–enable-codepin`` option.

For detailed information about debugging using the CodePin tool, refer to
`Debug Migrated Code Runtime Behavior <https://www.intel.com/content/www/us/en/docs/dpcpp-compatibility-tool/developer-guide-reference/2024-2/debug-with-codepin.html>`_.

Configure the Tool
******************

CUDA header files used by your project must be accessible to the tool. If you have
not already done so, configure the tool and ensure header files are available.

.. include:: /_include_files/before_begin_intro_dgr.rst

Record Compilation Commands
***************************


Use ``intercept-build`` to :ref:`gen_comp_db` to capture the detailed
build options for your project. The migration tool uses build information from
the database (such as header file paths, include paths, macro definitions, compiler
options, and the original compiler) to guide the migration of your CUDA code.

If your development environment prevents you from using intercept-build, use the
alternate method described in :ref:`gen_db_other_sys`.

.. note::

   If you need to re-run your migration after the original migration and the CUDA
   build script has changed, you need to either

   #. re-run intercept-build to get an updated compilation database to use in your
      migration or
   #. manually update the compilation database to capture the changes from the
      updated CUDA build script.

Set Up Revision Control
***********************

After migration, the recommendation is to maintain and develop your migrated
application in SYCL to avoid vendor lock-in, though you may choose to continue
your application development in CUDA. Continuing to develop in CUDA will result
in the need to migrate from CUDA to SYCL again.

Revision control allows comparison between versions of migrated code, which can
help you decide what previous manual changes to the SYCL code you want to merge
into the newly migrated code.

Make sure to have revision control for your original CUDA source before the first
migration. After the first migration, be sure to place the migrated SYCL code,
with all subsequent manual SYCL changes, under revision control as well.

Run Analysis Mode
*****************

You can use :ref:`analysis_mode` to generate a report before migration that will
indicate how much of your code will be migrated, how much will be partially migrated,
and an estimate of the manual effort needed to complete migration after you have
run the tool. This can be helpful to estimate the work required for your migration.

Stage 2: Migrate Your Code
--------------------------

Plan Your Migration
*******************

Before executing your migration, review the available tool features and options
that can be used to plan your specific migration.

Migration Rules
~~~~~~~~~~~~~~~

The tool uses a default set of migration rules for all migrations. If default
rules do not give the migration results you need, you can define custom rules for
your migration. This is helpful in multiple scenarios, for example:

* After migration, you discover multiple instances of similar or identical CUDA
  source code that were not migrated, and you know how the CUDA source code should
  be migrated to SYCL. In this case, you can define a custom rule and re-run the
  migration for better results. This is useful for :ref:`inc_mig` or scenarios
  where you may run multiple migrations over time.

* You know before migration that some code patterns in your original CUDA source
  will not be accurately migrated to SYCL using the built-in rules. In this case,
  you can define a custom migration rule to handle specific patterns in our CUDA
  source during migration.

For detailed information about defining custom rules, refer to :ref:`migration_rules`.

For working examples of custom rules, refer to the optional predefined rules
located in the ``extensions/opt_rules`` folder on the installation path of the tool.

Incremental Migration
~~~~~~~~~~~~~~~~~~~~~

.. include:: incremental-migration.rst
          :start-after: inc-mig-intro:
          :end-before: inc-mig-intro-end:

For detailed information and examples of incremental migration, refer to :ref:`inc_mig`.

Command-Line Options
~~~~~~~~~~~~~~~~~~~~

|tool_name| provides many command-line options to direct your migration.
Command-line options provide control to

* Configure your migration with :ref:`basic <mig_basic_opt>` and :ref:`advanced <adv_mig_opt>` options
* :ref:`Control code generation <code_gen_opt>`
* :ref:`Generate migration reports <report_gen_opt>`
* :ref:`Migrate build scripts <build_script_opt>`
* :ref:`Control warnings <warn_opt>`
* :ref:`Query API mapping <query_api_opt>`

Refer to the :ref:`alpha_opt` for a full list of all available command-line
options.

Buffer vs USM Code Generation
*****************************

Intel promotes both buffer and USM in the SYCL/oneAPI context. Some oneAPI libraries preferentially support buffer versus USM, so there may be some design consideration in configuring your migration. USM is used by default, but buffer may be a better fit for some projects.

The buffer model sets up a 1-3 dimensional array (buffer) and accesses its components via a C++ accessor class. This grants more control over the exact nature and size of the allocated memory, and how host and offload target compute units access it.
However, the buffer model can also create extra class management overhead, which can require more manual intervention and may yield less performance.

USM (unified shared memory) is a newer model, beginning with SYCL2020. USM is a pointer-based memory management model using
``malloc_device/malloc_shared/malloc_host``  allocator functions, similar to how C++ code usually handles memory accesses when no GPU device offload is involved. Choosing the USM model can make it easier to add to existing code and migrate from CUDA code.  Management of the USM memory space is however very much done by the SYCL runtime, reducing granularity of control for the developer.

* `SYCL Buffer Specification <https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:data.access.and.storage>`_
* `SYCL USM Specification <https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:usm>`_

For more information on USM versus Buffer modes, please see the following sections of the GPU Optimization Guide:
* `Unified Shared Memory Allocations <https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/usm-allocation.html>`_
* `Buffer Accessor Modes <https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/buffer-accessors.html>`_

What to Expect in Migrated Code
*******************************

When the tool migrates CUDA code to SYCL code, it inserts diagnostic messages as
comments in the migrated code. The DPCT diagnostic messages are logged as comments
in the migrated source files and output as warnings to the console during migration.
These messages identify areas in the migrated code that may require your attention
to make the code SYCL compliant or correct. This step is detailed in
`Stage 3: Review the Migrated Code`_.

The migrated code also uses DPCT helper functions to provide utility support for
the generated SYCL code. The helper functions use the ``dpct:: namespace``. Helper
function source files are located at ``<tool-installation-directory>/latest/include/dpct``.
DPCT helper functions can be left in migrated code but should not be used in new
SYCL code. Use standard SYCL and C++ when writing new code. For information about
the DPCT namespace, refer to the :ref:`dpct_name_ref`.

Run Migration
*************

After reviewing the available migration tool functionality and options, run your
migration.

.. include:: /_include_files/run_migration.rst

If your project uses a Makefile or CMake file, use the corresponding option to
automatically migrate the file to work with the migrated code:

* To migrate a Makefile, use the ``--gen-build-scripts`` option.
* To migrate a CMake file, use the ``--migrate-build-script`` or
  ``--migrate-build-script-only`` option. (Note that these options are experimental.)

For example:

.. code-block::

   c2s -p compile_commands.json --in-root ../../.. --gen-helper-function --gen-build-scripts

This example migrate command:

* uses the tool alias ``c2s``. ``dpct`` can also be used.
* uses a compilation database, specified with the ``-p`` option
* specifies the source to be migrated with the ``--in-root`` option
* instructs the tool to generate helper function files with the ``--gen-helper-function`` option
* instructs the tool to migrate the Makefile using the ``--gen-build-scripts`` option

The following samples show migrations of CUDA code using the tool and targets Intel and NVIDIA* hardware:

* |convo_sep_sample|_
* |conc_kern_sample|_

Stage 3: Review the Migrated Code
---------------------------------

After running |tool_name|, manual editing is usually required before the migrated
SYCL code can be compiled. DPCT warnings are logged as comments in the migrated
source files and output to the console during migration. These warnings identify
the portions of code that require manual intervention. Review these comments and
make the recommended changes to ensure the migrated code is consistent with the
original logic.

.. include:: ../reference/diagnostic_ref/dpct1009.rst
          :start-after: example_dpct:
          :end-before: example_dpct_end:

Note the :ref:`DPCT1009` warning inserted where additional review is needed.

For a detailed explanation of the comments, including suggestions to fix the issues,
refer to the :ref:`diag_ref`.

At this stage, you may observe that the same DPCT warnings were generated repeatedly
in your code or that the same manual edits were needed in multiple locations to fix
a specific pattern in your original source code. Consider defining the manual edits
needed to fix repeated DPCT warnings as a
:ref:`user-defined migration rule <migration_rules>`. This allows
you to save your corrections and automatically apply them to a future migration of
your CUDA source.

.. note::

   CUDA* API migration support is broad but not complete. If you encounter CUDA
   APIs that were not migrated due to a lack of tool support, please report it
   to the |sycl_forum|_ or
   `priority support <https://www.intel.com/content/www/us/en/developer/get-help/priority-support.html>`_.
   Alternatively, `submit an issue <https://github.com/oneapi-src/SYCLomatic/issues>`_ or
   `contribute to the SYCLomatic project <https://github.com/oneapi-src/SYCLomatic/blob/SYCLomatic/CONTRIBUTING.md>`_.
   This helps prioritize which CUDA APIs will be supported in future releases.

Stage 4: Build the New SYCL Code Base
-------------------------------------

After you have completed any manual migration steps, build your converted code.

Install New SYCL Code Base Dependencies
***************************************

Converted code makes use of oneAPI library APIs and Intel SYCL extensions. Before
compiling, install the appropriate oneAPI libraries and a compiler that supports
the Intel SYCL extensions.


.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - If your CUDA source uses ...
     - ... install this oneAPI library
   * - cuBLAS, cuFFT, cuRAND, cuSolver, cuSparse
     - Intel® oneAPI Math Kernel Library (oneMKL)
   * - Thrust, CUB
     - Intel® oneAPI DPC++ Library (oneDPL)
   * - cuDNN
     - Intel® oneAPI Deep Neural Network Library (oneDNN)
   * - NCCL
     - Intel® oneAPI Collective Communications Library (oneCCL)

The following compilers support Intel SYCL extensions:

* |dpcpp_compiler|_
* |oneapi_dpcpp_compiler|_

Most libraries and the |dpcpp_compiler| are included in the |base_kit_long|_. Libraries
and the compiler are also available as stand-alone downloads.

Compile for Intel CPU and GPU
*****************************

If your program targets Intel GPUs, install the latest Intel GPU drivers before
compiling.

* `Linux GPU Drivers <https://dgpu-docs.intel.com/driver/installation.html>`_
* `Windows GPU Drivers <https://www.intel.com/content/www/us/en/support/products/80939/graphics.html>`_

Use your updated Makefile or CMake file to build your program, or compile it
manually at the command line using a compiler that supports the Intel SYCL extensions.
Make sure that all linker and compilation commands use the ``-fsycl`` compiler
option with the C++ driver. For example:

.. code-block::

   icpx -fsycl migrated-file.cpp

For detailed information about compiling with the |dpcpp_compiler|, refer to the
|compiler_dev_guide|_.

Compile for AMD* or NVIDIA* GPU
*******************************

If your program targets AMD* or NVIDIA GPUs, install the appropriate Codeplay*
plugin for the target GPU before compiling. Instructions for installing the AMD
and NVIDIA GPU plugins, as well as how to compile for those targets, can be found
in the Codeplay plugin documentation:

* `Install the oneAPI for AMD GPUs plugin <https://developer.codeplay.com/products/oneapi/amd/guides/>`_ from Codeplay.
* `Install the oneAPI for NVIDIA GPUs plugin <https://developer.codeplay.com/products/oneapi/nvidia/guides/>`_ from Codeplay.

Stage 5: Validate the New SYCL Application
------------------------------------------

After you have built your converted code, validate your new SYCL application to
check for correct functionality after migration.

Use a Debugger to Validate Migrated Code
****************************************

After you have successfully compiled your new SYCL application, run the app in
debug mode using a debugger such as 
`Intel Distribution for GDB <https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-gdb.html>`_ to verify that your application runs
as expected after migration.

Learn more about `Debugging with Intel Distribution for GDB <https://www.intel.com/content/www/us/en/docs/distribution-for-gdb/tutorial-debugging-dpcpp-linux/2024-1/overview.html>`_. 

Use CodePIN to Validate Migrated Code
*************************************

If the CodePin feature has been enabled during the migration time,
project signature will be logged during the execution time.

The signature contains the data value of each execution checkpoint, which can be verified manually or with an auto-analysis tool.

For detailed information about debugging using the CodePin tool, refer to
`Debug Migrated Code Runtime Behavior <https://www.intel.com/content/www/us/en/docs/dpcpp-compatibility-tool/developer-guide-reference/2024-2/debug-with-codepin.html>`_.

Optimize Your Code
------------------

Optimize your migrated code for Intel GPUs using Intel® tools such as
|vtune|_ and |advisor|_. These tools help identify areas of
code to improve for optimizing your application performance.

Additional hardware- or library-specific optimization information is available:

* For detailed information about optimizing your code for Intel GPUs, refer to
  the `oneAPI GPU Optimization Guide <https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-2/overview.html>`_.
* For detailed information about optimizing your code for AMD GPUs, refer to the
  `Codeplay AMD GPU Performance Guide <https://developer.codeplay.com/products/oneapi/amd/2024.0.2/guides/performance/introduction>`_.
* For detailed information about optimizing your code for NVIDIA GPUS, refer to
  the `Codeplay NVIDIA GPU Performance Guide <https://developer.codeplay.com/products/oneapi/nvidia/2024.0.2/guides/performance/introduction>`_.


Find More
---------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Content
     - Description
   * - |compiler_dev_guide|_
     - Developer guide and reference for the Intel® oneAPI DPC++/C++ Compiler.
   * - `SYCL 2020 Specification <https://www.khronos.org/registry/SYCL/specs/sycl-2020/pdf/sycl-2020.pdf>`_
     - The SYCL 2020 Specification PDF.
   * - |dpcpp_compiler|_
     - Intel branded C++ compiler built from the open-source oneAPI DPC++ Compiler, with additional Intel hardware optimization.
   * - |oneapi_dpcpp_compiler|_
     - Open-source Intel LLVM-based compiler project that implements compiler and runtime support for the SYCL* language.
   * - `Basic migration samples <https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/Migration>`_
     - Sample CUDA projects with instructions on migrating to SYCL using the tool.
   * - Guided migration samples
     - Guided migration of two sample NVIDIA CUDA projects:

       * |convo_sep_sample|_
       * |conc_kern_sample|_
   * - `Jupyter notebook samples <https://github.com/oneapi-src/oneAPI-samples/tree/development/DirectProgramming/C%2B%2BSYCL/Jupyter/cuda-to-sycl-migration-training>`_
     - A Jupyter* Notebook that guides you through the migration of a simple example and four step-by-step sample migrations from CUDA to SYCL.
   * - `CUDA* to SYCL* Catalog <https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-cuda-to-sycl-library.html>`_
     - Catalog of CUDA projects that have been migrated to SYCL.
   * - |sycl_forum|_
     - Forum to get assistance when migrating your CUDA code to SYCL.
   * - `Intel® oneAPI Math Kernel Library Link Line Advisor <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html>`_
     - Intel® oneAPI Math Kernel Library tool to help determine how to include oneMKL libraries for your specific use case.
   * - `Tutorial: Debugging with Intel® Distribution for GDB* <https://www.intel.com/content/www/us/en/docs/distribution-for-gdb/tutorial-debugging-dpcpp-linux/2024-1/overview.html>`_
     - This tutorial describes the basic scenarios of debugging applications using Intel® Distribution for GDB*.
   * - `Intel® VTune™ Profiler Tutorials <https://www.intel.com/content/www/us/en/developer/articles/training/vtune-profiler-tutorials.html>`_
     - Tutorials demonstrating an end-to-end workflow using Intel® VTune™ Profiler that you can ultimately apply to your own applications.