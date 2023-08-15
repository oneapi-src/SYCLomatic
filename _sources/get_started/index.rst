.. _get_started:

Get Started with |tool_name|
============================

|tool_name| assists in the migration of your existing CUDA\* code to SYCL\* code.
The tool ports CUDA language kernels and library API calls, migrating 90%-95% of
CUDA code to SYCL code. The tool additionally inserts inline warnings to help you
complete the migration and tune your code.

|tool_name| supports migrating programs implemented with CUDA versions
|cuda_versions|. The list of supported languages and versions may be extended in
the future.

.. include:: /_include_files/intro_links_gsg.rst


Before You Begin
----------------

#. Install the tool.

   .. include:: /_include_files/before_begin_intro_gsg.rst

#. Make sure CUDA headers are accessible to the tool.

   Certain CUDA header files (specific to your project) may need to be accessible
   to |tool_name|. The tool looks for these CUDA header files in the following
   default locations:

   -  ``/usr/local/cuda/include``

   -  ``/usr/local/cuda-x.y/include``, where *x.y* is one of these values:
      |cuda_versions|.



   .. note::

      .. include:: /_include_files/alt_cuda_header_files.rst

      The CUDA include path should not be the same as, or a child path of, the
      directory where the source code (that needs to be migrated) is located.


#. Install a compiler that supports the DPC++ -specific extensions used in code
   migrated by |tool_name|.

   * |dpcpp_compiler|_
   * `oneAPI DPC++ Compiler <https://github.com/intel/llvm>`_

#. .. include:: /_include_files/env_setup_gsg.rst

#. Optional: If your program targets GPUs, install the appropriate GPU drivers or
   plug-ins to compile your program to run on Intel, AMD*, or NVIDIA* GPUs.

   - To use an Intel GPU, `install the latest Intel GPU drivers <https://dgpu-docs.intel.com/installation-guides/index.html>`_.
   - To use an AMD GPU, `install the oneAPI for AMD GPUs plugin <https://developer.codeplay.com/products/oneapi/amd/guides/>`_.
   - To use an NVIDIA GPU, `install the oneAPI for NVIDIA GPUs plugin <https://developer.codeplay.com/products/oneapi/nvidia/guides/>`_.


Run the Tool
------------

.. include:: /_include_files/run_tool_cmd.rst

To see the list of the language parser (Clang\*) options, pass ``-help``
as the Clang option:

.. code-block::

   dpct -- -help

.. include:: /_include_files/cross_ref_links_gsg.rst
          :start-after: refer-cmd-ref:
          :end-before: refer-cmd-ref-end:


Specify Files to Migrate
------------------------

If no directory or file is specified for migration, the tool will try to migrate
source files found in the current directory. The default output directory is
``dpct_output``. Use the ``--out-root`` option to specify an output directory.

You can optionally provide file paths for source files that should be migrated.
The paths can be found in the compilation database. The following examples show
ways to specify a file or directory for migration.

* Migrate a single source file:

  .. code-block::

     dpct source.cpp

* Migrate all files available in the compilation database:

  .. code-block::

     dpct -p=<path to the location of compilation database file>

* Migrate one file in the compilation database:

  .. code-block::

     dpct -p=<path to the location of compilation database file> source.cpp

* Migrate source files in the directory specified by the ``--in-root`` option and
  place generated files in the directory specified by the ``--out-root`` option:

  .. code-block::

     dpct --in-root=foo --out-root=bar


Understand Emitted Warnings
---------------------------

During file migration, |tool_name| identifies the places in the code that may
require your attention to make the code SYCL-compliant or correct.

Warnings are inserted into the generated source files and displayed as warnings
in the output. For example:

.. code-block::

   /path/to/file.hpp:26:1: warning: DPCT1003:0: Migrated API does not return error code. (*,0) is inserted. You may need to rewrite this code.
   // source code line for which warning was generated
   ^

.. include:: /_include_files/cross_ref_links_gsg.rst
          :start-after: refer-diag-ref:
          :end-before: refer-diag-ref-end:


Get Code Samples
----------------

Use the |tool_name| code samples to get familiar with the migration process and
tool features.

.. include:: /_include_files/access_samples.rst

.. include:: /_include_files/samples.rst


Migrate the Vector Add Sample
-----------------------------

The Vector Add sample shows how to migrate a simple CUDA program to SYCL-compliant
code. The simple program adds two vectors of [1..N] and prints the result. The
program is intended for CPU.

The following steps show how to migrate the Vector Add sample using |tool_name|:

#. Download the `Vector Add sample <https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/Migration/vector-add-dpct>`_.

#. Navigate to the root of the Vector Add sample. The sample contains a single
   CUDA file, ``vector_add.cu``, located in the ``src`` folder.

#. From the root folder of the sample project, run |tool_name|:

   .. code-block::

      dpct --in-root=. src/vector_add.cu

   The ``--in-root`` option specifies the root location of the program sources
   that should be migrated. Only files and folders within the
   ``--in-root`` directory will be considered for migration by the tool. Files
   outside the ``--in-root`` directory  will not be migrated, even if
   they are included by a source file within the ``--in-root`` directory.
   By default, the migrated files are created in a new folder named ``dpct_output``.

   As a result of the migration command, you should see the new SYCL source file
   in the output folder:

   .. code-block::

      dpct_output
      └── src
          └── vector_add.dp.cpp

   The relative paths of the migrated files are maintained.

#. Inspect the migrated source code and address any DPCT warnings generated by the
   too.

   This sample should generate the following warning:

   .. code-block::

      warning: DPCT1003:0: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.

   .. include:: /_include_files/cross_ref_links_gsg.rst
          :start-after: refer-dpct1003:
          :end-before: refer-dpct1003-end:

   The reference explains that SYCL uses exceptions to report errors instead of
   error codes. In this instance, the tool removed the conditional
   statement to exit on failure and instead wrapped the code in a try block. The
   tool retained the error status variable from the original code and changed the
   source to always assign an error code of 0 to the variable.

   The reference provides suggestions for how to fix this warning. In this sample,
   manually resolve the issue by removing the variable status, since it is not
   needed.

#. Compile the migrated code:

   .. code-block::

      icpx -fsycl -I<install_dir>/include src/vector_add.dp.cpp

   where <install_dir> is the |tool_name| installation directory.

#. Run the migrated program:

   .. note::

      The Vector Add sample is for CPU. Make sure to target your CPU by using the
      `ONEAPI_DEVICE_SELECTOR environment variable <https://intel.github.io/llvm-docs/EnvironmentVariables.html#oneapi-device-selector>`_:

      .. code-block::

         ONEAPI_DEVICE_SELECTOR=*:cpu

   .. code-block::

      ./vector_add

   You should see a block of even numbers, indicating the result of adding two vectors: ``[1..N] + [1..N]``.

.. include:: /_include_files/cross_ref_links_gsg.rst
          :start-after: refer-migrate-proj:
          :end-before: refer-migrate-proj-end:


Find More
---------

.. include:: /_include_files/find_more_gsg.rst

