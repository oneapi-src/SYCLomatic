.. _get_started:

Get Started with |tool_name|
============================

|tool_name| assists in the migration of your existing CUDA\* code to SYCL\* code.
The tool ports CUDA language kernels and library API calls, migrating 90%-95% of
CUDA code to SYCL code. The tool additionally inserts inline warnings to help you
complete the migration and tune your code.

|tool_name| supports the migration of programs implemented with CUDA versions
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

   .. include:: /_include_files/alt_cuda_header_files.rst

   .. note::

      The CUDA include path should not be the same as, or a child path of, the
      directory where the source code that needs to be migrated is located.

#. Configure the tool environment.

   .. include:: /_include_files/env_setup_gsg.rst

#. Install the |dpcpp_compiler|.

   |tool_name| migrates CUDA code to SYCL code for the |dpcpp_compiler|_.

   If your program targets GPUs, install the appropriate GPU drivers or plug-ins (optional),
   so you can compile your program to run on Intel, AMD*, or NVIDIA* GPUs.

   - To use an Intel GPU, install the latest Intel GPU drivers.
   - To use an AMD GPU, install the oneAPI for AMD GPUs plugin.
   - To use an NVIDIA GPU, install the oneAPI for NVIDIA GPUs plugin.


Run the Tool
------------

.. include:: /_include_files/run_tool_cmd.rst

To see the list of the language parser (Clang\*) options, pass ``-help``
as the Clang option:

.. code-block::

   dpct -- -help


Specify Files to Migrate
------------------------

If no directory or file is specified for migration, the tool will try to migrate
source files found in the current directory. The default output directory is
``dpct_output``. Use the ``--out-root`` option to specify an output directory.

You can optionally provide file paths for source files that should be migrated.
The paths can be found in the compilation database. The following examples show
ways to specify a file or directory for migration.

Migrate single source file:

.. code-block::

   dpct source.cpp

Migrate all files available in compilation database:

.. code-block::

   dpct -p=<path to location of compilation database file>

Migrate one file in compilation database:

.. code-block::

   dpct -p=<path to location of compilation database file> source.cpp

Migrate source files in the directory specified by the ``--in-root`` option and
place generated files in the directory specified by the ``--out-root`` option:

.. code-block::

   dpct --in-root=foo --out-root=bar


Understand Emitted Warnings
---------------------------

|tool_name| identifies the places in the code that may require your
attention during the migration of the files in order to make the code SYCL
compliant or correct.

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

The following steps show how to migrate the Vector Add sample using |tool_name|:

#. Download the `Vector Add sample <https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/Migration/vector-add-dpct>`_.

#. Navigate to the root of the Vector Add sample. The sample contains a single
   CUDA file, ``vector_add.cu``, located in the ``src`` folder.


#. From the root folder of the sample project, run |tool_name|:

   .. code-block::

      dpct --in-root=. src/vector_add.cu

   The ``--in-root`` option specifies the root location of the program sources
   that should be migrated. Only files and folders located within the
   ``--in-root`` directory will be considered for migration by the tool. Files
   located outside the ``--in-root`` directory  will not be migrated, even if
   they are included by a source file located within the ``--in-root`` directory.
   By default, the migrated files are created in a new folder named ``dpct_output``.

#. As a result of the migration command, you should see the new SYCL source file
   in the output folder:

   .. code-block::

      dpct_output
      └── src
          └── vector_add.dp.cpp

   The relative paths of the migrated files are maintained.

#. Navigate to the new SYCL source file:

   .. code-block::

      cd dpct_output/src

   Verify the migrated source code and fix any code that |tool_name|
   was unable to migrate. (The code used in this example is simple, so manual
   changes may not be needed).

   For the most accurate and detailed instructions on addressing warnings emitted
   from |tool_name|, see the **Addressing Warnings in Migrated Code** section of
   the `README files <https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/Migration>`_.

#. Compile the migrated code:

   .. code-block::

      icpx -fsycl -I<dpct_root_folder>/include src/vector_add.dp.cpp

   where <dpct_root_folder> is TODO.

#. Run the migrated program:

   .. code-block::

      ./vector_add

   You should see a block of even numbers, indicating the result of adding two vectors: ``[1..N] + [1..N]``.

.. include:: /_include_files/cross_ref_links_gsg.rst
          :start-after: refer-migrate-proj:
          :end-before: refer-migrate-proj-end:


Find More
---------

.. include:: /_include_files/find_more_gsg.rst

