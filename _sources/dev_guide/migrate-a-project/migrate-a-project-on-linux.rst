Migrate a Project on Linux\*
============================

Use the Command Line
--------------------

You can invoke |tool_name| at the command line.

Use the tool's ``--in-root`` option to specify the location of the
source that should be migrated:

* Any source within the ``--in-root`` directory (at any nesting
  level) may be migrated.
* Any header file within the ``--in-root`` directory (at any nesting level) that
  is included by the source or header file, which is being migrated, is also
  migrated.
* Files from outside the ``--in-root`` directory will not be migrated even if
  they are included by any of your program source files.
* If the ``--in-root`` option is not specified, the directory of the first input
  source file is implied.

Use the tool's ``--out-root`` option to specify the directory where the
SYCL\* code produced by |tool_name| is written:

* Relative paths of the migrated files are maintained.
* Extensions are changed to ``.dp.cpp``.
* If the ``--out-root`` option is not specified, ``./dpct_output`` is implied.

The following steps show how to migrate the Folder Options sample using |tool_name|:

#. Get the Folder Options sample:

   .. include:: /_include_files/open_sample_dgr.rst

#. Navigate to the root of the sample project.

   The Folder Options sample project contains a simple CUDA\* program with three
   files (``main.cu``, ``util.cu``, and ``util.h``) located in two folders
   (``foo`` and ``bar``):

   .. code-block:: none

      foo
      ├── bar
      │   ├── util.cu
      │   └── util.h
      └── main.cu

#. From the root folder of the sample project, run |tool_name|:

   .. code-block:: none

      dpct --in-root=foo --out-root=result/foo foo/main.cu foo/bar/util.cu --extra-arg="-Ifoo/bar/"

   The ``--in-root`` option specifies the location of the CUDA
   files that need migration. The ``--out-root`` option specifies the location for the migrated files.
#. As a result of the migration command, you should see the following files:

   .. code-block:: none

      result/foo
           ├── bar
           │   ├── util.dp.cpp
           │   └── util.h
           └── main.dp.cpp

#. Inspect the migrated source code, address any generated DPCT warnings, and
   verify correctness of the new program.


Review :ref:`emitted-warnings` for additional information about inserted warnings
and comments.

For the most accurate and detailed instructions of addressing warnings, see the
**Addressing Warnings in the Migrated Code** section of the sample README files.

For more information on command line capabilities, review the
:ref:`Command Line Options Reference <cmd_opt_ref>`.

.. MIGRATION_MAKE_CMAKE

Use Make/CMake\* to Migrate a Complete Project
----------------------------------------------

If your project uses Make or CMake, you can utilize compilation database support
to provide compilation options, settings, macro definitions, and include paths to
|tool_name|. Refer to :ref:`gen_comp_db` for detailed information about generating a compilation database.

|tool_name| parses the compilation database and applies the necessary
options when migrating the input sources.

This example uses the Rodinia needleman-wunsch sample to demonstrate the use of a
compilation database.

**Step 1: Create the Compilation Database**

#. Get the Rodinia needleman-wunsch sample:

   .. include:: /_include_files/open_sample_dgr.rst

#. When using CMake: Before running ``intercept-build``, configure and generate
   your Makefile out of ``CMakeLists.txt``. An example of a typical command is
   ``cmake ...``.

#. Invoke the build command, prepending it with ``intercept-build``.

   .. code-block:: none

      $ make clean
      $ intercept-build make

   This creates the file ``compile_commands.json`` in the working directory.

   The ``intercept-build`` script runs your project's build command without building
   the original program. It records all the compiler invocations and stores the
   names of the input files and the compiler options in the compilation database file
   ``compile_commands.json``.

   .. note::

      This example assumes the CUDA headers are available at ``/usr/local/cuda/include``.
      Replace this path according to where they reside on your system.

#. Once ``intercept build`` is run, review the output in the
   ``compile_commands.json`` file. The content of this file should look like
   this example:

   .. code-block:: none
      :linenos:

      [{
        "command" : "nvcc -c -o needle -I/usr/local/cuda/include -D__CUDA_ARCH__=400 "
                    "-D__CUDACC__=1 needle.cu",
        "directory" : "/home/user/projects/DPCPP_CT/rodinia_3.1/cuda/nw",
        "file" : "/home/user/projects/DPCPP_CT/rodinia_3.1/cuda/nw/needle.cu"
      }]

**Step 2: Use the Compilation Database with the Migration Tool**

By default, |tool_name| looks for the ``compile_commands.json`` file
in the current directory and uses the compiler options from it for each input file.

Use the following command to migrate the CUDA code in the Rodinia needleman-wunsch
sample, using the compilation database generated in the previous step:

.. code-block:: none

   dpct -p=compile_commands.json --in-root=. --out-root=migration

The ``--in-root`` option sets the root location of the CUDA files that need
migration. Only files and folders located within the ``--in-root`` directory will
be considered for migration by the tool.

The ``--out-root`` option specifies the location for the migrated files.
The new project will be created in the migration directory.

The ``-p`` option specifies the path for the compilation database.

After running the migration command, you should see the following files in the
``migration`` output folder:

.. code-block:: none

   migration
   └── src
       ├── needle.h
       ├── needle_kernel.dp.cpp
       └── needle.dp.cpp

**Step 3: Verify the Source for Correctness and Fix Anything the Tool was Unable
to Migrate**

Verify the migration of the source code that uses variables declared
using preprocessor directives. Inspect the migrated source code, address any
generated DPCT warnings, and verify correctness of the new program.

Review :ref:`emitted-warnings` for additional information about inserted warnings
and comments.

For the most accurate and detailed instructions on addressing warnings, see the
**Addressing Warnings in the Migrated Code** section of the samples README files.

.. include:: /_include_files/ide_linux_dgr.rst

