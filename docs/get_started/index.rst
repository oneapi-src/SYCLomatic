.. _get_started:

Get Started with |tool_name|
============================

|tool_name| assists in the migration of a developer's program that
is written in CUDA\* to a program written in |dpcpp_long|, which is based on
modern C++ and incorporates portable industry standards such as SYCL\*.

.. include:: /_include_files/intro_links_gsg.rst

.. note::

   Use of |tool_name| will result in a project that is not entirely
   migrated. Additional work, as outlined by the output of |tool_name|,
   is required to complete the migration.


Before You Begin
----------------

.. include:: /_include_files/before_begin_intro_gsg.rst

Certain CUDA header files (specific to your project) may need to be accessible
to |tool_name|. |tool_name| looks for these CUDA
header files in the default locations:

-  ``/usr/local/cuda/include``

-  ``/usr/local/cuda-x.y/include``, where *x.y* is one of these values:
   |cuda_versions|.

You can reference custom locations by pointing to them with the
``--cuda-include-path=<path/to/cuda/include>`` option in |tool_name|
command line.

.. note::

   The CUDA include path should not be the same as, or a child path of, the
   directory where the source code that needs to be migrated is located.

Currently, |tool_name| supports the migration of programs
implemented with CUDA versions |cuda_versions|. The list of supported languages
and versions may be extended in the future.

.. include:: /_include_files/env_setup_gsg.rst

The general invocation syntax from the operating system shell is:

.. code-block::

   dpct [options] [<source0>... <sourceN>]

.. note::

   ``c2s`` is an alias to the ``dpct`` command and may be used in it's place.

Built-in Usage Information
~~~~~~~~~~~~~~~~~~~~~~~~~~

To see the list of |tool_name|–specific options, use ``--help``:

.. code-block::

   dpct --help

To see the list of the language parser (Clang\*) options, pass ``-help``
as the Clang option:

.. code-block::

   dpct -- -help


Emitted Warnings
----------------

|tool_name| identifies the places in the code that may require your
attention during the migration of the files in order to make the code SYCL
compliant or correct.

Comments are inserted into the generated source files and displayed as warnings
in the output. For example:

.. code-block::

   /path/to/file.hpp:26:1: warning: DPCT1003:0: Migrated API does not return error code. (*,0) is inserted. You may need to rewrite this code.
   // source code line for which warning was generated
   ^

.. include:: /_include_files/cross_ref_links_gsg.rst
          :start-after: refer-diag-ref:
          :end-before: refer-diag-ref-end:


Migrate a Simple Test Project
-----------------------------

Several sample projects for |tool_name| are available to explore the tool and
familiarize yourself with how it functions.

.. include:: /_include_files/access_samples.rst

.. include:: /_include_files/samples.rst






Try a Sample Project
~~~~~~~~~~~~~~~~~~~~

The following steps show how to migrate the Vector Add sample using |tool_name|:

#. Download the `Vector Add sample <https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/Migration/vector-add-dpct>`_.

#. Navigate to the root of the Vector Add sample. The sample contains a single
   CUDA file, ``vector_add.cu``, located in the ``src`` folder.


#. From the root folder of the sample project, run |tool_name|:

   .. code-block::

      dpct --in-root=. src/vector_add.cu

   The ``--in-root`` option specifies the location of the CUDA file that needs
   migration. By default, the migrated files are created in a new folder named
   ``dpct_output``.

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

   .. note::

      To compile the migrated sample, add ``-I<dpct_root_folder>/include`` to
      your compile command.

.. include:: /_include_files/cross_ref_links_gsg.rst
          :start-after: refer-migrate-proj:
          :end-before: refer-migrate-proj-end:


Find More
---------

.. include:: /_include_files/find_more_gsg.rst

