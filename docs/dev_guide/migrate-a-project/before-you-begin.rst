Before You Begin
================

.. include:: /_include_files/before_begin_intro_dgr.rst

Samples
-------

|tool_name| comes with several sample projects so you can explore
the tool and familiarize yourself with how it functions.

.. include:: /_include_files/wip.rst

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Sample Project
     - Description
   * - Vector Add DPCT

       - ``vector_add.cu``
     - The Vector Add DPCT sample demonstrates how to migrate a simple program
       from CUDA\* to |dpcpp_long|. Vector Add provides an easy way to verify
       that your development environment is setup correctly to use |tool_name|.
   * - Folder Options DPCT

       - ``main.cu``
       - ``bar/util.cu``
       - ``bar/util.h``
     - The Folder Options DPCT sample provides an example of how to migrate more
       complex projects and use ``dpct`` options.
   * - Rodinia NW DPCT

       - ``needle.cu``
       - ``needle.h``
       - ``needle_kernel.cu``
     - The Rodinia NW DPCT project demonstrates how to migrate a Make/CMake\*
       project from CUDA to |dpcpp_long| using |tool_name|.

Review the README file provided with each sample for more detailed information
on the purpose and usage of the sample project.

.. include:: /_include_files/access_samples.rst


.. _emitted-warnings:

Emitted Warnings
----------------

During the migration of the files |tool_name| identifies places
in the code that may require your attention to make the code SYCL compliant
or correct.

|tool_name| inserts comments into the generated source files which
are displayed as warnings in the output. For example:

.. code-block:: none

  /path/to/file.hpp:26:1: warning: DPCT1003:0: Migrated API does not return error code. (*,0) is inserted. You may need to rewrite this code.
  // source code line for which warning was generated
  ^

For more details on what a particular warning means, see the
:ref:`Diagnostics Reference <diag_ref>`.
