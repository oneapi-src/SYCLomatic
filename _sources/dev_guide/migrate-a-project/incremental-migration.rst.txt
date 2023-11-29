Incremental Migration
=====================

|tool_name| provides incremental migration, which automatically
merges the results from multiple migrations into a single migrated project.

Incremental migration can be used to

* migrate a CUDA\* project incrementally, for example 10 files at a time
* migrate new CUDA files into an already migrated project
* migrate multiple code paths

Incremental migration is enabled by default. Disable incremental migration using
the ``--no-incremental-migration`` option.

Example 1: Migrate a File with Conditional Compilation Code
-----------------------------------------------------------

This example shows incremental migration for a file ``sample1.cu`` that
contains conditional compilation code. Content of ``sample1.cu``:

.. code-block:: none
   :linenos:

     #ifndef MACRO_A
     ... code path 1 ...
     #elif MACRO_A == 1
     ... code path 2 ...
     #else
     ... code path 3 ...
     #endif

Use the following steps to incrementally migrate ``sample1.cu``.

#. Generate ``sample1.dp.cpp``, which contains migrated code for code path 1.
   From the same working directory as the file, run:

   .. code-block:: none

	  dpct sample1.cu --out-root=out

#. Generate ``sample1.dp.cpp``, which contains migrated code for code path 1 and
   code path 2:

   .. code-block:: none

      dpct sample1.cu --out-root=out --extra-arg=”-DMACRO_A=1”

#. Generate ``sample1.dp.cpp``, which contains migrated code for code path 1,
   code path 2, and code path 3:

   .. code-block:: none

      dpct sample1.cu --out-root=out --extra-arg=”-DMACRO_A=2”

The result contains migrated code for each code path.

Example 2: Migrate a Header File Used by Multiple Source Files
--------------------------------------------------------------

This example shows the use of incremental migration for a header file that is
included in several source files, each with a different macro definition.

Content of header file ``sample_inc.h``:

.. code-block:: none
   :linenos:

     #ifdef MACRO_A
     ... code path 1...
     #else
     ... code path 2...
     #endif


Content of source file ``sample2.cu``:

.. code-block:: none
   :linenos:

     #define MACRO_A
     #include "sample_inc.h"
     #undef MACRO_A

Content of source file ``sample3.cu``:

.. code-block:: none

	#include "sample_inc.h"

Use the following steps to incrementally migrate the files.

#. Generate ``sample2.dp.cpp`` and ``sample_inc.h``, which contains migrated
   code for code path 1.

   From the same working directory as the file, run:

   .. code-block:: none

      dpct sample2.cu --out-root=out

#. Generate ``sample3.dp.cpp`` and ``sample_inc.h``, which contains migrated
   code for code path 1 and code path 2:

   .. code-block:: none

      dpct sample3.cu --out-root=out

The result contains migrated code for each code path.


Limitations
-----------

Incremental migration will not be triggered in the following conditions:

#. |tool_name| option ``--no-incremental-migration`` is specified.
#. Different versions of |tool_name| are used across multiple
   migration invocations.
#. Different options of |tool_name| are used across multiple migration
   invocations. If |tool_name| detects that a previous migration
   used a different option-set, |tool_name| will stop migration and
   exit.

The following options direct |tool_name| to generate different
migrated code and may break incremental migration. Use the same values for these
options across migration invocations to keep incremental migration working.

* ``--always-use-async-handler``
* ``--assume-nd-range-dim``
* ``--comments``
* ``--custom-helper-name``
* ``--enable-ctad``
* ``--keep-original-code``
* ``--no-cl-namespace-inline``
* ``--no-dpcpp-extensions``
* ``--no-dry-pattern``
* ``--optimize-migration``
* ``-p``
* ``--process-all``
* ``--sycl-named-lambda``
* ``--use-experimental-features``
* ``--use-explicit-namespace``
* ``--usm-level``
* ``--vcxprojfile``
