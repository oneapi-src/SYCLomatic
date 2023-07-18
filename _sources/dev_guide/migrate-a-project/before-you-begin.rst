Before You Begin
================

.. include:: /_include_files/before_begin_intro_dgr.rst

Samples
-------

Several sample projects for |tool_name| are available to explore the tool and
familiarize yourself with how it functions.

.. include:: /_include_files/access_samples.rst

.. include:: /_include_files/samples.rst

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
