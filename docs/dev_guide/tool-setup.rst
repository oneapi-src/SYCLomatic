Tool Setup and Basic Use
========================

Before You Begin
----------------

Install and set up |tool_name|.

.. include:: /_include_files/before_begin_intro_dgr.rst


Basic Use
---------

.. include:: /_include_files/run_tool_cmd.rst

If no directory or file is specified for migration, the tool will try to migrate
source files found in the current directory. The default output directory is
``dpct_output``. Use the ``--out-root`` option to specify an alternate output
directory.

You can specify the file path for source files that should be migrated. If using
a compilation database, you can find your source code paths in the compilation
database file.

The `Folder Options sample <https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/Migration/folder-options-dpct>`_
shows an example of specifying a directory for migration and a specific output folder. For example:

**Linux**

.. code-block:: none

   dpct --in-root=foo --out-root=result/foo foo/main.cu foo/bar/util.cu

**Windows**

.. code-block:: none

   dpct --in-root=foo --out-root=result\foo foo\main.cu foo\bar\util.cu


For detailed instructions on how to use the Folder Options sample, refer to the sample README.


.. TODO link: To learn more about planning and running a migration, review the :ref:`migration_workflow`.


.. MIGRATION_MAKE_CMAKE

Use Make/CMake\* to Migrate a Complete Project on Linux*
--------------------------------------------------------

If your project uses Make or CMake, you can use a compilation database to provide
compilation options, settings, macro definitions, and include paths to |tool_name|.
For example:

.. code-block:: none

   dpct --compilation-database=compile_commands.json --in-root=. --out-root=migration

|tool_name| parses the compilation database and applies the necessary options
when migrating the input sources. Refer to :ref:`gen_comp_db` for detailed
information about generating a compilation database.

The `Needleman-Wunsch Sample <https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/Migration/rodinia-nw-dpct>`_
shows an example of migrating a Make/CMake project, using a compilation database to provide project details to the tool.

For detailed instructions on how to use the Needleman-Wunsch sample, refer to the sample README.

.. include:: /_include_files/ide_linux_dgr.rst

.. include:: /_include_files/ide_ms_dgr.rst


Code Samples
------------

Use the |tool_name| code samples to get familiar with the migration process and
tool features.

.. include:: /_include_files/samples.rst

.. include:: /_include_files/access_samples.rst

