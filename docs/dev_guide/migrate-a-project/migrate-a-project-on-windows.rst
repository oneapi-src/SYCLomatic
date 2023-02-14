Migrate a Project on Windows\*
==============================

Use the Command Line
--------------------

|tool_name| can be invoked at the command line.

If your project uses Microsoft Visual Studio\*, you can use the
``--vcxprojfile`` option to point to the project that requires
migration. |tool_name| will migrate the files you
provided as input files.

The following steps show how to migrate the Vector Add sample using |tool_name|:

#. Get the Vector Add sample:

   .. include:: /_include_files/open_sample_dgr.rst

#. Navigate to the root of the Vector Add sample.

   The sample contains a single
   CUDA file, ``vector_add.cu``, located in the ``src`` folder.

#. From the root folder of the sample project, run |tool_name|.

   To migrate the single file ``src\vector_add.cu``, which is part of
   ``vector-add.vcxproj``, run:

   .. code-block:: none

       dpct --vcxprojfile=vector-add.vcxproj --in-root=./ --out-root=output_proj src\vector_add.cu

   Alternately, to migrate all relevant files found by the tool in
   ``vector-add.vcxproj``, run:

   .. code-block:: none

      dpct --vcxprojfile=vector-add.vcxproj --in-root=./ --out-root=output_proj

   In both cases, the result of the migration is sent to the ``output_proj``
   folder.

.. include:: /_include_files/ide_ms_dgr.rst