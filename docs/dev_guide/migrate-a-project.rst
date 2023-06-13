.. _mig_proj:

Migrate a Project
=================

.. toctree::
   :hidden:

   migrate-a-project/before-you-begin
   migrate-a-project/migrate-a-project-on-linux
   migrate-a-project/migrate-a-project-on-windows
   migrate-a-project/incremental-migration
   migrate-a-project/user-defined-migration-rules
   migrate-a-project/generate-compilation-db


|tool_name| ports CUDA\* language kernels and library API calls to
SYCL\* for the |dpcpp_compiler|_. Typically, 90%-95% of CUDA code automatically
migrates to SYCL. The tool inserts inline comments during migration to
help you complete the remaining code migration.

**CUDA‡ to SYCL‡ Code Migration & Development Workflow**

.. figure:: /_images/cuda-sycl-migration-workflow.png

|tool_name| migration workflow overview:

#. **Prepare CUDA source for migration**

   Start with a running CUDA project that can be built and run. |tool_name|
   looks for CUDA headers, so make sure the headers are accessible to the tool.

#. **Migrate your project**

   To generate annotated SYCL code, run |tool_name| with the
   original source as input to the tool.

   For simple projects, you can use file-to-file migration, with the option to
   migrate all files at once or to migrate files one-by-one.

   For complex projects, you can utilize the Microsoft Visual Studio\* project
   file or Make/Cmake file to build a :ref:`compilation database <gen_comp_db>`,
   used to migrate the complete project.

#. **Review converted code**

   Output files contain annotations to help migrate any remaining code that
   could not be automatically migrated. Inspect the converted code, review the
   annotations to help manually convert unmigrated code, and look for potential
   code improvements.

#. **Build the project with the Intel® oneAPI DPC++/C++ Compiler**

   Make sure your newly migrated project compiles successfully with the
   |dpcpp_compiler|_.

