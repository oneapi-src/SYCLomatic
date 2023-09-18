.. _mig_proj:

Migrate a Project
=================

.. toctree::
   :hidden:

   migrate-a-project/before-you-begin
   migrate-a-project/migrate-a-project-on-linux
   migrate-a-project/migrate-a-project-on-windows
   migrate-a-project/generate-compilation-db
   migrate-a-project/incremental-migration
   migrate-a-project/user-defined-migration-rules
   migrate-a-project/post-migration-pattern-rewrite



|tool_name| ports CUDA\* language kernels and library API calls to
SYCL\* for the |dpcpp_compiler|_. Typically, 90%-95% of CUDA code automatically
migrates to SYCL. The tool inserts inline comments during migration to
help you complete the remaining code migration.

**CUDA‡ to SYCL‡ Code Migration & Development Workflow**

.. figure:: /_images/cuda-sycl-migration-workflow.png

The |tool_name| migration workflow follows four high-level steps:

#. **Prepare your CUDA source for migration**

   Start with a running CUDA project that can be built and run. |tool_name|
   looks for CUDA headers, so make sure the headers are accessible to the tool.

#. **Migrate your project**

   Run |tool_name| with the original source as input to the tool to generate
   annotated SYCL code.

   You can use file-to-file migration for simple projects or a
   :ref:`compilation database <gen_comp_db>` for more complex projects.

#. **Review the converted code**

   Output files contain annotations to mark code that could not be automatically
   migrated. Review the annotations and manually convert any unmigrated code.
   Also look for potential code improvements.

#. **Build your project**

   To make sure your newly migrated project compiles successfully, build the
   project with the |dpcpp_compiler|_.

