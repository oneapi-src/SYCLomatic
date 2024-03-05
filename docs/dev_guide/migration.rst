.. _mig_proj:

Migration
=========

This section contains information about migration workflow, planning a migration,
and migration tool features.

Migration Workflow
------------------

The migration workflow guidelines describe the CUDA* to SYCL* code migration workflow, with
detailed information about how to prepare your code for migration, how to use
tool features to configure your migration, and how to build and optimize your
migrated code.

Review :ref:`mig_workflow`.

Features to Support Migration
-----------------------------

|tool_name| provides multiple features to aid migration preparation and customization.

* :ref:`gen_comp_db`
* :ref:`inc_mig`
* :ref:`migration_rules`
* :ref:`debug_codepin`


Query CUDA to SYCL API Mapping
------------------------------

You can query which SYCL\* API is functionally compatible with a specified CUDA\* API.

Learn how to :ref:`query_map`.

.. toctree::
   :hidden:

   migration/migration-workflow
   migration/generate-compilation-db
   migration/incremental-migration
   migration/migration-rules
   migration/debug-with-codepin
   migration/API-Mapping-query-guide

