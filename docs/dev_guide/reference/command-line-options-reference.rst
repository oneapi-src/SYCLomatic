.. _cmd_opt_ref:

Command Line Options Reference
==============================

The tool supports many command line options you can use in your migration. This
section provides:

* An :ref:`alphabetical list of command line options <alpha_opt>`
* Functional groups of options:
  
  * :ref:`mig_basic_opt`
  * :ref:`adv_mig_opt`
  * :ref:`code_gen_opt`
  * :ref:`report_gen_opt`
  * :ref:`build_script_opt`
  * :ref:`help_opt`
* A list of :ref:`intercept-build options <intercept_opt>`

Auto-Complete Command Line Options
----------------------------------

The tool supports auto-completion of command line options. You can auto-complete
the option name and option value (for enumeration options only) by using <tab>.

Enable this feature with the following command:

.. include:: /_include_files/enable_opt_auto_comp_dgr.rst

.. note::

   Auto-completion of command line options is only supported for Bash on Linux*.

   The auto-complete feature will not work if <tab> is used after <space>.


The following examples show how to use the auto-complete feature:

* Show all tool options:

  .. code-block::

     dpct --<tab><tab>

* Auto-complete the command to ``dpct --version``

  .. code-block::

     dpct --ver<tab>

* Show all available values of the ``--use-experimental-features`` option:

  .. code-block::

     dpct --use-experimental-features=<tab><tab>

* Auto-complete the command to ``dpct --use-experimental-features=logical-group,matrix``:

  .. code-block::

     dpct --use-experimental-features=logical-group,mat<tab>


.. toctree::
   :hidden:

   command-line-options-ref/migration-basic
   command-line-options-ref/migration-advance
   command-line-options-ref/code-gen
   command-line-options-ref/report-gen
   command-line-options-ref/build-script-opt
   command-line-options-ref/help-info
   command-line-options-ref/intercept-build-options
   command-line-options-ref/deprecated-options
   command-line-options-ref/alpha-list





