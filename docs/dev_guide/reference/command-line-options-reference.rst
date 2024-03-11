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

Auto complete command line options for |tool_name|
--------------------------------------------------

This tool supports auto-completion of command line options. You can auto complete the option name and option value (for enumeration options only)
by using <tab>.

This feature can be enabled via the following command:

.. code-block::

   source /path/to/SYCLomatic/install/folder/setvars.sh

.. note:: 
  
   Note that the auto-complete feature will not work if <tab> is used after <space>.

.. note::

   This feature is only supported for Bash on Linux platform.

Example 1:

.. code-block::

   dpct --<tab><tab>

This usage will show all dpct options.

Example 2:

.. code-block::

   dpct --ver<tab>

This usage will auto complete the command line to:

.. code-block::

   dpct --version

Example 3:

.. code-block::

   dpct --use-experimental-features=<tab><tab>

This usage will show all available values of this option.

Example 4:

.. code-block::

   dpct --use-experimental-features=logical-group,mat<tab>

This usage will auto complete the command line to:

.. code-block::

   dpct --use-experimental-features=logical-group,matrix

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





