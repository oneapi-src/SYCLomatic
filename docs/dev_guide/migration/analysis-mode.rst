.. _analysis_mode:

Analysis Mode
=============

Analysis Mode generates a summary report that shows:

* Lines of code that are migrated with confidence
* Lines of code that are migrated imperfectly due to the language gap between
  CUDA\* and SYCL\*
* An estimation of how much human effort will be required to complete the code migration

All migration issues are classified into one of three levels of manual effort:

* **Low Effort**: Indicates an issue that the developer can solve in minutes.
* **Medium Effort**: Indicates an issue that may take hours to solve.
* **High Effort**: Indicates that there is not yet equivalent functionality provided
  in SYCL or SYCL ecosystem libraries for the CUDA API or the CUDA library API.
  In these cases, the developer will need to manually rewrite the functionality.

An example analysis report can be seen as follows:

.. code-block:: none
    
    reduction.cpp:
      + 26 lines of code (100%) will be automatically migrated.
        - 22 APIs/Types - No manual effort.
        -  4 APIs/Types - Low manual effort for checking and code fixing.
        -  0 APIs/Types - Medium manual effort for code fixing.
      +  0 lines of code (  0%) will not be automatically migrated.
        -  0 APIs/Types - High manual effort for code fixing.
    reduction_kernel.cu:
      + 77 lines of code ( 89%) will be automatically migrated.
        -  7 APIs/Types - No manual effort.
        - 63 APIs/Types - Low manual effort for checking and code fixing.
        -  7 APIs/Types - Medium manual effort for code fixing.
      +  9 lines of code ( 10%) will not be automatically migrated.
        -  9 APIs/Types - High manual effort for code fixing.
    Total Project:
      +103 lines of code ( 91%) will be automatically migrated.
        - 29 APIs/Types - No manual effort.
        - 67 APIs/Types - Low manual effort for checking and code fixing.
        -  7 APIs/Types - Medium manual effort for code fixing.
      +  9 lines of code (  8%) will not be automatically migrated.
        -  9 APIs/Types - High manual effort for code fixing.


|tool_name| provides a command line option to enable this feature:

.. code-block:: bash

    $ dpct --analysis-mode sample.cu

|tool_name| supports saving the report to a file:

.. code-block:: bash

    $ dpct --analysis-mode --analysis-report-output-file=report.txt sample.cu
