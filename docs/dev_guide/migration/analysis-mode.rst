Analysis Mode Guide
=======================

Analysis Mode generates a summary report that shows how many lines of code are migrated with confidence and how many lines of code are not migrated perfectly due to the language gap between CUDA and SYCL, also it provides estimation of human efforts required to complete the code migration based on the effort levels. Three levels are defined for effort: low, medium, and high. All migration issues are classified into them. Effort required for each level: Typically, low effort level means developer can solve the issues in minutes, medium effort level issues take hours and if high effort level issue is reported, unfortunately, it means there is no equivalent functionality provided yet in SYCL or SYCL ecosystem libraries for the CUDA API or the CUDA library API, developer need manual rewrite the functionality.

Following is an example of analysis report:

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
