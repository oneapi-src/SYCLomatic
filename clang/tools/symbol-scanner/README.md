# BinarySymbolScanner.py
The Binary Symbol Scanner is a Python script based binary scanning utility to
capture performance library and middleware function names from application
binaries. It covers binaries that are linked with Intel and other vendor
implementations of performance and domain specific libraries. The function
names are extracted from application binary symbol information and saved to
CSV files. 

In addition to providing information on the performance library functions used
by the application, the CSV output file can be used to share customer library
optimization priorities with Intel.

## Overview
Software applications across the industry rely on performance libraries from
Intel, NVIDIA, AMD, ARM etc to build optimal solutions for end users. These 
applications benefit from thousands of functions that the libraries in question
contain. As the scope of these libraries grow over time, it is important to
periodically assess which functions in such libraries must be maintained at
optimal performances.

While it is easy to determine if an application uses a library, the exact
library functions that are being used are still elusive. The easiest way to
capture this information is through instrumentation of the library itself, so
when such data is desired, it can easily be accessed. However, this defeats the
purpose of such performance libraries as tolerance to any overhead is limited.
To simplify the capture of the actually used set of function calls, this script
has been implemented to scan binaries for the function signatures of interest.

## Intel performance Libraries and middleware 
  - Intel&reg; Integrated Performance Primitives Library (IPP)
  - Intel&reg; Math Kernel Library (Legacy MKL)
  - oneAPI Math Kernel Library (oneMKL)
  - oneAPI DPC++ Library (oneDPL) 
  - oneAPI Data Analytics Library (oneDAL)
  - oneAPI Deep Neural Network LIbrary (oneDNN) 
  - oneAPI Collective Communications Library (oneCCL)
  - oneAPI Threading Building Blocks (oneTBB)
  - Ray Tracing components (Embree, Open Volume Kernel Library, Open
    Image Denoise)
  - OSPRay 
  - oneAPI Level Zero
  - SYCL 

In addition to the Intel libraries listed above, the script can optionally also
scan equivalent non-Intel performance libraries. This feature is enabled using a
command line argument **(-x)**. 

If the output data from the script is shared with Intel, it will be used for 
product improvements, namely performance prioritization, missing functionality
and gap analysis of Intel libraries and the coverage support of 
[SYCLomatic](https://github.com/oneapi-src/SYCLomatic).

>**NOTE:** No information is shared with Intel as a part of the data collection
process itself. The script will not collect any personally identifiable
information, nor any specific workload information. A user can inspect the CSV
output files before sharing.


## Requirements
 - Windows 
   - The script must be run from a Visual Studio command window as it relies on
     the Visual Studio toolchain.
   - Python3 must be installed
 - Linux
   - Python3 must be installed
   - 'nm' command should be in the path 

## Usage

> %> BinarySymbolScanner.py [-h] -c "company-name" -p "product-name" [-b "path"] [-d] [-s] [-x]

The captured functions from the binaries in the search path will be saved in
'product-name.funcs.csv'. If statistics flag is turned on, function occurrence
histogram will be saved to 'product-name.stats.csv'.


 | Options/flags                                | Description                                                                                                                                                    |
 | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
 | -h, --help                                   | show this help message and exit                                                                                                                                |
 | -c COMPANY_NAME, --company-name COMPANY_NAME | Company name the product belongs to.                                                                                                                           |
 | -p PROD_NAME, --product-name PROD_NAME       | Product name for which the binaries are being scanned.                                                                                                         |
 | -b BIN_DIR, --binary-dir BIN_DIR             | Application binary directory to be scanned. If the  option is not provided, the script will scan the current directory and all child directories for binaries. |
 | -d, --debug                                  | Enable debugging help.                                                                                                                                         |
 | -s, --statistics                             | Create function histograms.                                                                                                                                    |
 | -x, --extend-non-intel                       | Extend the scanning to non-Intel performance libraries.                                                                                                        |
 | -m, --migration-status                       | Scans only for NVIDIA library APIs and emits the % of APIs which can be migrated to SYCL equiavlent
             
## Sample Output of Migration Status

$ python3 BinarySymbolScanner.py -c "Company" -p "Appication_Name" -b /home/anoop/application_binary_dir/ -m

.
.

Overwriting file:  Application_Name.funcs.csv

List of Migratable APIs: ['cudaDeviceSynchronize', 'cudaEventSynchronize', 'cudaGetSymbolSize', 'cudaProfilerInitialize', 'cudaStreamSynchronize', 'cudaThreadSynchronize']
 
List of Non-Migratable APIs: []

**********************************************************************************************
Percentage of CUDA Host/Library APIs Migratable by SYCLomatic: 100.00%
**********************************************************************************************
