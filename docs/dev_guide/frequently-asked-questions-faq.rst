
Frequently Asked Questions
==========================

**General Information**

* `How do I migrate source files that use C++11 or newer standard features on Linux\* and Windows\*?`_
* `How do I migrate files on Windows when using a CMake project?`_
* `How is the migrated code formatted?`_
* `Why does the compilation database not contain all source files in the project?`_
* `How do I use the migrated module file in the new project?`_
* `Is the memory space allocated by sycl::malloc_device, sycl::malloc_host, and dpct::dpct_malloc initialized?`_
* `How do I migrate CUDA\* source code that contains CUB library implementation source code?`_
* `How do I fix the issue of SYCL\* code hanging due to work group level synchronization, such as a group barrier used in a conditional statement?`_

**Troubleshooting Migration**

* `How do I fix an error such as "error: unknown type name" when I migrate files with "dpct --in-root=srcdir --out-root=dstdir \*.cu"?`_
* `How do I fix a parsing error such as "no member named 'max' in namespace 'std'" or "no member named 'min' in namespace 'std'" when migrating code on Windows?`_
* `How do I fix a compilation error such as "error: dlopen not declared" when I compile code on a Windows machine, that was originally migrated on Linux?`_
* `Why didn't the "atomic\*" APIs get migrated?`_
* `Why did my migration fail with "error: restrict requires a pointer or reference"?`_
* `How do I resolve incorrect runtime behavior for dpct::dev_mgr and dpct:mem_mgr in a library project that is loaded more than once in another application?`_
* `Why do I get "warning: shift count >= width of type" when I compile migrated code with the Intel® oneAPI DPC++/C++ Compiler?`_
* `How do I resolve missing include errors that occur when migrating my code?`_

General Information
-------------------

How do I migrate source files that use C++11 or newer standard features on Linux\* and Windows\*?
*************************************************************************************************

On Linux, the default C++ standard for |tool_name|'s
parser is C++98, with some C++11 features
accepted. If you want to enable other C++11 or newer standard
features in |tool_name|, you need to add
the ``--extra-arg="-std=<value>"`` option to the
command line. The supported values are:

-  ``c++11``
-  ``c++14``
-  ``c++17``

On Windows, the default C++ standard for |tool_name|'s
parser is C++14. If you want to enable C++17
features in |tool_name|, you need to add
the option ``--extra-arg="-std=c++17"`` to the command line.

How do I migrate files on Windows when using a CMake project?
*************************************************************

For a CMake project on a Windows OS, you can use CMake to generate
Microsoft Visual Studio\* project files (``vcxproj`` files). Then choose one of
the following options:

-  Migrate the source files on the command line by using the
   ``--vcxprojfile`` option of |tool_name|.

-  Migrate the entire project in Microsoft Visual Studio
   with an |tool_name| Microsoft Visual Studio plugin.

How is the migrated code formatted?
***********************************

|tool_name| provides two options to control the format of
migrated code: ``--format-range`` and ``--format-style`` .

If input source code is well formatted, |tool_name|
will use default options settings
``--format-range`` and ``--format-style`` to format the resulting
code.

If input source code is not well formatted (for example, the tool
detects mixed use of tabs and spaces or mixed indents) you can do
one of the following:

-  |tool_name| will try to detect the
   indent size of the original code and apply it to the resulting
   code. You can guide the tool by setting ``TabWidth`` and
   ``UseTab`` in the ``.clang-format`` file. Because the input source
   code is not well formatted, the indents in the resulting code
   may still be inconsistent.

-  Run |tool_name| with the
   ``--format-range=all`` option to format the entire resulting
   file. The change between input source code and resulting source
   code may be large and make it more difficult to compare the
   code.

-  Format your input source code, then use |tool_name|
   with the same ``.clang-format`` file for migration.


Why does the compilation database not contain all source files in the project?
******************************************************************************

In the project build folder, the command ``intercept-build make [target]`` is
used to generate the compilation database. The content of the compilation
database depends on the optional [target] parameter. If you need to get the
list of files corresponding to default build target, do not specify the [target]
parameter.

Make sure to disable ccache (compiler cache) in your project before using intercept-build.
If ccache is enabled, intercept-build cannot generate the complete compilation database as
some compile commands may be skipped if the target objects are already available in the cache.
Use the following command to disable ccache before running the intercept-build command:

.. code-block:: bash

   export CCACHE_DISABLE=1

How do I use the migrated module file in the new project?
*********************************************************

``.cu`` module files are compiled with the ``-ptx`` or ``-cubin`` options in the
original project and dynamically loaded into other ``*.cu`` files with
``cuModuleLoad()`` or ``cuModuleLoadData()``.

|tool_name| migrates module file code in the same way as other
``*.cu`` files. In addition, it adds a wrapper function for each function in the
module file that has the ``_global_`` attribute.

You can compile the migrated module file into a dynamic library and load the
library with a dynamic library API appropriate to your platform. For example:

- In Linux, load a dynamic library (``.so``) using ``dlopen()``
- In Windows, load a dynamic library (``.dll``) using ``LoadLibraryA()``

Is the memory space allocated by sycl::malloc_device, sycl::malloc_host, and dpct::dpct_malloc initialized?
***********************************************************************************************************

The memory allocated by ``sycl::malloc_device``, ``sycl::malloc_host``, and
``dpct::dpct_malloc`` is not initialized. If your program explicitly or
implicitly relies on the initial value of newly allocated memory, the program
may fail at runtime. Adjust your code to avoid such failures.

For example, the following original code:

.. code-block:: cpp
   :linenos:

   // original code

   int *device_mem = nullptr;device_mem = sycl::malloc_device<int>(size, dpct::get_default_queue());
   device_mem[0] += somevalue;

is adjusted to initialize the newly allocated memory to 0 before use:

.. code-block:: cpp
   :linenos:

   // fixed SYCL code

   int *device_mem = nullptr;device_mem = sycl::malloc_device<int>(size, dpct::get_default_queue());
   dpct::get_default_queue().memset(0, size).wait();
   device_mem[0] += somevalue;

How do I migrate CUDA\* source code that contains CUB library implementation source code?
*****************************************************************************************

If you migrate the CUB library implementation code directly, you may not get the
expected results. Instead, exclude CUB library implementation source code from
your migration by adding ``--in-root-exclude=<path to CUB library source code>``
to your migration command.

How do I fix the issue of SYCL\* code hanging due to work group level synchronization, such as a group barrier used in a conditional statement?
***********************************************************************************************************************************************

If synchronization API in control flow statements like a conditional statement and
loop statement are called in SYCL code, you may encounter a runtime hang issue.
The basic idea to fix the hang issue is to ensure that each synchronization API is
either reached by all work items of a workgroup, or skipped by all the work items of
a workgroup.

Here are two examples of how to fix:

In the first example, the synchronization API group barrier(nd_item.barrier()) is called
inside an if block. The evaluation results of the conditional statement differ in
each work item so not all work items can reach the synchronization API.

.. code-block:: cpp
   :linenos:

   // original code

   void kernel(const sycl::nd_item<3> &item_ct1) {
      unsigned int tid = item_ct1.get_local_id(2);
      if (tid < 32) {
         // CODE block 1
         ...
         item_ct1.barrier(sycl::access::fence_space::local_space);
         // CODE block 2
         ...
      }
   }

The following code shows how to fix the hang issue by moving the synchronization
statement out of the if block.

.. code-block:: cpp
   :linenos:

   // fixed SYCL code

   void kernel(const sycl::nd_item<3> &item_ct1) {
      unsigned int tid = item_ct1.get_local_id(2);
      if (tid < 32) {
         // CODE block 1
         ...
      }
      item_ct1.barrier(sycl::access::fence_space::local_space);
      if (tid < 32) {
         // CODE block 2
         ...
      }
   }

The second example demonstrates how to fix the hang issue when a synchronization
API is used in a for loop:

.. code-block:: cpp
   :linenos:

   // original code

   void compute(int id_space, const sycl::nd_item<3> &item_ct1) {
      unsigned int id = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
      for (; id < id_space; id += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
         ...
         item_ct1.barrier();
         ...
      }
   }

The following code shows how to fix the hang issue by making sure all work items
have same run footprint in the for loop.

.. code-block:: cpp
   :linenos:

   // fixed SYCL code

   void compute(int id_space, const sycl::nd_item<3> &item_ct1) {
      unsigned int id = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
      unsigned int num_workitem = item_ct1.get_group_range(2) * item_ct1.get_local_range(2);
      // The condition is updated to make sure all work items can enter the loop body in each iteration
      for (; id < ((id_space + num_workitem - 1) / num_workitem) * num_workitem;
      id += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
         ...
         item_ct1.barrier();
         ...
      }
   }

Troubleshooting Migration
-------------------------

How do I fix an error such as "error: unknown type name" when I migrate files with "dpct --in-root=srcdir --out-root=dstdir \*.cu"?
***********************************************************************************************************************************

The problem may be caused by files in the ``*.cu`` list, which can
be used as header files (included with an ``#include`` statement)
and are not supposed to be parsed as a standalone file. In this
case, |tool_name| reports an error if it
cannot parse the file because the file depends on the
definitions/declarations in other files. Use one of the methods
below to migrate your content:

-  Rely on |tool_name| to decide which
   files to migrate with:
   ``compile_commands.json: "dpct -p=compile_commands.json --in-root=srcdir --out-root=dstdir"``
-  Manually pass specific files to migrate, but do not pass the
   files that are included in other files and not supposed to be
   compiled as a standalone file in the original application. The
   header files are migrated automatically when they are included
   by the files provided as the input to the tool and are located
   within the ``in-root`` folder:
   ``dpct --in-root= srcdir --out-root=dstdir sample.cu``

How do I fix a parsing error such as "no member named 'max' in namespace 'std'" or "no member named 'min' in namespace 'std'" when migrating code on Windows?
***************************************************************************************************************************************************************

Use one of the following methods to resolve the error:

- Add ``#include <algorithm>`` to the source file before using ``std::min`` and
  ``std::max``
- Define the NOMINMAX macro by inserting ``#define NOMINMAX`` before including
  ``WinDef.h``


How do I fix a compilation error such as "error: dlopen not declared" when I compile code on a Windows machine, that was originally migrated on Linux?
********************************************************************************************************************************************************

When |tool_name| generates the source code, it uses dynamic loading
APIs specific to the OS on which |tool_name| is running.

For example, ``dlopen``, ``dlclose``, and ``dlsym`` are used on Linux and
``LoadLibraryA``, ``FreeLibrary``, and ``GetProcAddress`` are used on Windows.

If your code was migrated on a OS that is different from the OS you
need to compile the generated code on, migrate the project again with the
|tool_name| on the target OS or fix the code manually.


Why didn't the "atomic\*" APIs get migrated?
********************************************

|tool_name| may assume that the "atomic\*" APIs are user-defined
APIs, in which case they are not migrated.

This can occur in the following scenarios:

* The CUDA include path is specified by both ``--cuda-include-path`` and ``-I*``,
  but the paths are different
* The CUDA include path is specified by ``-I*``, but there are other CUDA include
  files located on the default CUDA install path

To make sure "atomic\*" APIs are migrated, don't use ``-I*`` to specify the CUDA
include path with the ``dpct`` migration command. Instead, use only
``--cuda-include-path`` to specify the CUDA include path.

Why did my migration fail with "error: restrict requires a pointer or reference"?
*********************************************************************************

The C++ standard does not support the restrict qualifier and the C standard
supports the restrict qualifier only on pointers to an object type.

Based on these language standards |tool_name| emits the parsing error.

You may need to adjust the source code.

How do I resolve incorrect runtime behavior for dpct::dev_mgr and dpct:mem_mgr in a library project that is loaded more than once in another application?
***********************************************************************************************************************************************************

``dpct::dev_mgr`` and ``dpct::mem_mgr`` are singleton classes in the
|tool_name| helper functions. When the helper function headers are used
to build an executable project, both ``dpct::dev_mgr`` and ``dpct::mem_mgr``
will have only one instance in the executable. However, when the helper function
headers are used to build a library project and the library project is loaded
more than once with ``dlopen()`` (or ``LoadLibraryA()`` for Windows) in an
application, more than two instances of ``dpct::dev_mgr`` and ``dpct::mem_mgr``
will be created and result in incorrect runtime behavior.

For example, both files ``libA.cpp`` and ``libB.cpp`` include |tool_name|
helper function header ``dpct.hpp``, and they are built into dynamic libraries
``libA.so`` and ``libB.so`` respectively. If an application ``main.cpp`` imports
the libraries with ``dlopen()``, there will be two instances of ``dpct::dev_mgr``
and ``dpct::mem_mgr`` in the runtime of the application.

To resolve this issue, separate the implementation and the declaration of
``dpct::dev_mgr`` and ``dpct::mem_mgr`` in |tool_name| helper function:

#. Create a new C++ file ``dpct_helper.cpp``.
#. Move the implementation of ``instance()`` in ``class dev_mgr`` from
   ``dpct/device.hpp`` to ``dpct_helper.cpp``.

   For example, the original ``dpct/device.hpp``:

   .. code-block:: cpp
      :linenos:

       class dev_mgr {
       public:
         static dev_mgr &instance() { // the implementation and the declaration of dev_mgr::instance
           static dev_mgr d_m;
           return d_m;
         }
         ...
       }

   is updated to:

   .. code-block:: cpp
        :linenos:

         class dev_mgr {
         public:
           static dev_mgr &instance();//the declaration of dev_mgr::instance
           ...
         }

   and the new ``dpct_helper.cpp`` now contains the implementation of
   ``dev_mgr::instance()``:

   .. code-block:: cpp
        :linenos:

        #include <dpct/device.hpp>
        dpct::dev_mgr &dev_mgr::instance(){ // the implementation of dev_mgr::instance
          static dev_mgr d_m;
          return d_m;
        }

#. Similar to step two, move the implementation of ``instance()`` in the
   ``class mem_mgr`` from ``dpct/memory.hpp`` to ``dpct_helper.cpp``.
#. Build ``dpct_helper.cpp`` into a dynamic library ``libdpct_helper``.

   * In Linux:

     .. code-block:: bash

         dpcpp -g -shared -o libdpct_helper.so -fPIC ./dpct_helper.cpp

   * In Windows:

     .. code-block:: bash

         cl.exe /LD dpct_helper.cpp

#. Add library ``libdpct_helper`` to the environment variables.

   * In Linux: Add the location of ``libdpct_helper.so`` into ``LD_LIBRARY_PATH``.
   * In Windows: Add the location of ``libdpct_helper.dll`` into ``PATH``.
#. Dynamically link ``libdpct_helper`` when building libraries and applications.

After performing the update steps, all the libraries and applications will share
the same instance of the device manager ``dpct::dev_mgr`` and the memory manager
``dpct::mem_mgr`` in |tool_name| helper functions.

Why do I get "warning: shift count >= width of type" when I compile migrated code with the Intel® oneAPI DPC++/C++ Compiler?
****************************************************************************************************************************

Shifting bits where the shift is greater than the type length is undefined
behavior for the |dpcpp_compiler|_ and may result in different behavior on
different devices. Adjust your code to avoid this type of shift.

For example, the migrated SYCL\* code:

.. code-block:: cpp
   :linenos:

   // migrated SYCL code

   void foo() {
     ...
     unsigned int bit = index[tid] % 32;
     unsigned int val = in[tid] << 32 - bit;
     ...
   }

is adjusted to avoid a bit shift that is greater than the type length:

.. code-block:: cpp
   :linenos:

   // fixed SYCL code

   void foo() {
     ...
     unsigned int bit = index[tid] % 32;
     unsigned int val;
     if(32 - bit == 32)
       val = 0;
     else
       val = in[tid] << 32 - bit;
     ...
   }

How do I resolve missing include errors that occur when migrating my code?
**************************************************************************

Use the option ``--extra-arg=-v`` to prompt |tool_name| to use verbose
output, which includes information about which paths the tool searches
for includes.

You can provide an additional path to look for includes in one of the following
ways:

* Use the ``--extra-arg="-I<extra include path>"`` option in your migration command
  to specify an additional path for the tool to use when searching for includes
  during migration.

* If you are using a compilation database, add the ``-I<extra include path>``
  option to the compile command in the database for the source files, to 
  specify the include path.
