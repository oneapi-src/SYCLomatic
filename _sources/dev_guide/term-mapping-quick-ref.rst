CUDA* to SYCL* Term Mapping Quick Reference
===========================================

This quick reference maps common CUDA* terms to SYCL* terms, to help start the
migration process.


Architecture Terminology Mapping
--------------------------------

+-------------------------------+------------------------------------+-------------------------------------------+
| CUDA Capable GPU              + SYCL Capable GPU from Intel                                                    |
+===============================+====================================+===========================================+
|                               | **Xe-LP and prior generations**    | **Xe-HPG and Xe-HPC**                     |
+-------------------------------+------------------------------------+-------------------------------------------+
| CUDA Core                     | Execution Unit (EU)                | Vector Engine & Matrix Engine (XVE & XMX) |
+-------------------------------+------------------------------------+-------------------------------------------+
| Streaming Multiprocessor (SM) | Subslice(SS) or Dual Subslice(DSS) | Xe-Core                                   |
+-------------------------------+------------------------------------+-------------------------------------------+
| Processor Clusters (PC)       | Slice                              | Xe-Slice                                  |
+-------------------------------+------------------------------------+-------------------------------------------+
| N/A                           | N/A                                | Xe-Stack                                  |
+-------------------------------+------------------------------------+-------------------------------------------+

Execution Model Mapping
-----------------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - CUDA
     - SYCL
   * - Thread
     - Work-item
   * - Warp
     - Sub-group
   * - Block
     - Work-group
   * - Grid
     - ND-range

Memory Model Mapping
--------------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - CUDA
     - SYCL
   * - Register
     - Private memory
   * - Shared memory
     - Local memory
   * - Constant memory
     - N/A
   * - Global memory
     - Global memory

Memory Specifier Mapping
------------------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - CUDA
     - SYCL
   * - ``__device__``
     - N/A
   * - ``__shared__``
     - N/A
   * - ``__constant__``
     - N/A
   * - ``__managed__``
     - N/A

Function Execution Space Specifiers Mapping
-------------------------------------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - CUDA
     - SYCL
   * - ``__global__``
     - Not required
   * - ``__device__``
     - Not required
   * - ``__host__``
     - Not required

Mapping of Key Host Type Used to Offload Task
---------------------------------------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - CUDA
     - SYCL
   * - device
     - ``sycl::device``
   * - stream
     - ``sycl::queue``
   * - event
     - ``sycl::event``

Kernel Execution Configures Mapping
-----------------------------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - CUDA
     - SYCL
   * - ``dim3``
     - ``sycl::range<3>``
   * - ``Kernel<<<gridDim, blockDim>>>(...)``
     - #. Member function ``parallel_for`` of class ``sycl::queue``:

          .. code-block::

               sycl::queue q;
               q.parallel_for(sycl::nd_range<3>(girdDim * blockDim, blockDim), [=](sycl::nd_item<3> item){
                 kernel(...);
               });

       #. Member function ``submit`` of class ``sycl::queue``:

          .. code-block::

               sycl::queue q;
               q.submit([&](sycl::handler &cgh){
                 cgh.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim), [=](sycl::nd_item<3> item){
                   kernel(...);
                 });
               });

Built-In Execution Space Index Mapping
--------------------------------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - CUDA
     - SYCL
   * - ``gridDim.x/y/z``
     - ``sycl::nd_item.get_group_range(2/1/0)``
   * - ``blockDim.x/y/z``
     - ``sycl::nd_item.get_local _range().get(2/1/0)``
   * - ``blockIdx.x/y/z``
     - ``sycl::nd_item.get_group(2/1/0)``
   * - ``threadIdx.x/y/z``
     - ``sycl::nd_item.get_local_id(2/1/0)``
   * - ``warpSize``
     - ``sycl::nd_item. .get_sub_group().get_local_range().get(0)``
