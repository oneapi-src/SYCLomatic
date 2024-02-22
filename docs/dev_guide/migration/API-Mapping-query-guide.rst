Query CUDA* to SYCL* API Mapping
================================

Use the ``--query-api-mapping`` option to discover which SYCL\* API is functionally
compatible with a specified CUDA\* API. The API query functionality can assist users
on manual code migration and help users understand the SYCL API.

The following example queries the functionally compatible SYCL API for the
``cudaMalloc`` function:

.. code-block:: bash

   $ dpct --query-api-mapping=cudaMalloc

     CUDA API:
       cudaMalloc(pDev /*void ***/, s /*size_t*/);
     Is migrated to:
       *pDev = (void *)sycl::malloc_device(s, dpct::get_in_order_queue());

On Linux, you can use the ``tab`` key to auto-complete the CUDA API name
specified for the ``--query-api-mapping`` option. For example :

1. Specify the partial CUDA function name "cudaMa" to the ``--query-api-mapping`` option:

   .. code-block:: bash

      dpct --query-api-mapping=cudaMa
2. Press ``tab``.

   The CUDA API name will auto-complete to ``cudaMalloc``:

   .. code-block:: bash

      dpct --query-api-mapping=cudaMalloc

|tool_name| also supports fuzzy matching of CUDA API names. For example:

.. code-block:: bash

   $ dpct --query-api-mapping=cudamalloc

     CUDA API:
       cudaMalloc(pDev /*void ***/, s /*size_t*/);
     Is migrated to:
       *pDev = (void *)sycl::malloc_device(s, dpct::get_in_order_queue());
