API-Mapping Query Guide
=======================

|tool_name| provides the functionality for users to query functionally
compatible SYCL API for some CUDA API. The API query functionality can assist
user on manual code migration and help user understand the SYCL API.

API mapping query example:

   .. code-block:: bash

      $ dpct --query-api-mapping=cudaMalloc
      CUDA API:
        cudaMalloc(pDev /*void ***/, s /*size_t*/);
      Is migrated to:
        *pDev = (void *)sycl::malloc_device(s, dpct::get_in_order_queue());

- You can use ``tab`` button to auto-complete the CUDA API names (In Linux OS only).

    .. code-block:: bash

        $ dpct --query-api-mapping=cudaMa (press tab)
        $ dpct --query-api-mapping=cudaMalloc

- |tool_name| supports fuzzy match of CUDA API names.
  For example:

    .. code-block:: bash

        $ dpct --query-api-mapping=cudamalloc
        CUDA API:
          cudaMalloc(pDev /*void ***/, s /*size_t*/);
        Is migrated to:
          *pDev = (void *)sycl::malloc_device(s, dpct::get_in_order_queue());
