API-Mapping Query Guide
=======================

|tool_name| provides the functionality for users to query functionally
compatible SYCL API for some CUDA API. The API query functionality can assist
users on manual code migration, as well as help users understand the SYCL API.

API mapping query example:

   .. code-block:: bash

      $ dpct --query-api-mapping=cudaMalloc
      CUDA API:
        cudaMalloc(pDev /*void ***/, s /*size_t*/);
      Is migrated to:
        *pDev = (void *)sycl::malloc_device(s, dpct::get_in_order_queue());

- On a Linux OS, you can use the ``tab`` button to auto-complete the CUDA API
  names.

    .. code-block:: bash

        $ dpct --query-api-mapping=cudaMa (press tab)
        $ dpct --query-api-mapping=cudaMalloc

- |tool_name| supports fuzzy matching of CUDA API names.
  For example:

    .. code-block:: bash

        $ dpct --query-api-mapping=cudamalloc
        CUDA API:
          cudaMalloc(pDev /*void ***/, s /*size_t*/);
        Is migrated to:
          *pDev = (void *)sycl::malloc_device(s, dpct::get_in_order_queue());
