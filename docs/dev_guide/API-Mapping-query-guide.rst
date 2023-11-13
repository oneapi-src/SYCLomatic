API-Mapping Query Guide
=======================

|tool_name| provides an option for users to get which SYCL API will be used when
migrating a CUDA API.

For example:

   .. code-block:: bash

      $ dpct --query-api-mapping=cudaMalloc
      CUDA API:
        cudaMalloc(pDev /*void ***/, s /*size_t*/);
      Is migrated to:
        *pDev = (void *)sycl::malloc_device(s, dpct::get_in_order_queue());

- You can use ``tab`` button to auto-complete the CUDA API names.

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
