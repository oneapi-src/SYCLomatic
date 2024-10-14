Configure environment variables.

To configure the |tool_name| environment, set the following environment variables:

**Linux**

.. code-block::

	export PATH=$PATH_TO_C2S_INSTALL_FOLDER/bin:$PATH
	export CPATH=$PATH_TO_C2S_INSTALL_FOLDER/include:$CPATH

**Windows (64-bit)**

.. code-block::

	SET PATH=%PATH_TO_C2S_INSTALL_FOLDER%\bin;%PATH%
	SET INCLUDE=%PATH_TO_C2S_INSTALL_FOLDER%\include;%INCLUDE%
	SET CPATH=%PATH_TO_C2S_INSTALL_FOLDER%\include;%CPATH%


If you use the |dpcpp_compiler|, make sure to `set environment variables using
the setvars script <https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2024-2/oneapi-development-environment-setup.html#SETVARS-AND-VARS-FILES>`_.