################################################################################
#
# Copyright (C) 2019 Intel Corporation. All rights reserved.
#
# The information and source code contained herein is the exclusive property of
# Intel Corporation and may not be disclosed, examined or reproduced in whole or
# in part without explicit written authorization from the company.
#
################################################################################

# Compatibility Tool System Check
#
# a. Detect CUDA toolkit.
# b. profit

# 'echo' should be used for outputting messages in response to errors. 'echo' is always output.
# 'speak' outputs only if the -v verbose flag it used. Affirmative messages ( "Everything OK!" ) should
#    use 'speak', as well as advice, informative messages, or possibly longer explanations of an error. (eg "your cmake installation is not the latest" )
#
#  colors for use with 'echo' and 'speak' are defined.  See  common.sh for list and usage example.
#
# any arguments passed to the root syscheck script are passed on to this one.

# ERRORSTATE: 0 if OK, 1 if not.

#load common file
source common.sh $@

#every syscheck script should set up an ERRORSTATE variable and return it on completion.
ERRORSTATE=0

# CUDA TOOLKIT
if [ -z $(which nvcc) ]; then
    echo -e "The OneAPI Compatibility Tool requires the Nvidia CUDA Toolkit to be installed."
    ERRORSTATE=1
fi

if [ -z $(which python) ]; then
    echo -e "The OneAPI Compatibility Tool requires the python to be installed."
    ERRORSTATE=1
fi


if [ $ERRORSTATE -eq 0 ]; then
    speak "OK"
fi

#always return ERRORSTATE ( which is 0 if no error )
return $ERRORSTATE
