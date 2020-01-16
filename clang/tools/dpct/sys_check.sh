###############################################################################
#
#Copyright 2019 - 2020 Intel Corporation.
#
#This software and the related documents are Intel copyrighted materials,
#and your use of them is governed by the express license under which they
#were provided to you ("License"). Unless the License provides otherwise,
#you may not use, modify, copy, publish, distribute, disclose or transmit
#this software or the related documents without Intel's prior written
#permission.
#
#This software and the related documents are provided as is, with no express
#or implied warranties, other than those that are expressly stated in the
#License.
#
###############################################################################

# Intel(R) DPC++ Compatibility Tool System Check
#
# 'echo' should be used for outputting messages in response to errors. 'echo' is always output.
# 'speak' outputs only if the -v verbose flag it used. Affirmative messages ( "Everything OK!" ) should
#    use 'speak', as well as advice, informative messages, or possibly longer explanations of an error. (eg "your cmake installation is not the latest" )
#
#  colors for use with 'echo' and 'speak' are defined.  See  common.sh for list and usage example.
#
# any arguments passed to the root syscheck script are passed on to this one.

# ERRORSTATE: 0 if OK, 1 if not.

#location of this sh file
LOC=$(dirname $(realpath "${BASH_SOURCE[0]}"))

#load common file
source $LOC/../../../common.sh $@

#every syscheck script should set up an ERRORSTATE variable and return it on completion.
ERRORSTATE=0

if [ -z $(which python) ]; then
    echo -e "The Intel(R) DPC++ Compatibility Tool requires the python to be installed."
    ERRORSTATE=1
fi

if [ $ERRORSTATE -eq 0 ]; then
    speak "OK"
fi

#always return ERRORSTATE ( which is 0 if no error )
return $ERRORSTATE
