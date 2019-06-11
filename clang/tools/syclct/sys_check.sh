#
# Compatibility Tool System Check
#
# a. Detect CUDA toolkit.
# b. profit


#this syscheck script is provided as a model for every OneAPI component
#each syscheck script should do five things:
#
# 1 - load the common.sh file   to gain the common API functions
# 2 - initialize an ERRORSTATE value to 0,
# 3 - return ERRORSTATE at the end of the shell script
# 4 - 'echo'   any problems out to the user. (and adjust ERRORSTATE if so)
# 5 - 'speak'  other messages to the user.

# 'echo' should be used for outputting messages in response to errors. 'echo' is always output.
# 'speak' outputs only if the -v verbose flag it used. Affirmative messages ( "Everything OK!" ) should
#    use 'speak', as well as advice, informative messages, or possibly longer explanations of an error. (eg "your cmake installation is not the latest" )
#
#  colors for use with 'echo' and 'speak' are defined.  See  common.sh for list and usage example.

# any arguments passed to the root syscheck script are passed on to this one.

# ERRORSTATE: 0 if OK, 1 if not.



#load common file
source common.sh $@

#every syscheck script should set up an ERRORSTATE variable and return it on completion.
ERRORSTATE=0


# CUDA TOOLKIT
# the compatibility tool requires the cuda toolkit to be installed. There are various ways this could be done.
# could look for nvcc on the PATH, or search for 'cuda' on the LD_LIBRARY_PATH
# opting for the latter,
if [ -z $(grep 'cuda' <<< $LD_LIBRARY_PATH) ]; then
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
