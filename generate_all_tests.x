#!/bin/bash

# Check for missing argument
if [ $# -eq 0 ]; then
    echo "Error: Argument missing. Please provide a file name without the .yaml extension."
    exit 1
fi

# Check if the argument has a .yaml extension
if [[ $1 == *.yaml ]]; then
    echo "Error: The argument should not have a .yaml extension."
    exit 1
fi

# $1 is the yaml file or yaml files (without the extension)
./generate_homework_tests.x $1
