#!/bin/bash

# Uninstall the Python package using pip
yes | pip uninstall tinynet

# Read the version from pyproject.toml and store it in a variable
version=$(awk -F'"' '/version =/ {print $2}' pyproject.toml)

# Print the version
echo "Version: $version"

if [ -d "dist" ]; then
  rm -rf dist
fi

if [ -d "tinynet.egg-info" ]; then
  rm -rf tinynet.egg-info
fi

# build the package
python -m build
# python setup.py build

# install builded package 
cd dist
command="pip install tinynet-${version}.tar.gz"
eval $command


