#!/bin/bash

# Set the package directory
PACKAGE_DIR="ai4ecg"

# Find all directories in the package
DIRS=$(find $PACKAGE_DIR -type d)

# Create __init__.py files in all directories
for DIR in $DIRS; do
    touch $DIR/__init__.py
done

# Add import statements to __init__.py files
for DIR in $DIRS; do
    # Get the relative path of the directory
    RELATIVE_PATH=${DIR#$PACKAGE_DIR/}
    # Get the list of subdirectories and files in the directory
    SUBDIRS=$(find $DIR -maxdepth 1 -type d ! -path $DIR)
    FILES=$(find $DIR -maxdepth 1 -type f -name "*.py" ! -name "__init__.py")
    # Add import statements for subdirectories and files
    for SUBDIR in $SUBDIRS; do
        MODULE_NAME=${SUBDIR#$DIR/}
        echo "from .$MODULE_NAME import *" >> $DIR/__init__.py
    done
    for FILE in $FILES; do
        MODULE_NAME=$(basename $FILE .py)
        echo "from .$MODULE_NAME import *" >> $DIR/__init__.py
    done
done