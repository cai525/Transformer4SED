#!/bin/bash
# Usage: script.sh directory suffix
# Example: script.sh /home/user txt

# Check if the number of arguments is correct
if [ $# -ne 2 ]; then
  echo "Two arguments are required: directory and suffix"
  exit 1
fi

# Check if the directory exists
if [ ! -d "$1" ]; then
  echo "Directory $1 does not exist"
  exit 2
fi

# Delete all files with suffix $2 in the directory
find "$1" -type f -name "*.$2" -delete
echo "All files with suffix $2 in $1 have been deleted"
