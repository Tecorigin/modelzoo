#!/bin/bash
# Func:
# Collect json path to one single file.
# Usage:
# $1 cases_dir, specify input dir contains json cases
# $2 output_file, specify output file contains json cases paths
#
# e.g. bash collect_dir.sh json/resnet50/

set -e

# Define the cases path
cases_dir=$1

# Define the output file
output_file=$2

# Clear the output file at the beginning
> "$output_file"

# Use find to iterate over all files in the current directory and subdirectories
find $cases_dir -type f | while read file; do
    # Check if the file has a .json extension
    if echo "$file" | grep -q "\.json$"; then
        # If it is a JSON file, get its directory name and append to the output file
        echo "$file" >> "$output_file"
    fi
done

# Sort the output file in-place and remove duplicates
sort -u "$output_file" -o "$output_file"

echo "Directory names of JSON files collected in $output_file"
