#!/bin/bash
#----------------------------------------------------------
# File-name: <michelle_dateien_suchen.sh>
# Language: bash script
# Synopsis: michelle_dateien_suchen.sh (Task 2)
# Description: find files starts with date of creation 
#              with the corresponding requirement in a directory
# Project: Shell Script Programming Course
# Author: michellemi76924@th-nuernberg.de
#----------------------------------------------------------

# First requirement = find all files created in 2019
#----------------------------------------------------------

find ./ -name "2019*" -print

# Second requirement = find all files created on first January
#----------------------------------------------------------

find ./ -name "????0101*" -print

# Third requirement = find all and only files started  
#                     with capital letters recursively
#----------------------------------------------------------

find ./ -type f -name "[[:upper:]]*" -print
