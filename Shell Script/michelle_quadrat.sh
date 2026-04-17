#!/bin/bash
#----------------------------------------------------------
# File-name: <michelle_quadrat.sh>
# Language: bash script
# Synopsis: michelle_quadrat.sh (Task 4)
# Description: Draw a square with a given parameter
# Project: Shell Script Programming Course
# Author: michellemi76924@th-nuernberg.de
#----------------------------------------------------------

function GetRow ()
# DESCRIPTION: Print as many characters as the given number
# PARAMETER: 1. Total number of characters 
#            2. Type of character
# RETURN VALUE: 0 for success, 1 for fail
# STDOUT: one line of given type of character
{
    for ((i = 1; i <= $1; i++)); do 
        echo -n "$2"
    done
}

echo -n "quadrat "

# input parameter for the square
read width

# determine the inner length and inner width of the square
# inner = lines between the first and the last 
#         line of the square
inner_length=$(( $width/2-2 ))
inner_width=$(( $width-2 ))

# print the first line of the square
GetRow $width '*'

# print each line of the inner square 
for ((l = 1; l <= $inner_length; l++)); do
# repeat as many as the inner length is
    echo " "
    # left frame
    echo -n '*'
    GetRow $inner_width " "
    # right frame
    echo -n '*'
done 
echo ""
# print the last line of the square
GetRow $width '*'
echo ""