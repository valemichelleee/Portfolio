#!/bin/bash
#----------------------------------------------------------
# File-name: <michelle_maxint.sh>
# Language: bash script
# Synopsis: michelle_maxint.sh (Task 5)
# Description: maximum number that bash can calculate with
# Project: Shell Script Programming Course
# Author: michellemi76924@th-nuernberg.de
#----------------------------------------------------------

# status of the architecture type (unsigned or signed)
status=0

# increment of 1 byte
bit=8

# while status of the architecture isn't determined
while [[ $status == 0 ]] ; do
    # maximum value
    max=$(( 2**$bit ))

    # if maximum value isn't reached
    if [[ $max -ne 0 ]] ; then
        # increment to the next byte
        bit=$(( $bit*2 ))
    else
        # check the minimum value 
        # by increment it one bit before (2^(n-1))
        bit_check=$(( $bit-1 ))
        test=$(( 2**$bit_check ))

        # if the minimum value is negative
        if [[ $test -lt 0 ]] ; then
            status="signed integer"
            # maximum value = 2^(n-1)-1
            max=$(( 2**$bit_check-1 ))
        else
            status="unsigned integer"
            # maximum value = 2^n-1
            max=$(( 2**$bit-1 ))
        fi
    fi
done

echo "Maximale Anzahl, die die Bash rechnen kann = $max"
echo "Type = $status"
