#!/bin/bash
#----------------------------------------------------------
# File-name: <michelle_hauptspeicher.sh>
# Language: bash script
# Synopsis: michelle_hauptspeicher.sh (Task 3)
# Description: read the size of the installed main memory
# Project: Shell Script Programming Course
# Author: michellemi76924@th-nuernberg.de
#----------------------------------------------------------

# Command line to show the size of installed main memory in GB
#-------------------------------------------------------------

awk '$1=="MemTotal:" {print $2/(1024^2)}' /proc/meminfo

function GetHardwareMemoryTotal ()
# DESCRIPTION: get the size of the total physical memory (RAM)
# PARAMETER: filename (incl. path)
# RETURN VALUE: size of installed main memory and 0 for success
#               print error message and 1 for fail
# STDOUT: Size of file in GBytes
{
    if [[ ! -f "$1" ]] ; then
    echo "ERROR: File not found" >&2
    return 1
    fi
    grep MemTotal "$1" | awk '{$2=$2/(1024^2); print $2}'
    return 0
}

# returned value from GetHardwareMemoryTotal() is saved in the following variable
size=$(GetHardwareMemoryTotal /proc/meminfo)

# status zero (error free) then print 
[[ $? == 0 ]] && echo "die Größe des installierten Hauptspeichers : $size GB." 

# check the type of computer based on the size of the total memory
echo $size | awk '
{
    if ($1 < 2) print "Kinderrechner"
    else if ($1 > 8) print "Profirechner"
    else print "Standardrechner"
}
'
