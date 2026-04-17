#!/bin/bash
#----------------------------------------------------------
# File-name: <michelle_videofiles_konvertieren.sh>
# Language: bash script
# Synopsis: michelle_videofiles_konvertieren.sh (Task 6)
# Description: Find files with the given argument,
#              then convert file to MP4 
#              if the file is not in mp4 form
# Project: Shell Script Programming Course
# Author: michellemi76924@th-nuernberg.de
#----------------------------------------------------------

# check if the keyword and filename are given as arguments
if [[ $# -eq 0 ]] ; then
    echo 'Please enter mp4Converter [FILENAME]'

# check if the keyword "mp4Converter" is given
elif [[ $1 != "mp4Converter" ]] ; then
    echo 'Please enter mp4Converter before the [FILENAME]'
else
    filename=$2
    file=$(find ./ -type f -name "$filename" -print)
    # check if the file exists
    if [[ -z $file ]] ; then 
        echo "File not found, please only convert existing file"
    else
        extension=$(echo ${file##*.})
        # check if the file is not in mp4 form
        if [[ $extension == "mp4" ]] ; then
            echo "file is already in mp4 form"
        else
            name=$(echo ${file%.*})
            echo $name
            ffmpeg -i $file -vcodec libx264 -acodec aac -qscale:v 0 \
                -qscale:a 0 -af volume=1.1 -ab 192k $name.mp4 
        fi
    fi
fi