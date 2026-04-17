#!/bin/bash
#----------------------------------------------------------
# File-name: <michelle_dollarkurs.sh>
# Language: bash script
# Synopsis: michelle_dollarkurs.sh (Task 7)
# Description: Print the current dollar to euro rate 
# Project: Shell Script Programming Course
# Author: michellemi76924@th-nuernberg.de
#----------------------------------------------------------

# download the website and search for a line contains "currentValue"
line=$(wget -qO- https://markets.businessinsider.com/currencies/eur-usd | grep "currentValue")

# delete words before and after the rate
delete_front=${line#*\"currentValue\":}
delete_back=${delete_front%%,*}

# print the current dollar rate
echo "$delete_back $"