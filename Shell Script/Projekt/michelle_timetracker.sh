#!/bin/bash
#----------------------------------------------------------
# File-name: <michelle_timetracker.sh>
# Language: bash script
# Synopsis: michelle_timetracker.sh (mini project)
# Description: Check and track activity time of the active window
#              further infos read readme
# Project: Shell Script Programming Course
# Author: michellemi76924@th-nuernberg.de
#----------------------------------------------------------

# generates log files if they are not exist
LOGS="${LOGS:-$HOME/.logs}"
OTHER_LOGS="${OTHER_LOGS:-$HOME/.otherlogs}"
HISTORY_LOGS="${HISTORY_LOGS:-$HOME/.historylogs}"

# DESCRIPTION: help option (or manual)
function help()
{
    echo '
    Time Tracker
    ----------------------------
    Check and track activity time of the active window
    during user presence by evaluating mouse movement
    Activities are classified into suitable categories
    Only one active window is allowed at a time
    Time record is saved (unautomatically) on the log file
    ----------------------------
    Options : 
    -h or --help                                          # show this help
    -x or --start                                         # start tracking
    -s or --save                                          # save log entries to .historylogs file
    -w or --show [start date] [finish date] ($2 optional) # show .historylogs file from start until finish date
                                                          # date format YYYY-MM-DD
                                                          # second argument = end -> show .historylogs from start date until last entry of .historylogs file
                                                          # no second argument -> show logs only on start date
    -d or --delete [finish date]                          # delete log entries of .historylogs file from first entry until finish date
                                                          # date format YYYY-MM-DD
    ----------------------------
    Category :
    Browsing            -> Mozilla Firefox
    Email               -> Mozilla Thunderbird Mail
    Edit Source Code    -> Visual Studio Code
    Virtual Meeting     -> Microsoft Teams and Zoom
    Productivity        -> LibraOffice and Gedit
    Linux terminal'

}

# DESCRIPTION: track the time (in minutes) of active window
# PARAMETER: none
# RETURN VALUE: 0 for success, 1 for fail
# STDOUT: tracked time of active window saved in .logs file
function start_track()
{
    # DESCRIPTION: increment the time (in minutes) of the category
    # PARAMETER: category name
    # RETURN VALUE: 0 for success, 1 for fail
    # STDOUT: the time of the category +1 in .logs file
    function track()
    {
        category=$1
        count=$(grep $category $LOGS | grep -oE '[0-9]+')
        (( count++ ))
        sed --in-place -e "s/$category=.*/$category=$count/" $LOGS
    }

    echo 'Time tracking is started'
    echo 'Ctrl + C to stop the program'

    while true ; do
        # get mouse location
        eval $(xdotool getmouselocation --shell 2>/dev/null)
        # mouse movement occured?
        if [[ $X != $X0 ]] && [[ $Y != $Y0 ]] ; then
            window=$(xdotool getwindowfocus getwindowname 2>/dev/null)
            # activities categorizing
            case $window in
                *"Mozilla Firefox")
                    track "browsing"     
                    ;;

                 *"Visual Studio Code")
                    track "edit source code"
                    ;;

                *"Mozilla Thunderbird")
                    track "email"
                    ;;

                $(whoami)@$(hostname)* | root@$(hostname)*)
                    track "linux terminal"
                    ;;

                *LibreOffice* | *gedit)
                    track "productivity"
                    ;;

                *"Microsoft Teams" | Zoom*)
                    track "virtual meeting"
                    ;;
                
                # other category will be stored on OTHER_LOGS
                # purpose : to review other activities
                *) 
                    # to write only once in OTHER_LOGS, preventing clutter
                    if [[ $(grep $window $OTHER_LOGS) -eq 0 ]] ; then
                        echo $window >> "$OTHER_LOGS"
                    fi
                    ;;
            esac
        fi
        # previous location stored 
        X0=$(( X ))
        Y0=$(( Y ))
        # one minute interval 
        sleep 60
    done 
}

# DESCRIPTION: summarize and save log into .historylogs file
# PARAMETER: none
# RETURN VALUE: 0 for success, 1 for fail
# STDOUT: time summary, on the date this function called, 
#         saved in .historylogs file
function save_track()
{
    # save the date when this option is called
    echo -n $(date +"%Y-%m-%d :") >> "$HISTORY_LOGS" 
    for session in "browsing" "edit source code" "email" "linux terminal" "productivity" "virtual meeting" ; do
        line=$(grep "$session" "$LOGS")
        # more than 1 minute -> print (time) minutes
        if [[ $(echo $line | grep -oE '[0-9]+') -gt 1 ]] ; then 
            echo -n $line minutes >> "$HISTORY_LOGS" 
        else
            echo -n $line minute >> "$HISTORY_LOGS" 
        fi
        # for the last category -> no comma in the end but new line
        if [[ $session != "virtual meeting" ]] ; then
            echo -n ", " >> "$HISTORY_LOGS"
        else
            echo " " >> "$HISTORY_LOGS"
        fi
    done
}

# DESCRIPTION: show time summary from .historylogs file
# PARAMETER: start date and finish date
# RETURN VALUE: 0 for success, 1 for fail
# STDOUT: time summary from .historylogs file on or from the start 
#         until end date or until last .logs file entry
function show_track()
{
    # declare start and finish date variable
    start=$1
    finish=$2

    # start date available
    if [[ ! -z "$start" ]] ; then
        # check start date format
        # start date = start -> print out all entries of .historylogs file
        if [[ $start =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] && date -d "$start" >/dev/null 2>&1 || [[ $start=="start" ]]; then
            # check if there is log entry on the start date
            if [[ $(grep $start $HISTORY_LOGS | wc -l) -gt 0 ]] ; then
                # empty finish date, show the log only on the start date!
                if [[ -z "$finish" ]] ; then
                    grep "$start" "$HISTORY_LOGS"
                # show log from start date to the last log entry
                elif [[ $finish == "end" ]] ; then
                    sed -n "/$start/",'/$!d/p' "$HISTORY_LOGS"
                else
                    # check finish date format
                    if [[ $finish =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] && date -d "$finish" >/dev/null 2>&1 ; then
                        # check if the finish date is available, otherwise print until the last entry
                        if [[ $(grep $finish $HISTORY_LOGS | wc -l) -gt 0 ]] ; then
                            sed -n "/^$start/,/^$finish/p" $HISTORY_LOGS
                        else
                            echo "No log entry on finish date"
                            sed -n "/$start/",'/$!d/p' "$HISTORY_LOGS"
                        fi
                    else 
                        echo "Please enter a correct date on the format YYYY-MM-DD"
                        help
                    fi
                fi
            else
                echo "No history on this start date"
                cat $HISTORY_LOGS
            fi
        else
            echo "Please enter a correct date on the format YYYY-MM-DD"
            help
        fi      
    else
        # error -> no date at all
        echo "Please enter the date"
        help
    fi 

}

# DESCRIPTION: delete log entries in .historylogs file from first entry until given date
# PARAMETER: start date (until then the entries will be deleted)
# RETURN VALUE: 0 for success, 1 for fail
# STDOUT: no output
function delete_log()
{
    # declare start date variable
    until=$1

    if [[ -z $until ]] ; then
        echo "Please enter the date, until then the log entries will be deleted"
        help
    else
        # check if it's correct date on the correct format
        # $until = all -> delete all
        if [[ $until =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] && date -d "$until" >/dev/null 2>&1 || [[ $until == "all" ]]; then
            # check if start date is available
            if [[ $(grep $until $HISTORY_LOGS | wc -l) -gt 0 ]] ; then
                sed -i "1,/$until/d" $HISTORY_LOGS
            # delete all
            elif [[ $until == "all" ]] ; then
                echo -n "" > $HISTORY_LOGS
            else
                # start date unavailable -> print out all entries of .historylogs
                show_track start end
            fi
        else
            echo "Please enter correct date on the format YYYY-MM-DD"
        fi
    fi 
}

# DESCRIPTION: redirect the program according to given option
# PARAMETER: options (and arguments for some options)
# RETURN VALUE: 0 for success, 1 for fail
# STDOUT: output depends on the given option
function options()
{
    case $1 in
        -h | --help)
            help
            ;;

        -x | --start)
            # initializing fresh .logs and .otherlogs file
            echo 'browsing=0
            edit source code=0
            email=0
            linux terminal=0
            productivity=0
            virtual meeting=0' > $LOGS
            echo -n "" > $OTHER_LOGS
            start_tracḱ
            ;;

        -s | --save)
            save_track
            ;;
        
        -w | --show)
            # passing second argument as start date
            # and third argument as finish date
            show_track $2 $3
            
            ;;
        
        -d | --delete)
            # passing second argument as start date
            delete_log $2
            ;;
        
        *)
            echo "Invalid arguments, please check manual"
            help
            ;;
    esac
}

#------------------------------
# MAIN
#------------------------------

if [[ $# -eq 0 ]] ; then
# No parameters = show help
    echo "Please enter an option"
    help
else
    options "$1" "$2" "$3"
fi
