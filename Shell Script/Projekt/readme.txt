Time Tracker
----------------------------
Check and track activity time of the active window, during user presence by evaluating mouse movement. 
Evaluation period is every one minute
Activities are classified into suitable categories. Tracked apps should be adjusted in the source code
Activities (or apps) that are not listed are tracked in .otherlogs file (will be saved only before the next program initiation)
Only one active window is allowed at a time and will be tracked 
Time record is saved on the log file, until the next initiation of the program. 
To save it in longer period, log file should be saved on the history log file -> option -s
Past time record that is not saved in history log file will be permanently deleted after next program initiation
History log file can be deleted with option -d
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
                                                      # date format YYYY-MM-DD or all (-> for delete all)
----------------------------
Category :
Browsing            -> Mozilla Firefox
Email               -> Mozilla Thunderbird Mail
Edit Source Code    -> Visual Studio Code
Virtual Meeting     -> Microsoft Teams and Zoom
Productivity        -> LibraOffice and Gedit
Linux terminal

Author = Michelle Michelle, BMF7 (Matrikelnr. 3275722)

------------------------------------------------------------------------------------------------------------------------------------------------------------------

Time Tracker
----------------------------
Überprüfung und Verfolgung der Aktivitätszeit des aktiven Fensters, während der Anwesenheit des Benutzers durch Auswertung der Mausbewegungen. 
Auswerteperiode ist jede Minute
Aktivitäten werden in geeignete Kategorien eingeteilt. Verfolgte Apps sollten im Quellcode angepasst werden
Aktivitäten (oder Applikationen), die nicht aufgelistet sind, werden in der Datei .otherlogs verfolgt (wird nur vor dem nächsten Programmstart gespeichert)
Nur ein aktives Fenster ist gleichzeitig erlaubt und wird verfolgt 
Die Zeitaufzeichnung wird in der Protokolldatei bis zum nächsten Programmstart gespeichert. 
Um sie über einen längeren Zeitraum zu speichern, sollte die .logs Datei in der .historylogs Datei gespeichert werden -> Option -s
Vergangene Zeitaufzeichnungen, die nicht in der .historylogs Datei gespeichert sind, werden nach dem nächsten Programmstart endgültig gelöscht
Die .historylogs Datei kann mit der Option -d gelöscht werden.
----------------------------
Optionen: 
-h oder --help # diese Hilfe anzeigen
-x oder --start # Aufzeichnung starten
-s oder --save # speichert Protokolleinträge in der Datei .historylogs
-w oder --show [Startdatum] [Enddatum] ($2 optional)    # zeige .historylogs Datei vom Start bis zum Enddatum
                                                        # Datumsformat JJJJ-MM-TT
                                                        # zweites Argument = end -> zeige .historylogs vom Startdatum bis zum letzten Eintrag der .historylogs-Datei
                                                        # kein zweites Argument -> zeige Protokolle nur am Startdatum
-d oder --delete [finish date]                          # löscht Protokolleinträge der .historylogs Datei vom ersten Eintrag bis zum Enddatum
                                                        # Datumsformat JJJJ-MM-TT oder all (für alle löschen)
----------------------------
Kategorie :
Browsing            -> Mozilla Firefox
Email               -> Mozilla Thunderbird Mail
Edit Source Code    -> Visual Studio Code
Virtual Meeting     -> Microsoft Teams and Zoom
Productivity        -> LibraOffice and Gedit
Linux terminal

Autor = Michelle Michelle, BMF7 (Matrikelnr. 3275722)