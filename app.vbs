Set WshShell = CreateObject("WScript.Shell")
WshShell.Run Chr(34) & WScript.ScriptFullName & "\..\run_app.bat" & Chr(34), 0
Set WshShell = Nothing
