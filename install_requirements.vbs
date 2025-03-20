Set WshShell = CreateObject("WScript.Shell")
WshShell.Run Chr(34) & WScript.ScriptFullName & "\..\install_requirements.bat" & Chr(34), 0, True
Set WshShell = Nothing
