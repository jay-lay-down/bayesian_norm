Option Explicit

' Dashboard의 B4 또는 C4가 바뀌면 즉시 새로고침
Private Sub Workbook_SheetChange(ByVal Sh As Object, ByVal Target As Range)
    On Error GoTo ExitHandler
    If Not TypeOf Sh Is Worksheet Then Exit Sub
    If Sh.Name <> "Dashboard" Then Exit Sub

    ' [--- 삭제된 부분 1: 특정 셀(B4:C4)이 변경되었는지 확인하는 조건문 ---]
    If [...주요 조건문...] Then Exit Sub


    ' [--- 삭제된 부분 2: 무한 루프를 방지하는 코드 ---]


    ' === 모듈의 Public Sub 호출 ===
    On Error Resume Next
    
    ' [--- 삭제된 부분 3: 실제로 호출할 매크로 이름 ---]
    Application.Run [...호출할 매크로 이름...]
    
    If Err.Number <> 0 Then
        Err.Clear
        Application.Run [...호출할 매크로 이름...]
    End If
    On Error GoTo 0

ExitHandler:
    Application.EnableEvents = True
End Sub
