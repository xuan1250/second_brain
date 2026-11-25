; Second Brain Desktop — Inno Setup 6 Hardware-Aware Installer Script
; Produces dist/SecondBrain-Setup-x64.exe (Option 1: Slim Installer + First-Run Model Bootstrap)

#define MyAppName "Second Brain"
#define MyAppVersion "0.1.0"
#define MyAppId "{{B7D8F2E4-4F1D-4C2B-9A3E-7C6B9B4E2A10}"
#define MyAppPublisher "Second Brain"
#define MyEntryRelPath "app\installer\launcher_entry.py"

[Setup]
AppId={#MyAppId}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
PrivilegesRequired=lowest
DefaultDirName={autopf}\SecondBrain
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=no
AppMutex=Local\SecondBrainDesktopMutex
LicenseFile=THIRD_PARTY_NOTICES.txt
OutputDir=..\dist
OutputBaseFilename=SecondBrain-Setup-x64
Compression=lzma2/normal
SolidCompression=yes
WizardStyle=modern
SetupIconFile=second-brain.ico
UninstallDisplayIcon={app}\app\installer\second-brain.ico
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\build\staging\app\*"; DestDir: "{app}\app"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\build\staging\bin\*"; DestDir: "{app}\bin"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\build\staging\python\*"; DestDir: "{app}\python"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\build\staging\venv\*"; DestDir: "{app}\venv"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Second Brain"; Filename: "{app}\venv\Scripts\pythonw.exe"; Parameters: """{app}\{#MyEntryRelPath}"""; WorkingDir: "{app}\app\backend"; IconFilename: "{app}\app\installer\second-brain.ico"; AppUserModelID: "SecondBrain.Desktop.App"; Comment: "Launch Second Brain Knowledge Repository"
Name: "{group}\Uninstall Second Brain"; Filename: "{uninstallexe}"
Name: "{userdesktop}\Second Brain"; Filename: "{app}\venv\Scripts\pythonw.exe"; Parameters: """{app}\{#MyEntryRelPath}"""; WorkingDir: "{app}\app\backend"; IconFilename: "{app}\app\installer\second-brain.ico"; AppUserModelID: "SecondBrain.Desktop.App"; Comment: "Launch Second Brain Knowledge Repository"; Tasks: desktopicon

[Run]
Filename: "{app}\venv\Scripts\pythonw.exe"; Parameters: """{app}\{#MyEntryRelPath}"""; WorkingDir: "{app}\app\backend"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\app"
Type: filesandordirs; Name: "{app}\bin"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\venv"
Type: filesandordirs; Name: "{app}"
Type: filesandordirs; Name: "{group}"
Type: filesandordirs; Name: "{userprograms}\Second Brain"
Type: files; Name: "{userdesktop}\Second Brain.lnk"
Type: filesandordirs; Name: "{localappdata}\SecondBrain\runtime"
Type: filesandordirs; Name: "{localappdata}\SecondBrain\webview2_cache"
Type: filesandordirs; Name: "{localappdata}\SecondBrain\logs"
Type: filesandordirs; Name: "{localappdata}\SecondBrain\data\models"

[Code]
type
  TMemoryStatusEx = record
    dwLength: DWORD;
    dwMemoryLoad: DWORD;
    ullTotalPhys: Int64;
    ullAvailPhys: Int64;
    ullTotalPageFile: Int64;
    ullAvailPageFile: Int64;
    ullTotalVirtual: Int64;
    ullAvailVirtual: Int64;
    ullAvailExtendedVirtual: Int64;
  end;

function GlobalMemoryStatusEx(var lpBuffer: TMemoryStatusEx): BOOL;
  external 'GlobalMemoryStatusEx@kernel32.dll stdcall';

function GetDiskFreeSpaceExW(
  lpDirectoryName: string;
  var lpFreeBytesAvailableToCaller: Int64;
  var lpTotalNumberOfBytes: Int64;
  var lpTotalNumberOfFreeBytes: Int64
): BOOL;
  external 'GetDiskFreeSpaceExW@kernel32.dll stdcall';

const
  EDGE_WEBVIEW2_GUID = '{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}';
  WEBVIEW2_DOWNLOAD_URL = 'https://go.microsoft.com/fwlink/p/?LinkId=2124703';
  VCREDIST_DOWNLOAD_URL = 'https://aka.ms/vs/17/release/vc_redist.x64.exe';
  DESKTOP_MUTEX_NAME = 'Local\SecondBrainDesktopMutex';
  UNINSTALL_REG_SUBKEY = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{#StringChange(MyAppId, '{{', '{')}_is1';

var
  HwPage: TWizardPage;
  SummaryHeaderLabel: TNewStaticText;
  CpuLabel: TNewStaticText;
  RamLabel: TNewStaticText;
  GpuLabel: TNewStaticText;
  DiskLabel: TNewStaticText;
  PrereqLabel: TNewStaticText;
  StatusNoticeLabel: TNewStaticText;
  ModeHeaderLabel: TNewStaticText;
  RadioCpu: TNewRadioButton;
  RadioCuda: TNewRadioButton;
  ModeHintLabel: TNewStaticText;
  InstallPrereqButton: TNewButton;
  RecheckButton: TNewButton;

  DetectedCpuName: string;
  DetectedRamGB: Extended;
  DetectedRamMB: Int64;
  DetectedGpuName: string;
  DetectedGpuVramMB: Integer;
  HasCompatibleNvidia: Boolean;
  DetectedDrive: string;
  DetectedFreeDiskGB: Extended;
  DetectedFreeDiskMB: Int64;
  HasWebView2: Boolean;
  WebView2Version: string;
  HasVcRedist: Boolean;
  UserOverrodeMode: Boolean;
  IsCheckingConfig: Boolean;
  SuppressRadioClick: Boolean;

function StripUtf8Bom(const Text: string): string;
begin
  Result := Text;
  if (Length(Result) >= 1) and (Ord(Result[1]) = $FEFF) then
    Delete(Result, 1, 1);
end;

function LoadUtf8TextFile(const FilePath: string; var TextOut: string): Boolean;
var
  RawBytes: AnsiString;
begin
  TextOut := '';
  Result := LoadStringFromFile(FilePath, RawBytes);
  if Result then
    TextOut := StripUtf8Bom(Utf8Decode(RawBytes));
end;

function SaveUtf8TextFile(const FilePath, TextContent: string): Boolean;
var
  Utf8Bytes: AnsiString;
  TmpPath: string;
begin
  Utf8Bytes := Utf8Encode(TextContent);
  TmpPath := FilePath + '.tmp';
  Result := SaveStringToFile(TmpPath, Utf8Bytes, False);
  if Result then
  begin
    if FileExists(FilePath) then
      DeleteFile(FilePath);
    Result := RenameFile(TmpPath, FilePath);
    if not Result then
    begin
      DeleteFile(TmpPath);
      Result := SaveStringToFile(FilePath, Utf8Bytes, False);
    end;
  end;
end;

function JsonEscape(const S: string): string;
var
  I: Integer;
  Ch: Char;
begin
  Result := '';
  for I := 1 to Length(S) do
  begin
    Ch := S[I];
    if Ch = '\' then
      Result := Result + '\\'
    else if Ch = '"' then
      Result := Result + '\"'
    else if Ch = #13 then
      Result := Result + '\r'
    else if Ch = #10 then
      Result := Result + '\n'
    else if Ch = #9 then
      Result := Result + '\t'
    else
      Result := Result + Ch;
  end;
end;

function IsJsonWhitespace(Ch: Char): Boolean;
begin
  Result := (Ch = ' ') or (Ch = #9) or (Ch = #13) or (Ch = #10);
end;

function FindJsonValueStart(const JsonText, KeyName: string): Integer;
var
  SearchToken: string;
  SearchFrom, FoundIdx, AfterKeyIdx, Len: Integer;
  SubTail: string;
begin
  Result := 0;
  SearchToken := '"' + KeyName + '"';
  Len := Length(JsonText);
  SearchFrom := 1;

  while SearchFrom <= Len do
  begin
    SubTail := Copy(JsonText, SearchFrom, Len - SearchFrom + 1);
    FoundIdx := Pos(SearchToken, SubTail);
    if FoundIdx = 0 then
      Exit;

    FoundIdx := SearchFrom + FoundIdx - 1;
    AfterKeyIdx := FoundIdx + Length(SearchToken);
    while (AfterKeyIdx <= Len) and IsJsonWhitespace(JsonText[AfterKeyIdx]) do
      AfterKeyIdx := AfterKeyIdx + 1;

    if (AfterKeyIdx <= Len) and (JsonText[AfterKeyIdx] = ':') then
    begin
      AfterKeyIdx := AfterKeyIdx + 1;
      while (AfterKeyIdx <= Len) and IsJsonWhitespace(JsonText[AfterKeyIdx]) do
        AfterKeyIdx := AfterKeyIdx + 1;
      if AfterKeyIdx <= Len then
        Result := AfterKeyIdx;
      Exit;
    end;

    SearchFrom := FoundIdx + Length(SearchToken);
  end;
end;

function ExtractJsonStringField(
  const JsonText, KeyName: string;
  var Found: Boolean;
  var IsNull: Boolean
): string;
var
  ValIdx, Len: Integer;
  Ch, NextCh: Char;
begin
  Result := '';
  Found := False;
  IsNull := True;

  ValIdx := FindJsonValueStart(JsonText, KeyName);
  if ValIdx = 0 then
    Exit;

  Found := True;
  if LowerCase(Copy(JsonText, ValIdx, 4)) = 'null' then
  begin
    IsNull := True;
    Exit;
  end;

  Len := Length(JsonText);
  if JsonText[ValIdx] <> '"' then
  begin
    Found := False;
    Exit;
  end;

  IsNull := False;
  ValIdx := ValIdx + 1;
  while ValIdx <= Len do
  begin
    Ch := JsonText[ValIdx];
    if Ch = '"' then
      Exit;
    if (Ch = '\') and (ValIdx < Len) then
    begin
      ValIdx := ValIdx + 1;
      NextCh := JsonText[ValIdx];
      if NextCh = 'n' then
        Result := Result + #10
      else if NextCh = 'r' then
        Result := Result + #13
      else if NextCh = 't' then
        Result := Result + #9
      else
        Result := Result + NextCh;
    end
    else
      Result := Result + Ch;
    ValIdx := ValIdx + 1;
  end;
end;

function ExtractJsonBoolField(const JsonText, KeyName: string; DefaultVal: Boolean): Boolean;
var
  ValIdx: Integer;
begin
  Result := DefaultVal;
  ValIdx := FindJsonValueStart(JsonText, KeyName);
  if ValIdx = 0 then
    Exit;

  if LowerCase(Copy(JsonText, ValIdx, 4)) = 'true' then
    Result := True
  else if LowerCase(Copy(JsonText, ValIdx, 5)) = 'false' then
    Result := False;
end;

function ExtractJsonIntField(const JsonText, KeyName: string; DefaultVal: Integer): Integer;
var
  ValIdx, Len: Integer;
  Digits: string;
begin
  Result := DefaultVal;
  ValIdx := FindJsonValueStart(JsonText, KeyName);
  if ValIdx = 0 then
    Exit;

  Digits := '';
  Len := Length(JsonText);
  while (ValIdx <= Len) and (JsonText[ValIdx] >= '0') and (JsonText[ValIdx] <= '9') do
  begin
    Digits := Digits + JsonText[ValIdx];
    ValIdx := ValIdx + 1;
  end;

  if Digits <> '' then
    Result := StrToIntDef(Digits, DefaultVal);
end;

procedure ProbeCpuInfo();
var
  CpuVal: string;
begin
  DetectedCpuName := '';
  if not RegQueryStringValue(
    HKEY_LOCAL_MACHINE,
    'HARDWARE\DESCRIPTION\System\CentralProcessor\0',
    'ProcessorNameString',
    CpuVal
  ) then
    CpuVal := '';
  DetectedCpuName := Trim(CpuVal);
  if DetectedCpuName = '' then
    DetectedCpuName := 'Unknown x64 Processor';
end;

procedure ProbeRamInfo();
var
  MemStatus: TMemoryStatusEx;
begin
  DetectedRamMB := 0;
  DetectedRamGB := 0.0;
  MemStatus.dwLength := SizeOf(MemStatus);
  if GlobalMemoryStatusEx(MemStatus) then
  begin
    DetectedRamMB := MemStatus.ullTotalPhys div 1048576;
    DetectedRamGB := Extended(DetectedRamMB) / 1024.0;
  end;
end;

function FindNvidiaSmiPath(): string;
var
  Candidate: string;
begin
  Result := '';
  Candidate := ExpandConstant('{sys}\nvidia-smi.exe');
  if FileExists(Candidate) then
  begin
    Result := Candidate;
    Exit;
  end;

  Candidate := ExpandConstant('{commonpf}\NVIDIA Corporation\NVSMI\nvidia-smi.exe');
  if FileExists(Candidate) then
  begin
    Result := Candidate;
    Exit;
  end;

  if IsWin64 then
  begin
    Candidate := ExpandConstant('{commonpf64}\NVIDIA Corporation\NVSMI\nvidia-smi.exe');
    if FileExists(Candidate) then
      Result := Candidate;
  end;
end;

procedure ProbeNvidiaGpu();
var
  SmiPath, CmdExe, CmdArgs, ProbeOutFile, LineText, NamePart, VramPart: string;
  Lines: TArrayOfString;
  I, CommaPos, DotPos, VramVal, ResultCode: Integer;
begin
  DetectedGpuName := 'No NVIDIA CUDA GPU detected';
  DetectedGpuVramMB := 0;
  HasCompatibleNvidia := False;

  SmiPath := FindNvidiaSmiPath();
  if SmiPath = '' then
    Exit;

  ProbeOutFile := ExpandConstant('{tmp}\sb_nvidia_smi_probe.txt');
  if FileExists(ProbeOutFile) then
    DeleteFile(ProbeOutFile);

  CmdExe := ExpandConstant('{cmd}');
  CmdArgs := '/C ""' + SmiPath + '" --query-gpu=name,memory.total --format=csv,noheader,nounits > "' + ProbeOutFile + '" 2>NUL"';
  if not Exec(CmdExe, CmdArgs, '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    DeleteFile(ProbeOutFile);
    Exit;
  end;

  if (ResultCode = 0) and FileExists(ProbeOutFile) and LoadStringsFromFile(ProbeOutFile, Lines) then
  begin
    for I := 0 to GetArrayLength(Lines) - 1 do
    begin
      LineText := Trim(Lines[I]);
      if LineText = '' then
        Continue;
      CommaPos := Pos(',', LineText);
      if CommaPos > 0 then
      begin
        NamePart := Trim(Copy(LineText, 1, CommaPos - 1));
        VramPart := Trim(Copy(LineText, CommaPos + 1, Length(LineText) - CommaPos));
        DotPos := Pos('.', VramPart);
        if DotPos > 0 then
          VramPart := Copy(VramPart, 1, DotPos - 1);
        VramVal := StrToIntDef(VramPart, 0);
        if (VramVal > DetectedGpuVramMB) or (DetectedGpuName = 'No NVIDIA CUDA GPU detected') then
        begin
          DetectedGpuName := NamePart;
          DetectedGpuVramMB := VramVal;
        end;
      end;
    end;
  end;

  DeleteFile(ProbeOutFile);
  HasCompatibleNvidia := (DetectedGpuVramMB >= 4096);
end;

function ResolveExistingProbeDir(const TargetPath: string): string;
var
  ProbeDir, ParentDir: string;
begin
  ProbeDir := RemoveBackslashUnlessRoot(Trim(TargetPath));
  while (ProbeDir <> '') and (not DirExists(ProbeDir)) do
  begin
    ParentDir := ExtractFileDir(ProbeDir);
    if (ParentDir = '') or (ParentDir = ProbeDir) then
      Break;
    ProbeDir := ParentDir;
  end;

  if (ProbeDir = '') or (not DirExists(ProbeDir)) then
    ProbeDir := ExpandConstant('{localappdata}');

  if (Length(ProbeDir) = 2) and (ProbeDir[2] = ':') then
    ProbeDir := ProbeDir + '\';

  Result := ProbeDir;
end;

procedure ProbeDiskSpaceForTarget(const TargetPath: string);
var
  ProbeDir: string;
  FreeCaller, TotalBytes, TotalFree: Int64;
begin
  DetectedDrive := ExtractFileDrive(TargetPath);
  if DetectedDrive = '' then
    DetectedDrive := 'Target Drive';
  DetectedFreeDiskMB := 0;
  DetectedFreeDiskGB := 0.0;

  ProbeDir := ResolveExistingProbeDir(TargetPath);
  if GetDiskFreeSpaceExW(ProbeDir, FreeCaller, TotalBytes, TotalFree) then
  begin
    DetectedFreeDiskMB := FreeCaller div 1048576;
    DetectedFreeDiskGB := Extended(DetectedFreeDiskMB) / 1024.0;
  end;
end;

procedure ProbePrerequisites();
var
  SubKey, VerStr: string;
  InstalledVal: Cardinal;
begin
  HasWebView2 := False;
  WebView2Version := '';
  SubKey := 'SOFTWARE\Microsoft\EdgeUpdate\Clients\' + EDGE_WEBVIEW2_GUID;

  if RegQueryStringValue(HKEY_LOCAL_MACHINE, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\' + EDGE_WEBVIEW2_GUID, 'pv', VerStr) and (VerStr <> '') and (VerStr <> '0.0.0.0') then
  begin
    HasWebView2 := True;
    WebView2Version := VerStr;
  end
  else if IsWin64 and RegQueryStringValue(HKLM64, SubKey, 'pv', VerStr) and (VerStr <> '') and (VerStr <> '0.0.0.0') then
  begin
    HasWebView2 := True;
    WebView2Version := VerStr;
  end
  else if RegQueryStringValue(HKLM32, SubKey, 'pv', VerStr) and (VerStr <> '') and (VerStr <> '0.0.0.0') then
  begin
    HasWebView2 := True;
    WebView2Version := VerStr;
  end
  else if RegQueryStringValue(HKEY_CURRENT_USER, SubKey, 'pv', VerStr) and (VerStr <> '') and (VerStr <> '0.0.0.0') then
  begin
    HasWebView2 := True;
    WebView2Version := VerStr;
  end
  else if IsWin64 and RegQueryStringValue(HKCU64, SubKey, 'pv', VerStr) and (VerStr <> '') and (VerStr <> '0.0.0.0') then
  begin
    HasWebView2 := True;
    WebView2Version := VerStr;
  end
  else if RegQueryStringValue(HKCU32, SubKey, 'pv', VerStr) and (VerStr <> '') and (VerStr <> '0.0.0.0') then
  begin
    HasWebView2 := True;
    WebView2Version := VerStr;
  end;

  HasVcRedist := False;
  if IsWin64 then
  begin
    if RegQueryDWordValue(HKLM64, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Installed', InstalledVal) and (InstalledVal = 1) then
      HasVcRedist := True;
  end;
  if (not HasVcRedist) and RegQueryDWordValue(HKEY_LOCAL_MACHINE, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Installed', InstalledVal) and (InstalledVal = 1) then
    HasVcRedist := True;
end;

procedure UpdateModeHintLabel();
begin
  if not Assigned(ModeHintLabel) then
    Exit;

  if RadioCuda.Checked then
  begin
    if HasCompatibleNvidia and (DetectedFreeDiskMB >= 12288) then
      ModeHintLabel.Caption := 'Selected: NVIDIA CUDA 12.4 mode. CUDA wheels (~2.5 GB) install into %LOCALAPPDATA%\SecondBrain\runtime\venv on first launch and survive app updates.'
    else if not HasCompatibleNvidia then
      ModeHintLabel.Caption := 'Warning: No NVIDIA GPU with >= 4.0 GB VRAM detected. Standard CPU Mode is strongly recommended.'
    else
      ModeHintLabel.Caption := 'Warning: Less than 12.0 GB free disk space detected for CUDA 12.4 wheels + models.';
  end
  else
    ModeHintLabel.Caption := 'Selected: Standard CPU Mode. Uses bundled CPU PyTorch + FastEmbed BM25 (~3.8 GB offline AI models downloaded on first launch).';
end;

procedure EvaluateHwPageGate();
var
  CudaEligible, AllChecksOk: Boolean;
begin
  CudaEligible := HasCompatibleNvidia and (DetectedFreeDiskMB >= 12288);

  if Assigned(RadioCuda) and Assigned(RadioCpu) then
  begin
    RadioCuda.Enabled := CudaEligible;
    if (not CudaEligible) and (not RadioCpu.Checked) then
    begin
      SuppressRadioClick := True;
      RadioCuda.Checked := False;
      RadioCpu.Checked := True;
      SuppressRadioClick := False;
      UpdateModeHintLabel();
    end;
  end;

  if Assigned(InstallPrereqButton) then
    InstallPrereqButton.Visible := (not HasVcRedist) or (not HasWebView2);

  AllChecksOk :=
    (not IsCheckingConfig) and
    (DetectedRamMB >= 8192) and
    (DetectedFreeDiskMB >= 8192) and
    HasVcRedist and
    HasWebView2 and
    ((not Assigned(RadioCuda)) or (not RadioCuda.Checked) or CudaEligible);

  if (not WizardSilent) and Assigned(HwPage) and (WizardForm.CurPageID = HwPage.ID) then
    WizardForm.NextButton.Enabled := AllChecksOk;
end;

procedure OnComputeRadioClick(Sender: TObject);
begin
  if SuppressRadioClick or IsCheckingConfig then
    Exit;
  UserOverrodeMode := True;
  UpdateModeHintLabel();
  EvaluateHwPageGate();
end;

procedure RefreshHardwareDiagnosticsPage();
var
  GpuText, PrereqText, WarningsText, InstallDirLower: string;
  VramGB: Extended;
begin
  IsCheckingConfig := True;
  if (not WizardSilent) and Assigned(HwPage) and (WizardForm.CurPageID = HwPage.ID) then
    WizardForm.NextButton.Enabled := False;
  if Assigned(RecheckButton) then
    RecheckButton.Enabled := False;
  if Assigned(StatusNoticeLabel) then
    StatusNoticeLabel.Caption := 'Checking system configuration and prerequisites... Please wait.';
  if not WizardSilent then
    WizardForm.Update;

  ProbeCpuInfo();
  ProbeRamInfo();
  ProbeNvidiaGpu();
  ProbeDiskSpaceForTarget(WizardDirValue);
  ProbePrerequisites();

  CpuLabel.Caption := 'CPU: ' + DetectedCpuName;

  if DetectedRamMB >= 8192 then
    RamLabel.Caption := Format('RAM: %.1f GB (meets 8.0 GB minimum)', [DetectedRamGB])
  else
    RamLabel.Caption := Format('RAM: %.1f GB [BLOCKED: < 8.0 GB minimum required]', [DetectedRamGB]);

  if DetectedGpuVramMB > 0 then
  begin
    VramGB := Extended(DetectedGpuVramMB) / 1024.0;
    if HasCompatibleNvidia then
      GpuText := Format('%s (%.1f GB VRAM - CUDA 12.4 compatible)', [DetectedGpuName, VramGB])
    else
      GpuText := Format('%s (%.1f GB VRAM - below 4.0 GB CUDA threshold)', [DetectedGpuName, VramGB]);
  end
  else
    GpuText := DetectedGpuName;
  GpuLabel.Caption := 'GPU / VRAM: ' + GpuText;

  if DetectedFreeDiskMB >= 12288 then
    DiskLabel.Caption := Format('Free Disk (%s): %.1f GB (sufficient for CPU & CUDA 12.4)', [DetectedDrive, DetectedFreeDiskGB])
  else if DetectedFreeDiskMB >= 8192 then
    DiskLabel.Caption := Format('Free Disk (%s): %.1f GB (CPU OK; >= 12.0 GB needed for CUDA)', [DetectedDrive, DetectedFreeDiskGB])
  else
    DiskLabel.Caption := Format('Free Disk (%s): %.1f GB [BLOCKED: < 8.0 GB minimum required]', [DetectedDrive, DetectedFreeDiskGB]);

  if HasWebView2 then
    PrereqText := 'WebView2 Runtime: Installed (' + WebView2Version + ')'
  else
    PrereqText := 'WebView2 Runtime: [REQUIRED - MISSING]';

  if HasVcRedist then
    PrereqText := PrereqText + '  |  VC++ x64: Installed'
  else
    PrereqText := PrereqText + '  |  VC++ x64: [REQUIRED - MISSING]';
  PrereqLabel.Caption := PrereqText;

  WarningsText := '';
  if DetectedRamMB < 8192 then
    WarningsText := 'BLOCKER: At least 8.0 GB system RAM is required.'
  else if DetectedFreeDiskMB < 8192 then
    WarningsText := 'BLOCKER: At least 8.0 GB free disk space is required on the target drive.'
  else if (not HasWebView2) or (not HasVcRedist) then
    WarningsText := 'BLOCKER: Missing required Microsoft runtime prerequisite(s). Click "Download Missing Runtime(s)", install them, then click "Re-check Config" to unlock Next.'
  else
  begin
    InstallDirLower := LowerCase(WizardDirValue);
    if Pos('onedrive', InstallDirLower) > 0 then
      WarningsText := 'Note: Target folder is under OneDrive. Avoid placing your data directory inside OneDrive to prevent SQLite/Qdrant lock conflicts.'
    else if DetectedRamMB < 16384 then
      WarningsText := 'Advisory: < 16 GB RAM detected. Use Fast CPU PDF upload mode for large documents.'
    else
      WarningsText := 'All hardware and system prerequisite checks passed.';
  end;
  StatusNoticeLabel.Caption := WarningsText;

  if not UserOverrodeMode then
  begin
    SuppressRadioClick := True;
    if HasCompatibleNvidia and (DetectedFreeDiskMB >= 12288) then
      RadioCuda.Checked := True
    else
      RadioCpu.Checked := True;
    SuppressRadioClick := False;
  end;

  UpdateModeHintLabel();

  IsCheckingConfig := False;
  if Assigned(RecheckButton) then
    RecheckButton.Enabled := True;
  EvaluateHwPageGate();
end;

procedure OnInstallPrereqClick(Sender: TObject);
var
  ErrCode: Integer;
begin
  if not HasWebView2 then
    ShellExec('open', WEBVIEW2_DOWNLOAD_URL, '', '', SW_SHOWNORMAL, ewNoWait, ErrCode);
  if not HasVcRedist then
    ShellExec('open', VCREDIST_DOWNLOAD_URL, '', '', SW_SHOWNORMAL, ewNoWait, ErrCode);

  if Assigned(StatusNoticeLabel) then
    StatusNoticeLabel.Caption := 'Opened Microsoft runtime download link(s) in your browser. Finish installing the runtime(s), then click "Re-check Config" to continue.';
end;

procedure OnRecheckClick(Sender: TObject);
begin
  if IsCheckingConfig then
    Exit;
  RefreshHardwareDiagnosticsPage();
end;

function ReadConfiguredComputeProfile(const ConfigPath: string): string;
var
  JsonContent, ParsedProfile: string;
  Found, IsNull: Boolean;
begin
  Result := '';
  if not FileExists(ConfigPath) then
    Exit;
  if not LoadUtf8TextFile(ConfigPath, JsonContent) then
    Exit;

  ParsedProfile := LowerCase(Trim(ExtractJsonStringField(JsonContent, 'compute_profile', Found, IsNull)));
  if Found and (not IsNull) and ((ParsedProfile = 'cpu') or (ParsedProfile = 'cuda')) then
    Result := ParsedProfile;
end;

procedure InitializeWizard();
var
  TopPos, ButtonHeight: Integer;
  ExistingConfigPath, ExistingProfile, ParamProfile: string;
begin
  UserOverrodeMode := False;
  IsCheckingConfig := False;
  SuppressRadioClick := False;

  HwPage := CreateCustomPage(
    wpSelectDir,
    'Hardware Diagnostics & Compute Mode',
    'Review detected hardware capabilities and select CPU or NVIDIA CUDA 12.4 mode.'
  );

  TopPos := ScaleY(0);

  SummaryHeaderLabel := TNewStaticText.Create(HwPage);
  SummaryHeaderLabel.Parent := HwPage.Surface;
  SummaryHeaderLabel.Left := 0;
  SummaryHeaderLabel.Top := TopPos;
  SummaryHeaderLabel.Width := HwPage.SurfaceWidth;
  SummaryHeaderLabel.Height := ScaleY(14);
  SummaryHeaderLabel.Font.Style := [fsBold];
  SummaryHeaderLabel.Caption := 'Detected System Profile';
  TopPos := TopPos + ScaleY(15);

  CpuLabel := TNewStaticText.Create(HwPage);
  CpuLabel.Parent := HwPage.Surface;
  CpuLabel.Left := ScaleX(8);
  CpuLabel.Top := TopPos;
  CpuLabel.Width := HwPage.SurfaceWidth - ScaleX(8);
  CpuLabel.Height := ScaleY(14);
  TopPos := TopPos + ScaleY(14);

  RamLabel := TNewStaticText.Create(HwPage);
  RamLabel.Parent := HwPage.Surface;
  RamLabel.Left := ScaleX(8);
  RamLabel.Top := TopPos;
  RamLabel.Width := HwPage.SurfaceWidth - ScaleX(8);
  RamLabel.Height := ScaleY(14);
  TopPos := TopPos + ScaleY(14);

  GpuLabel := TNewStaticText.Create(HwPage);
  GpuLabel.Parent := HwPage.Surface;
  GpuLabel.Left := ScaleX(8);
  GpuLabel.Top := TopPos;
  GpuLabel.Width := HwPage.SurfaceWidth - ScaleX(8);
  GpuLabel.Height := ScaleY(14);
  TopPos := TopPos + ScaleY(14);

  DiskLabel := TNewStaticText.Create(HwPage);
  DiskLabel.Parent := HwPage.Surface;
  DiskLabel.Left := ScaleX(8);
  DiskLabel.Top := TopPos;
  DiskLabel.Width := HwPage.SurfaceWidth - ScaleX(8);
  DiskLabel.Height := ScaleY(14);
  TopPos := TopPos + ScaleY(14);

  PrereqLabel := TNewStaticText.Create(HwPage);
  PrereqLabel.Parent := HwPage.Surface;
  PrereqLabel.Left := ScaleX(8);
  PrereqLabel.Top := TopPos;
  PrereqLabel.Width := HwPage.SurfaceWidth - ScaleX(8);
  PrereqLabel.Height := ScaleY(14);
  TopPos := TopPos + ScaleY(15);

  StatusNoticeLabel := TNewStaticText.Create(HwPage);
  StatusNoticeLabel.Parent := HwPage.Surface;
  StatusNoticeLabel.Left := ScaleX(8);
  StatusNoticeLabel.Top := TopPos;
  StatusNoticeLabel.Width := HwPage.SurfaceWidth - ScaleX(8);
  StatusNoticeLabel.Height := ScaleY(38);
  StatusNoticeLabel.WordWrap := True;
  TopPos := TopPos + ScaleY(39);

  ModeHeaderLabel := TNewStaticText.Create(HwPage);
  ModeHeaderLabel.Parent := HwPage.Surface;
  ModeHeaderLabel.Left := 0;
  ModeHeaderLabel.Top := TopPos;
  ModeHeaderLabel.Width := HwPage.SurfaceWidth;
  ModeHeaderLabel.Height := ScaleY(14);
  ModeHeaderLabel.Font.Style := [fsBold];
  ModeHeaderLabel.Caption := 'Compute Mode Selection';
  TopPos := TopPos + ScaleY(16);

  RadioCpu := TNewRadioButton.Create(HwPage);
  RadioCpu.Parent := HwPage.Surface;
  RadioCpu.Left := ScaleX(8);
  RadioCpu.Top := TopPos;
  RadioCpu.Width := HwPage.SurfaceWidth - ScaleX(8);
  RadioCpu.Height := ScaleY(16);
  RadioCpu.Caption := 'Standard CPU Mode (~3.8 GB models)';
  RadioCpu.Checked := True;
  TopPos := TopPos + ScaleY(17);

  RadioCuda := TNewRadioButton.Create(HwPage);
  RadioCuda.Parent := HwPage.Surface;
  RadioCuda.Left := ScaleX(8);
  RadioCuda.Top := TopPos;
  RadioCuda.Width := HwPage.SurfaceWidth - ScaleX(8);
  RadioCuda.Height := ScaleY(16);
  RadioCuda.Caption := 'NVIDIA GPU Accelerated Mode (CUDA 12.4, +2.5 GB runtime)';
  TopPos := TopPos + ScaleY(17);

  ModeHintLabel := TNewStaticText.Create(HwPage);
  ModeHintLabel.Parent := HwPage.Surface;
  ModeHintLabel.Left := ScaleX(8);
  ModeHintLabel.Top := TopPos;
  ModeHintLabel.Width := HwPage.SurfaceWidth - ScaleX(8);
  ModeHintLabel.Height := ScaleY(37);
  ModeHintLabel.WordWrap := True;
  TopPos := TopPos + ScaleY(38);

  ButtonHeight := ScaleY(23);

  InstallPrereqButton := TNewButton.Create(HwPage);
  InstallPrereqButton.Parent := HwPage.Surface;
  InstallPrereqButton.Left := ScaleX(8);
  InstallPrereqButton.Top := TopPos;
  InstallPrereqButton.Width := ScaleX(190);
  InstallPrereqButton.Height := ButtonHeight;
  InstallPrereqButton.Caption := 'Download Missing Runtime(s)';
  InstallPrereqButton.Visible := False;
  InstallPrereqButton.OnClick := @OnInstallPrereqClick;

  RecheckButton := TNewButton.Create(HwPage);
  RecheckButton.Parent := HwPage.Surface;
  RecheckButton.Left := InstallPrereqButton.Left + InstallPrereqButton.Width + ScaleX(10);
  RecheckButton.Top := TopPos;
  RecheckButton.Width := ScaleX(130);
  RecheckButton.Height := ButtonHeight;
  RecheckButton.Caption := 'Re-check Config';
  RecheckButton.Visible := True;
  RecheckButton.OnClick := @OnRecheckClick;

  ParamProfile := LowerCase(Trim(ExpandConstant('{param:PROFILE|}')));
  ExistingConfigPath := ExpandConstant('{localappdata}\SecondBrain\config\desktop_config.json');
  ExistingProfile := ReadConfiguredComputeProfile(ExistingConfigPath);

  if (ParamProfile = 'cuda') or (ParamProfile = 'cpu') then
  begin
    RadioCuda.Checked := (ParamProfile = 'cuda');
    RadioCpu.Checked := (ParamProfile <> 'cuda');
    UserOverrodeMode := True;
  end
  else if ExistingProfile = 'cuda' then
  begin
    RadioCuda.Checked := True;
    UserOverrodeMode := True;
  end
  else if ExistingProfile = 'cpu' then
  begin
    RadioCpu.Checked := True;
    UserOverrodeMode := True;
  end;

  RadioCpu.OnClick := @OnComputeRadioClick;
  RadioCuda.OnClick := @OnComputeRadioClick;
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if Assigned(HwPage) and (CurPageID = HwPage.ID) then
  begin
    if not WizardSilent then
      WizardForm.NextButton.Enabled := False;
    RefreshHardwareDiagnosticsPage();
  end
  else if Assigned(HwPage) and (CurPageID <> HwPage.ID) then
  begin
    if (not WizardSilent) and (CurPageID <> wpLicense) and (CurPageID <> wpInstalling) then
      WizardForm.NextButton.Enabled := True;
  end;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  MissingMsg: string;
  WasCudaSelected: Boolean;
begin
  Result := True;
  if not Assigned(HwPage) or (CurPageID <> HwPage.ID) then
    Exit;

  if IsCheckingConfig then
  begin
    Result := False;
    Exit;
  end;

  WasCudaSelected := Assigned(RadioCuda) and RadioCuda.Checked;
  RefreshHardwareDiagnosticsPage();

  if DetectedRamMB < 8192 then
  begin
    SuppressibleMsgBox(
      Format('Second Brain requires at least 8.0 GB of system RAM (detected %.1f GB). Installation cannot continue on this machine.', [DetectedRamGB]),
      mbCriticalError,
      MB_OK,
      IDOK
    );
    Result := False;
    Exit;
  end;

  if DetectedFreeDiskMB < 8192 then
  begin
    SuppressibleMsgBox(
      Format('At least 8.0 GB of free disk space is required on %s (detected %.1f GB). Free up space or select another drive before continuing.', [DetectedDrive, DetectedFreeDiskGB]),
      mbCriticalError,
      MB_OK,
      IDOK
    );
    Result := False;
    Exit;
  end;

  if (not WizardSilent) and ((not HasWebView2) or (not HasVcRedist)) then
  begin
    MissingMsg := 'The following required Microsoft runtime prerequisites were not detected:' + #13#10 + #13#10;
    if not HasWebView2 then
      MissingMsg := MissingMsg + '- Microsoft Edge WebView2 Runtime' + #13#10;
    if not HasVcRedist then
      MissingMsg := MissingMsg + '- Microsoft Visual C++ 2015-2022 Redistributable (x64)' + #13#10;
    MissingMsg := MissingMsg + #13#10 + 'Click "Download Missing Runtime(s)", install the missing component(s), and then click "Re-check Config" to continue.';
    SuppressibleMsgBox(MissingMsg, mbError, MB_OK, IDOK);
    Result := False;
    Exit;
  end;

  if (RadioCuda.Checked or WasCudaSelected) and ((not HasCompatibleNvidia) or (DetectedFreeDiskMB < 12288)) then
  begin
    if WizardSilent then
    begin
      SuppressRadioClick := True;
      RadioCpu.Checked := True;
      SuppressRadioClick := False;
      UserOverrodeMode := True;
      UpdateModeHintLabel();
    end
    else
    begin
      SuppressRadioClick := True;
      RadioCpu.Checked := True;
      SuppressRadioClick := False;
      UpdateModeHintLabel();
      EvaluateHwPageGate();
      SuppressibleMsgBox(
        'NVIDIA GPU Accelerated Mode (CUDA 12.4) requires an NVIDIA GPU with >= 4.0 GB VRAM and >= 12.0 GB free disk space. Standard CPU Mode has been selected instead.',
        mbInformation,
        MB_OK,
        IDOK
      );
      Result := False;
      Exit;
    end;
  end;
end;

procedure UpdateVenvPyvenvCfg();
var
  CfgPath, RawContent, LineText, NewContent, KeyPart, TargetPythonHome: string;
  NlPos, EqPos: Integer;
  ReplacedHome: Boolean;
begin
  CfgPath := ExpandConstant('{app}\venv\pyvenv.cfg');
  if not FileExists(CfgPath) then
    Exit;

  if not LoadUtf8TextFile(CfgPath, RawContent) then
    Exit;

  TargetPythonHome := ExpandConstant('{app}\python');
  NewContent := '';
  ReplacedHome := False;

  while RawContent <> '' do
  begin
    NlPos := Pos(#10, RawContent);
    if NlPos > 0 then
    begin
      LineText := Copy(RawContent, 1, NlPos - 1);
      Delete(RawContent, 1, NlPos);
    end
    else
    begin
      LineText := RawContent;
      RawContent := '';
    end;

    if (Length(LineText) > 0) and (LineText[Length(LineText)] = #13) then
      Delete(LineText, Length(LineText), 1);

    EqPos := Pos('=', LineText);
    if EqPos > 0 then
    begin
      KeyPart := Trim(LowerCase(Copy(LineText, 1, EqPos - 1)));
      if KeyPart = 'home' then
      begin
        LineText := 'home = ' + TargetPythonHome;
        ReplacedHome := True;
      end;
    end;

    NewContent := NewContent + LineText + #13#10;
  end;

  if not ReplacedHome then
    NewContent := 'home = ' + TargetPythonHome + #13#10 + NewContent;

  SaveUtf8TextFile(CfgPath, NewContent);
end;

function FormatNullableJsonString(HasValue: Boolean; IsNull: Boolean; const Val: string): string;
begin
  if (not HasValue) or IsNull or (Trim(Val) = '') then
    Result := 'null'
  else
    Result := '"' + JsonEscape(Val) + '"';
end;

function BoolToJson(Val: Boolean): string;
begin
  if Val then
    Result := 'true'
  else
    Result := 'false';
end;

procedure PersistDesktopConfigJson();
var
  ConfigDir, ConfigPath, ExistingJson, SelectedProfile: string;
  SparseOverride, DataDirOverride, OutJson: string;
  SparseFound, SparseIsNull, DataFound, DataIsNull: Boolean;
  ModelsReady, CudaReady, ShowTrayHint: Boolean;
  PreferredPort, PreferredQdrantPort: Integer;
begin
  ConfigDir := ExpandConstant('{localappdata}\SecondBrain\config');
  ForceDirectories(ConfigDir);
  ConfigPath := ConfigDir + '\desktop_config.json';

  SparseFound := False;
  SparseIsNull := True;
  SparseOverride := '';
  DataFound := False;
  DataIsNull := True;
  DataDirOverride := '';
  ModelsReady := False;
  CudaReady := False;
  PreferredPort := 8000;
  PreferredQdrantPort := 6333;
  ShowTrayHint := True;

  if FileExists(ConfigPath) and LoadUtf8TextFile(ConfigPath, ExistingJson) then
  begin
    SparseOverride := ExtractJsonStringField(ExistingJson, 'sparse_mode_override', SparseFound, SparseIsNull);
    DataDirOverride := ExtractJsonStringField(ExistingJson, 'data_dir_override', DataFound, DataIsNull);
    ModelsReady := ExtractJsonBoolField(ExistingJson, 'models_ready', False);
    CudaReady := ExtractJsonBoolField(ExistingJson, 'cuda_runtime_ready', False) and
      FileExists(ExpandConstant('{localappdata}\SecondBrain\runtime\venv\Scripts\python.exe'));
    PreferredPort := ExtractJsonIntField(ExistingJson, 'preferred_port', 8000);
    PreferredQdrantPort := ExtractJsonIntField(ExistingJson, 'preferred_qdrant_port', 6333);
    ShowTrayHint := ExtractJsonBoolField(ExistingJson, 'show_tray_hint', True);
  end;

  if RadioCuda.Checked then
    SelectedProfile := 'cuda'
  else
    SelectedProfile := 'cpu';

  OutJson := '{' + #13#10 +
    '  "compute_profile": "' + SelectedProfile + '",' + #13#10 +
    '  "sparse_mode_override": ' + FormatNullableJsonString(SparseFound, SparseIsNull, SparseOverride) + ',' + #13#10 +
    '  "data_dir_override": ' + FormatNullableJsonString(DataFound, DataIsNull, DataDirOverride) + ',' + #13#10 +
    '  "models_ready": ' + BoolToJson(ModelsReady) + ',' + #13#10 +
    '  "cuda_runtime_ready": ' + BoolToJson(CudaReady) + ',' + #13#10 +
    '  "last_installed_version": "' + JsonEscape('{#MyAppVersion}') + '",' + #13#10 +
    '  "preferred_port": ' + IntToStr(PreferredPort) + ',' + #13#10 +
    '  "preferred_qdrant_port": ' + IntToStr(PreferredQdrantPort) + ',' + #13#10 +
    '  "show_tray_hint": ' + BoolToJson(ShowTrayHint) + #13#10 +
    '}' + #13#10;

  SaveUtf8TextFile(ConfigPath, OutJson);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    UpdateVenvPyvenvCfg();
    PersistDesktopConfigJson();
  end;
end;

function StopRunningDesktopInstance(const InstallRoot: string): Boolean;
var
  PythonwExe, EntryScript, CmdParams, WorkDir: string;
  ResultCode, ElapsedMs: Integer;
begin
  Result := True;
  if not CheckForMutexes(DESKTOP_MUTEX_NAME) then
    Exit;

  PythonwExe := RemoveBackslashUnlessRoot(InstallRoot) + '\venv\Scripts\pythonw.exe';
  EntryScript := RemoveBackslashUnlessRoot(InstallRoot) + '\{#MyEntryRelPath}';
  WorkDir := RemoveBackslashUnlessRoot(InstallRoot) + '\app\backend';

  if FileExists(PythonwExe) and FileExists(EntryScript) then
  begin
    CmdParams := '"' + EntryScript + '" --quit';
    Exec(PythonwExe, CmdParams, WorkDir, SW_HIDE, ewWaitUntilTerminated, ResultCode);
  end;

  ElapsedMs := 0;
  while CheckForMutexes(DESKTOP_MUTEX_NAME) and (ElapsedMs < 10000) do
  begin
    Sleep(250);
    ElapsedMs := ElapsedMs + 250;
  end;

  Result := (not CheckForMutexes(DESKTOP_MUTEX_NAME));
end;

function ResolveExistingInstallRoot(): string;
var
  RegLoc: string;
begin
  Result := '';
  if RegQueryStringValue(HKEY_CURRENT_USER, UNINSTALL_REG_SUBKEY, 'InstallLocation', RegLoc) and (Trim(RegLoc) <> '') then
    Result := RemoveBackslashUnlessRoot(Trim(RegLoc))
  else if IsWin64 and RegQueryStringValue(HKCU64, UNINSTALL_REG_SUBKEY, 'InstallLocation', RegLoc) and (Trim(RegLoc) <> '') then
    Result := RemoveBackslashUnlessRoot(Trim(RegLoc))
  else
    Result := ExpandConstant('{localappdata}\Programs\SecondBrain');
end;

function InitializeSetup(): Boolean;
var
  ExistingRoot: string;
begin
  Result := True;
  if not CheckForMutexes(DESKTOP_MUTEX_NAME) then
    Exit;

  ExistingRoot := ResolveExistingInstallRoot();
  if not StopRunningDesktopInstance(ExistingRoot) then
  begin
    if not WizardSilent then
      SuppressibleMsgBox(
        'Second Brain appears to still be running in the system tray. Please quit Second Brain from the tray icon and retry installation.',
        mbError,
        MB_OK,
        IDOK
      );
    Result := False;
  end;
end;

function InitializeUninstall(): Boolean;
begin
  Result := True;
  if not CheckForMutexes(DESKTOP_MUTEX_NAME) then
    Exit;

  if not StopRunningDesktopInstance(ExpandConstant('{app}')) then
  begin
    if not UninstallSilent then
      SuppressibleMsgBox(
        'Second Brain is still running and could not be closed automatically. Please quit Second Brain from the system tray and retry uninstallation.',
        mbError,
        MB_OK,
        IDOK
      );
    Result := False;
  end;
end;

function HasPurgeCliFlag(): Boolean;
var
  I: Integer;
  ParamUpper: string;
begin
  Result := (ExpandConstant('{param:PURGE|0}') = '1');
  if Result then
    Exit;

  for I := 1 to ParamCount do
  begin
    ParamUpper := UpperCase(Trim(ParamStr(I)));
    if (ParamUpper = '/PURGE=1') or (ParamUpper = '/PURGE') then
    begin
      Result := True;
      Exit;
    end;
  end;
end;

function ReadConfiguredDataDirOverride(const ConfigPath: string): string;
var
  JsonContent, ParsedPath: string;
  Found, IsNull: Boolean;
begin
  Result := '';
  if not FileExists(ConfigPath) then
    Exit;
  if not LoadUtf8TextFile(ConfigPath, JsonContent) then
    Exit;

  ParsedPath := Trim(ExtractJsonStringField(JsonContent, 'data_dir_override', Found, IsNull));
  if Found and (not IsNull) and (Length(ParsedPath) > 3) then
    Result := RemoveBackslashUnlessRoot(ParsedPath);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  UserRoot, DataDir, ConfigDir, ConfigPath, CustomDataDir, CustomModelsDir: string;
  ShouldPurge: Boolean;
begin
  if CurUninstallStep <> usUninstall then
    Exit;

  UserRoot := ExpandConstant('{localappdata}\SecondBrain');
  DataDir := UserRoot + '\data';
  ConfigDir := UserRoot + '\config';
  ConfigPath := ConfigDir + '\desktop_config.json';
  CustomDataDir := ReadConfiguredDataDirOverride(ConfigPath);

  if CustomDataDir <> '' then
  begin
    CustomModelsDir := CustomDataDir + '\models';
    if DirExists(CustomModelsDir) then
      DelTree(CustomModelsDir, True, True, True);
  end;

  ShouldPurge := HasPurgeCliFlag();
  if (not ShouldPurge) and (not UninstallSilent) then
  begin
    ShouldPurge := (
      SuppressibleMsgBox(
        'Do you also want to delete your personal Second Brain documents, databases, and settings in ' + UserRoot + '?',
        mbConfirmation,
        MB_YESNO or MB_DEFBUTTON2,
        IDNO
      ) = IDYES
    );
  end;

  if not ShouldPurge then
    Exit;

  if (CustomDataDir <> '') and DirExists(CustomDataDir) then
    DelTree(CustomDataDir, True, True, True);

  if DirExists(DataDir) then
    DelTree(DataDir, True, True, True);
  if DirExists(ConfigDir) then
    DelTree(ConfigDir, True, True, True);
  if DirExists(UserRoot) then
    DelTree(UserRoot, True, True, True);
end;
