# Replace amdhip64_7.dll in System32 with patched version
# Requires running as Administrator
$src = 'C:\development\LlamaServer\amdhip64_7.dll.patched'
$dst = 'C:\Windows\SYSTEM32\amdhip64_7.dll'
$backup = 'C:\development\LlamaServer\amdhip64_7.dll.original'

try {
    # Take ownership
    takeown /F $dst /A 2>&1 | Out-Null
    icacls $dst /grant Administrators:F 2>&1 | Out-Null
    
    # Copy patched file
    Copy-Item $src $dst -Force -ErrorAction Stop
    
    # Verify
    $hash = (Get-FileHash $dst -Algorithm SHA256).Hash
    $expected = 'F17827EE5F86AFEF30D5E642095DA2E58B40A5FF59AC85CE6F9AC1431326C476'
    if ($hash -eq $expected) {
        "SUCCESS: System DLL patched successfully" | Out-File 'C:\development\LlamaServer\patch-system-result.txt'
        "Hash verified: $hash" | Out-File 'C:\development\LlamaServer\patch-system-result.txt' -Append
    } else {
        "WARNING: Hash mismatch. Got: $hash" | Out-File 'C:\development\LlamaServer\patch-system-result.txt'
    }
} catch {
    "FAILED: $($_.Exception.Message)" | Out-File 'C:\development\LlamaServer\patch-system-result.txt'
}
