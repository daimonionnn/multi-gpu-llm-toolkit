$payload = @{
    prompt = "test"
    n_predict = 10
}
$payloadJson = $payload | ConvertTo-Json -Compress

$client = New-Object System.Net.Http.HttpClient
$client.Timeout = [TimeSpan]::FromSeconds(10)

$t_list = New-Object System.Collections.Generic.List[System.Threading.Tasks.Task]
$sw = [System.Diagnostics.Stopwatch]::StartNew()
for ($k = 1; $k -le 2; $k++) {
    $content = New-Object System.Net.Http.StringContent($payloadJson, [System.Text.Encoding]::UTF8, "application/json")
    $t_list.Add($client.PostAsync("http://127.0.0.1:8080/completion", $content))
}

try { [System.Threading.Tasks.Task]::WaitAll($t_list.ToArray()) } catch {}
$sw.Stop()

foreach ($t in $t_list) {
    if ($t.Status -eq 'RanToCompletion') {
         $json = $t.Result.Content.ReadAsStringAsync().Result
         $obj = $json | ConvertFrom-Json
         Write-Host "Success:" $obj.content
    } else {
         Write-Host "Error:" $t.Exception
    }
}
