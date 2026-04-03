$ErrorActionPreference = "Stop"

$root = Join-Path $PSScriptRoot "..\data\rag_sources"

function Get-UtlaanStatus {
    param(
        [string]$FileName,
        [string]$Content
    )

    $lower = $Content.ToLowerInvariant()

    if ($FileName -eq "utlaan.txt") {
        return "Generelle utlånsregler for utstyr i Læringslaben."
    }

    $notForLoanPhrases = @(
        "ikke til utl",
        "kun til bruk p",
        "skal brukes p",
        "brukes p",
        "ikke lånes ut"
    )

    foreach ($phrase in $notForLoanPhrases) {
        if ($lower.Contains($phrase)) {
            return "Ikke til utlån. Skal brukes på stedet eller etter særskilt avtale."
        }
    }

    $loanPhrases = @(
        "kan lånes",
        "lånes ut",
        "til utlån"
    )

    foreach ($phrase in $loanPhrases) {
        if ($lower.Contains($phrase)) {
            return "Kan normalt lånes av studenter og ansatte hvis ikke annet er oppgitt. Se utlaan.txt for generelle regler og spør ansatte om tilgjengelighet og eventuelle krav."
        }
    }

    return "Kan normalt lånes av studenter og ansatte hvis ikke annet er oppgitt. Se utlaan.txt for generelle regler og spør ansatte om tilgjengelighet, opplæring og eventuelle begrensninger."
}

function Get-UtlaanGuide {
    param([string]$FileName)

    if ($FileName -eq "utlaan.txt") {
        return "Bruk denne filen sammen med utstyrsfilene når noen spør om hvem som kan låne, krav til opplæring, reservasjon eller tilbakelevering."
    }

    return "Denne filen beskriver utstyret eller ressursen. For generelle regler om utlån, opplæring, ansvar og tilbakelevering, kombiner informasjonen her med utlaan.txt."
}

Get-ChildItem -Path $root -Filter *.txt -File | ForEach-Object {
    $path = $_.FullName
    $name = $_.Name
    if ($name -eq "README.txt") {
        return
    }

    $content = Get-Content -Path $path -Raw -Encoding UTF8
    $content = [regex]::Replace(
        $content,
        "(?ms)\r?\nUtlånsstatus:\r?\n.*?\r?\n\r?\nUtlånsveiledning:\r?\n.*?(?=(\r?\nNøkkelpunkter:|\r?\nBruk:|\r?\nSøkbare nøkkelord:|$))",
        ""
    ).TrimEnd()

    $status = Get-UtlaanStatus -FileName $name -Content $content
    $guide = Get-UtlaanGuide -FileName $name
    $section = "Utlånsstatus:`r`n$status`r`n`r`nUtlånsveiledning:`r`n$guide`r`n`r`n"

    if ($content -match "(?s)Kort beskrivelse:\r?\n.*?\r?\n\r?\n") {
        $updated = [regex]::Replace(
            $content,
            "(?s)(Kort beskrivelse:\r?\n.*?\r?\n\r?\n)",
            ('$1' + $section),
            1
        )
    } else {
        $updated = $content.TrimEnd() + "`r`n`r`n" + $section
    }

    Set-Content -Path $path -Value $updated -Encoding utf8
}
