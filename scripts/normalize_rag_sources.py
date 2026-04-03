from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent / "data" / "rag_sources"


def get_utlaansstatus(file_name: str, content: str) -> str:
    lower = content.lower()

    if file_name == "utlaan.txt":
        return "Generelle utlånsregler for utstyr i Læringslaben."

    not_for_loan_phrases = (
        "ikke til utlån",
        "kun til bruk på stedet",
        "skal brukes på stedet",
        "brukes på stedet",
        "ikke lånes ut",
    )
    if any(phrase in lower for phrase in not_for_loan_phrases):
        return "Ikke til utlån. Skal brukes på stedet eller etter særskilt avtale."

    loan_phrases = (
        "kan lånes",
        "lånes ut",
        "til utlån",
    )
    if any(phrase in lower for phrase in loan_phrases):
        return (
            "Kan normalt lånes av studenter og ansatte hvis ikke annet er oppgitt. "
            "Se utlaan.txt for generelle regler og spør ansatte om tilgjengelighet og eventuelle krav."
        )

    return (
        "Kan normalt lånes av studenter og ansatte hvis ikke annet er oppgitt. "
        "Se utlaan.txt for generelle regler og spør ansatte om tilgjengelighet, opplæring og eventuelle begrensninger."
    )


def get_utlaansveiledning(file_name: str) -> str:
    if file_name == "utlaan.txt":
        return (
            "Bruk denne filen sammen med utstyrsfilene når noen spør om hvem som kan låne, "
            "krav til opplæring, reservasjon eller tilbakelevering."
        )
    return (
        "Denne filen beskriver utstyret eller ressursen. For generelle regler om utlån, "
        "opplæring, ansvar og tilbakelevering, kombiner informasjonen her med utlaan.txt."
    )


def remove_existing_sections(content: str) -> str:
    patterns = [
        r"\nUtlånsstatus:\n.*?\n\nUtlånsveiledning:\n.*?(?=\n(?:Nøkkelpunkter:|Bruk:|Søkbare nøkkelord:)|\Z)",
        r"\nUtlÃ¥nsstatus:\n.*?\n\nUtlÃ¥nsveiledning:\n.*?(?=\n(?:Nøkkelpunkter:|Bruk:|Søkbare nøkkelord:)|\Z)",
    ]
    cleaned = content
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.S)
    return cleaned.rstrip()


def insert_sections(content: str, status: str, guide: str) -> str:
    section = f"Utlånsstatus:\n{status}\n\nUtlånsveiledning:\n{guide}\n\n"
    match = re.search(r"(Kort beskrivelse:\n.*?\n\n)", content, flags=re.S)
    if match:
        return content[: match.end()] + section + content[match.end() :]
    return content.rstrip() + "\n\n" + section


def normalize_file(path: Path) -> None:
    if path.name == "README.txt":
        return

    content = path.read_text(encoding="utf-8")
    content = content.lstrip("\ufeff")
    content = content.replace("\r\n", "\n")
    cleaned = remove_existing_sections(content)
    status = get_utlaansstatus(path.name, cleaned)
    guide = get_utlaansveiledning(path.name)
    updated = insert_sections(cleaned, status, guide).rstrip() + "\n"
    path.write_text(updated, encoding="utf-8")


def main() -> None:
    for path in sorted(ROOT.glob("*.txt")):
        normalize_file(path)


if __name__ == "__main__":
    main()
