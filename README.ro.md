# Aliniere locală — algoritmul Smith-Waterman
> English version: [README.md](README.md)

Proiect de examen la Bioinformatică (tema 4). Autor: Prodan Constantin-Iustin.

Aplicația aliniază local două secvențe (ADN/ARN sau proteine) cu algoritmul Smith-Waterman:

- **matrice de substituție la alegere**, specifică tipului de secvență
  - nucleotide: `IDENTITY` (potrivire / nepotrivire alese de utilizator), `TRANSITION_TRANSVERSION`, `NUC.4.4`
  - aminoacizi: `BLOSUM45`, `BLOSUM50`, `BLOSUM62`, `BLOSUM80`, `BLOSUM90`, `PAM30`, `PAM70`, `PAM250`
  - orice altă matrice, din fișier în format NCBI
- **penalizare liniară** a gap-urilor (k gap-uri costă `k·d`)
- **penalizare afină** a gap-urilor (k gap-uri costă `d + (k-1)·e`) — facilitatea suplimentară, cu trei matrici (Gotoh)

Este nevoie doar de Python 3.10 sau mai nou. Nu trebuie instalat niciun pachet.

## Rulare în VS Code

1. `File > Open Folder...` și se alege acest folder.
2. Din panoul **Run and Debug** (Ctrl+Shift+D) se alege o configurație și se apasă F5:

| Configurație | Ce face |
|---|---|
| GUI | interfața grafică |
| CLI: interactive mode | cere secvențele și parametrii în terminal |
| CLI: course example (linear gaps) | exemplul din curs, cu matricea afișată |
| CLI: hemoglobin (BLOSUM62, affine gaps) | hemoglobina umană beta față de alfa |
| Demo: all examples from the report | toate exemplele din referat |
| Tests | cele 25 de teste automate |
| Benchmark | timpul de execuție în funcție de lungime |

## Rulare din terminal

Interfața grafică:

```bash
python gui.py
```

Modul interactiv:

```bash
python main.py
```

Exemplul din curs, cu matricea de programare dinamică:

```bash
python main.py --seq1 ATCTGA --seq2 TGCA --gap-open 2 --show-matrix
```

Proteine din fișiere FASTA, cu penalizare afină:

```bash
python main.py --fasta1 examples/hbb_human.fasta --fasta2 examples/hba_human.fasta --matrix BLOSUM62 --gap affine --gap-open 10 --gap-extend 1
```

Lista matricilor și toate opțiunile:

```bash
python main.py --list-matrices
```

```bash
python main.py --help
```

Testele:

```bash
python -m unittest discover -s tests -v
```

## Opțiunile liniei de comandă

| Opțiune | Semnificație |
|---|---|
| `--seq1`, `--seq2` | secvențele, scrise direct |
| `--fasta1`, `--fasta2` | secvențele, din fișiere FASTA |
| `--type auto\|dna\|protein` | tipul secvențelor (implicit: detectat automat) |
| `--matrix NUME` | matricea de substituție |
| `--matrix-file FIȘIER` | matrice de substituție din fișier (format NCBI) |
| `--match`, `--mismatch` | scorurile pentru matricea `IDENTITY` (implicit 1 și -1) |
| `--gap linear\|affine` | modelul de penalizare (implicit liniar) |
| `--gap-open`, `-d` | d: penalizarea unui gap (liniar) sau a deschiderii (afin) |
| `--gap-extend`, `-e` | e: penalizarea extinderii (doar afin) |
| `--show-matrix` | afișează matricile S (și Gb, Ga pentru afin) |
| `--show-substitution` | afișează matricea de substituție |
| `--all` | afișează toate alinierile optime |

Penalizările se dau ca numere pozitive și se scad din scor.

## Structura proiectului

```
main.py                      linia de comandă
gui.py                       interfața grafică (tkinter)
demo.py                      exemplele din referat
benchmark.py                 măsurarea timpului de execuție
smith_waterman/
    core.py                  algoritmul (liniar și afin) și reconstrucția alinierii
    matrices.py              matricile de substituție
    sequences.py             FASTA, curățare și validare
    formatting.py            afișarea rezultatelor
    cli.py                   opțiunile liniei de comandă și modul interactiv
    data/                    matricile BLOSUM, PAM și NUC.4.4 (format NCBI)
examples/                    secvențe de test în format FASTA
tests/                       teste automate
docs/
    Referat_Smith-Waterman.docx / .pdf
    Prezentare_Smith-Waterman.pptx
    Report_Smith-Waterman.docx / .pdf, Presentation_Smith-Waterman.pptx   (versiunile în engleză)
```

## Verificarea corectitudinii

- exemplul din curs (ATCTGA / TGCA, scor 2) și cel din Durbin et al. (HEAGAWGHEE / PAWHEAE, BLOSUM50, scor 28) sunt reproduse exact;
- pe secvențe scurte, scorul este comparat cu cel obținut prin căutare exhaustivă;
- scorurile au fost comparate cu Biopython (`PairwiseAligner`, mod local) pe 3300 de perechi aleatoare, fără nicio diferență.
