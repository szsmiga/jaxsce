# Requested basis-set run summary (2026-04-21)

## Request
- H⁻: aug-cc-pVQZ
- He: aug-cc-pVQZ
- Li, Li⁻, Be: aug-cc-pCVQZ
- Ne: aug-cc-pCVDZ
- For both HF and CCSD densities.

All runs use `run_optimization.py` with `--grid Ne`.

## Current results

| System | Density | Basis | Nel | local_min | spherical_full | reduced | status | time_s |
|---|---|---|---:|---|---:|---:|---|---:|
| H⁻ | HF | aug-cc-pVQZ | 2 | n/a | 0.38510226852122287 | 0.38510226872252434 | OK | 295.3 |
| H⁻ | CCSD | aug-cc-pVQZ | 2 | n/a | 0.36780089994378307 | 0.36780090013477695 | OK | 277.7 |
| He | HF | aug-cc-pVQZ | 2 | n/a | 1.1451690412941389 | 1.1451690414299567 | OK | 238.5 |
| He | CCSD | aug-cc-pVQZ | 2 | n/a | 1.141620035486994 | 1.1416200374781815 | OK | 240.6 |
| Li | HF | aug-cc-pCVQZ | - | - | - | - | PENDING | - |
| Li | CCSD | aug-cc-pCVQZ | - | - | - | - | PENDING | - |
| Li⁻ | HF | aug-cc-pCVQZ | - | - | - | - | PENDING | - |
| Li⁻ | CCSD | aug-cc-pCVQZ | - | - | - | - | PENDING | - |
| Be | HF | aug-cc-pCVQZ | - | - | - | - | PENDING | - |
| Be | CCSD | aug-cc-pCVQZ | - | - | - | - | PENDING | - |
| Ne | HF | aug-cc-pCVDZ | - | - | - | - | PENDING | - |
| Ne | CCSD | aug-cc-pCVDZ | - | - | - | - | PENDING | - |

## Stored artifacts

Completed rows were archived under:
- `data/requested_2026-04-21/Hneg/aug-cc-pVQZ/{hf,ccsd}/`
- `data/requested_2026-04-21/He/aug-cc-pVQZ/{hf,ccsd}/`

Partial CSV summary:
- `data/requested_2026-04-21/summary_requested_basis_partial.csv`
