# Ne-grid benchmark progress (2026-04-08)

Requested rows:
- H⁻, He with aug-cc-pV6Z (HF and CCSD)
- Li, Li⁻, Be with aug-cc-pCV5Z (HF and CCSD)
- B with aug-cc-pCVQZ (HF and CCSD)

## Current computed rows

| System | Density | Basis | HF method | Nel | local_min | spherical_full | reduced | status |
|---|---|---|---|---:|---|---:|---:|---|
| H⁻ | HF | aug-cc-pV6Z | RHF | 2 | n/a* | 0.3845978467221964 | 0.3845978466445861 | OK |

\* For Nel=2, the local-minimum diagnostic is not meaningful in the same way as Nel>2 angular optimizations.

## Blocking issues / partial artifacts

- `aug-cc-pCV5Z` is not available for Li and Be in the installed basis-set sources (PySCF/BSE), so these requested rows fail at density construction with `BasisNotFoundError`.
- Partial files (checkpoint/density matrix only) were produced for some pending rows (`He` and `B` CCSD/HF), but no full optimization output table rows are present yet.

## Suggested fallback bases

- For Li, Li⁻, Be: use `aug-cc-pCVQZ` (available) for both HF and CCSD to complete comparable rows.
- For B: requested `aug-cc-pCVQZ` is available.
