# More systems run request (2026-04-21)

Requested setup:
- Systems: H, He, Be, Li⁻, Li
- Basis: use `aug-cc-pCVQZ` where available, otherwise `aug-cc-pVQZ`
- Integration grid: `Ne`
- Wavefunction: RHF for all except Li (UHF)
- Densities: HF and CCSD

Runtime settings used for optimization in this session: `N_grid=257`, `N_random=200`, `N_random_last=600`, `N_select=8`.

| System | Density | Basis used | HF/UHF setting | Nel | local_min | spherical_full | reduced | status | time_s | note |
|---|---|---|---|---:|---|---:|---:|---|---:|---|
| H | HF | aug-cc-pVQZ | RHF | 1 | n/a | - | - | ERROR | 5.9 | Nel=1 path errors (`empty array` ambiguity) |
| H | CCSD | aug-cc-pVQZ | RHF | 1 | n/a | - | - | ERROR | 8.2 | Nel=1 path errors (`empty array` ambiguity) |
| He | HF | aug-cc-pVQZ | RHF | 2 | n/a | 1.1441772987503402 | 1.1441772984398997 | OK | 11.4 |  |
| He | CCSD | aug-cc-pVQZ | RHF | 2 | n/a | 1.1406381329870783 | 1.1406381351601642 | OK | 10.9 |  |
| Be | HF | aug-cc-pCVQZ | RHF | - | - | - | - | PENDING | - | long run not completed in-session |
| Be | CCSD | aug-cc-pCVQZ | RHF | - | - | - | - | PENDING | - | long run not completed in-session |
| Li⁻ | HF | aug-cc-pCVQZ | RHF | - | - | - | - | PENDING | - | long run not completed in-session |
| Li⁻ | CCSD | aug-cc-pCVQZ | RHF | - | - | - | - | PENDING | - | long run not completed in-session |
| Li | HF | aug-cc-pCVQZ | UHF | - | - | - | - | PENDING | - | long run not completed in-session |
| Li | CCSD | aug-cc-pCVQZ | UHF | - | - | - | - | PENDING | - | long run not completed in-session |
