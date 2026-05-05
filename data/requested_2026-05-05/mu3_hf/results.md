# HF test with corrected mu-start convention

- systems: He, Be, Li, Ne
- mode: spherical_full and reduced
- eig_transform: sqrt
- prefactor: 0.5
- mu_start: 3
- grid: Ne

| System | Basis | Wavefunction | Nel | spherical_full | reduced | local_min | status | time_s |
|---|---|---|---:|---:|---:|---|---|---:|
| He | aug-cc-pVQZ | RHF | 2 | 0.0 | 0.0 | n/a | OK | 9.2 |
| Be | aug-cc-pCVQZ | RHF | 4 | 2.6467130072414875 | 2.6467130364229288 | False | OK | 260.8 |
| Li | aug-cc-pCVQZ | UHF | 3 | 1.2657614089548588 | 1.2657614139219124 | True | OK | 285.8 |
| Ne | aug-cc-pCVQZ | RHF | 10 | 27.576699962218598 | 27.57670024037361 | True | OK | 382.4 |
