# More systems summary (prefactor=0.5, eig_transform=sqrt)

Requested settings:
- basis: `aug-cc-pCVQZ` where available (fallback to `aug-cc-pVQZ`)
- integration grid: `Ne`
- model modes: `spherical_full` and `reduced`
- model options: `eig_transform="sqrt"`, `prefactor=0.5`
- wavefunctions: RHF for H/He/Be/Li⁻, UHF for Li
- densities: HF and CCSD

Run settings used in-session to reduce cost: `N_grid=129` (Nel=2) and `N_grid=65`, `N_random=30`, `N_random_last=100`, `N_select=5` (Nel>2).

| System | Density | Basis | Wavefunction | Nel | spherical_full | reduced | status | time_s | note |
|---|---|---|---|---:|---:|---:|---|---:|---|
| H | HF | aug-cc-pVQZ | RHF | 1 | - | - | ERROR | 3.1 | Nel<2 not supported for this SCE optimization/model path |
| H | CCSD | aug-cc-pVQZ | RHF | 1 | - | - | ERROR | 6.2 | Nel<2 not supported for this SCE optimization/model path |
| He | HF | aug-cc-pVQZ | RHF | 2 | 0.5713030950088676 | 0.5713030958625799 | OK | 8.4 |  |
| He | CCSD | aug-cc-pVQZ | RHF | 2 | 0.5695414518510613 | 0.5695414534032656 | OK | 7.3 |  |
| Be | HF | aug-cc-pCVQZ | RHF | - | - | - | PENDING | - | long run did not finish in-session |
| Be | CCSD | aug-cc-pCVQZ | RHF | - | - | - | PENDING | - | long run did not finish in-session |
| Li⁻ | HF | aug-cc-pCVQZ | RHF | - | - | - | PENDING | - | long run did not finish in-session |
| Li⁻ | CCSD | aug-cc-pCVQZ | RHF | - | - | - | PENDING | - | long run did not finish in-session |
| Li | HF | aug-cc-pCVQZ | UHF | - | - | - | PENDING | - | long run did not finish in-session |
| Li | CCSD | aug-cc-pCVQZ | UHF | - | - | - | PENDING | - | long run did not finish in-session |
