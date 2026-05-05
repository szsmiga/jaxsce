# Summary table (mu_start=4, spherical_full)

- basis: aug-cc-pCVQZ where available (He uses aug-cc-pVQZ)
- grid: Ne
- mode: spherical_full
- eig_transform: sqrt
- prefactor: 0.5
- mu_start: 4
- wavefunction: RHF for He/Be/Li-, UHF for Li

| System | Density | Basis | Wavefunction | Value | Local min | Status | Time (s) |
|---|---|---|---|---:|---|---|---:|
| He | HF | aug-cc-pVQZ | RHF | 0.0 | n/a | OK | 12.4 |
| He | CCSD | aug-cc-pVQZ | RHF | 0.0 | n/a | OK | 9.9 |
| Be | HF | aug-cc-pCVQZ | RHF | 2.50604264313976 | True | OK | 276.8 |
| Be | CCSD | aug-cc-pCVQZ | RHF | 2.5277699772268534 | False | OK | 402.6 |
| Li- | HF | aug-cc-pCVQZ | RHF | 1.4705193893363078 | True | OK | 417.0 |
| Li- | CCSD | aug-cc-pCVQZ | RHF | 1.493486403177182 | False | OK | 359.7 |
| Li | HF | aug-cc-pCVQZ | UHF | 1.1163155206789572 | True | OK | 265.1 |
| Li | CCSD | aug-cc-pCVQZ | UHF | 1.1155856816718974 | True | OK | 397.4 |
