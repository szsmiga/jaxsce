# Be summary (Ne grid, prefactor=0.5)

- Basis: aug-cc-pCVQZ
- Wavefunction: RHF
- Densities: HF and CCSD
- Model settings: mode in {spherical_full, reduced}, eig_transform="sqrt", prefactor=0.5

| System | Density | Basis | Wavefunction | Grid | spherical_full | reduced | local_min | status | time_s |
|---|---|---|---|---|---:|---:|---|---|---:|
| Be | HF | aug-cc-pCVQZ | RHF | Ne | 2.8140477668493986 | 2.814048084740837 | True | OK | 530.1 |
| Be | CCSD | aug-cc-pCVQZ | RHF | Ne | 2.8439296068002777 | 2.843929208194216 | False | OK | 754.9 |
