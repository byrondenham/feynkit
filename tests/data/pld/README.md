# Principal Landau determinant database: three entries

The files in this directory are copied unchanged from the database of Fevola, Mizera and Telen,
Principal Landau determinants, Comput. Phys. Commun. 303 (2024) 109278, arXiv:2311.16219,
published on MathRepo at https://mathrepo.mis.mpg.de/PLD/ as `PLD_database.zip`. That content is
licensed under the Creative Commons Attribution 4.0 International licence (CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/).

- `A4_zero_generic.txt`: the massless box
- `par_zero_generic.txt`: the diagram `par` with massless propagators
- `kite_generic_generic.txt`: the massive kite

`tests/test_pld.py` reads U, F, the variables and `f_vector` from each file. Set
`FEYNKIT_PLD_DATA` to the unpacked `database` directory to check all 114 entries.
