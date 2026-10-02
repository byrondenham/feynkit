# Principal Landau determinant database: four entries

The files in this directory are copied unchanged from the database of Fevola, Mizera and Telen,
Principal Landau determinants, Comput. Phys. Commun. 303 (2024) 109278, arXiv:2311.16219,
published on MathRepo at https://mathrepo.mis.mpg.de/PLD/ as `PLD_database.zip`. That content is
licensed under the Creative Commons Attribution 4.0 International licence (CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/).

- `A4_zero_generic.txt`: the massless box
- `A4_zero_zero.txt`: the massless box with on-shell legs
- `par_zero_generic.txt`: the diagram `par` with massless propagators
- `kite_generic_generic.txt`: the massive kite

`tests/test_pld.py` reads U, F, the variables and `f_vector` from each file. Set
`FEYNKIT_PLD_DATA` to the unpacked `database` directory to check all 114 entries.

The headers of sixteen more entries, up to but not including their first component, serve
`tests/test_degeneracy_table.py`: `A4`, `B4`, `par`, `acn`, `env`, `npltrb`, `tdetri`, `debox`,
`tdebox`, `pltrb` and `dbox` with generic masses, `pentb_zero_zero`, and the custom entries
`inner-dbox`, `outer-dbox`, `Bhabha-dbox` and `Bhabha2-dbox`. Their text is otherwise unchanged,
and the licence above applies to it.

The headers of six more, cut in the same way, serve `tests/test_regression_tables.py`:
`Hj-npl-dbox`, `Bhabha-npl-dbox` and `Hj-npl-pentb` with custom kinematics, and `dpent`,
`npl-dpent` and `npl-dpent2` with massless propagators and legs.
