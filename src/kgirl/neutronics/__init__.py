"""Neutronics surrogate on real PWR lattice-physics simulations (NuRea_sim dataset).

    python -m kgirl.neutronics info                 # locate and validate the dataset
    python -m kgirl.neutronics fit --out runs/neutronics

Data (9x25dillon/NuRea_sim, Git LFS; `git lfs pull` in that repo): 24 000 training and 360 held-out
lattice-physics simulations of a PWR fuel assembly, 41 whitespace-separated columns per row:

    k_inf   infinite multiplication factor            (column 0)
    pppf    pin power peaking factor                   (column 1)
    e1..e39 fuel-rod U-235 enrichments, wt%            (columns 2..40, 0.7 to 5.0)

The column meaning is checked against the data, not assumed (dataset.validate): k_inf correlates
0.96 with mean enrichment and PPPF 0.78 with enrichment non-uniformity (max/mean), as reactor physics
says they should.

The surrogate follows Eopiez's refactored/neutronics_surrogate.py (polynomial features + closed-form
ridge, LIMPS payload) with one correction: Eopiez paired row i of raw.csv with row i of test.csv as
(input, target), but both files are samples with the same columns; here the enrichments predict
(k_inf, PPPF), fitted on raw.csv and scored on held-out test.csv.
"""
