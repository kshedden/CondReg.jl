#=
Fit conditional logistic regression models to two datasets discussed
in the Stata documentation.  The documentation is available at

stata.com/manuals/rclogit.pdf.

The datasets can be downloaded from these links:

http://www.stata-press.com/data/r11/clogitid.dta
http://www.stata-press.com/data/r17/lowbirth2.dta
=#

using StatFiles, ReadStat, CondReg, DataFrames, StatsModels

# Fit a model to the clogitid data
d1 = DataFrame(load("clogitid.dta"))
m1 = clogit(@formula(y ~ 0 + x1 + x2), d1, d1[:, :id])

# Get the value labels for the low birth weight data
d2r = read_dta("lowbirth2.dta")
vk = d2r.val_label_dict

# Fit a mode to the low birth weight data
d2 = DataFrame(load("lowbirth2.dta"))
d2[:, :raceblack] = [x == 2 ? 1 : 0 for x in d2[:, :race]]
d2[:, :raceother] = [x == 3 ? 1 : 0 for x in d2[:, :race]]
m2 = clogit(
    @formula(low ~ 0 + lwt + smoke + ptd + ht + ui + raceblack + raceother),
    d2,
    d2[:, :pairid],
)
