# Plan checklist

Each `##` heading is an item id. A plan passes an item when it says how it will
handle the requirement. Edit freely; the checker reads this file on every run.

## target-confidence
Keeps only records that ChEMBL assigns directly to the CB2 protein (target confidence score 9).
Why: lower scores mean the measurement may come from a protein family, complex or tissue, not CB2 itself.

## species
States which species' CB2 is studied (normally human) and leaves out the others.
Why: human and rodent CB2 differ enough that their affinities are not interchangeable.

## activity-types
Names the measurement types used (for example Ki, IC50) and whether they are combined.
Why: IC50 depends on assay conditions, so mixing it with Ki adds noise unless done deliberately.

## units
Puts every value on one scale, such as pChEMBL (-log10 of molar).
Why: values arrive in mixed units (nM, uM), and a missed conversion is off by a factor of 1000.

## censored-values
Says how values reported as bounds (">" or "<" rather than "=") are handled.
Why: a bound is not a measurement; treating "> 10 uM" as exactly 10 uM biases the data.

## duplicates
Says how several measurements of the same compound become one value (for example the median).
Why: otherwise a compound is counted more than once and can land in both the training and test sets.

## record-counts
Records how many rows remain after each filtering step.
Why: anyone can then check that the numbers add up from download to final dataset.

## held-out-test
Sets aside a test set before modelling, split by chemical scaffold rather than at random.
Why: a random split puts near-identical molecules on both sides, which overstates accuracy.

## baseline
Compares results with a simple baseline (for example predicting the average value).
Why: a model is only useful if it beats what a trivial guess already achieves.
