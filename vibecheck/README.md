# vibecheck

This directory is a tiny little Python package that we wrote to be used throughout our analyses.
These codebits aren't terribly generalizable but are re-used at various points throughout analysis and results generation so they exist here rather than in any one single script.

You can check out the docstring of each module or simply read below to get an idea of what they do:

## [vibecheck.abautils](./vibecheck/abautils.py)

Contains functionality for generating versions of the Desikan-Killiany atlas (provided by `abagen`) that omit the subcortex.

## [vibecheck.analysis](./vibecheck/analysis.py)

Contains all the functions for running the three "prototypical" analyses examined in the current study (i.e., correlating CGE and regional distance, computing silhouette scores on gene communities for the GCE matrix, and correlating T1w/T2w and RGE PC1).

## [parspin.plotting](./vibecheck/plotting.py)

Very minimal functionality for saving plots.
