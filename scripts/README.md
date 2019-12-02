# scripts

This directory contains the analytic scripts comprising the backbone of our manuscript.
If you read something in the manuscript and have a question about the methodology or implementation, chances are you can find the answer in one of these files.

- [`generate_parameters.py`](./generate_parameters.py): This script will generate a CSV file of all the parameter combinations we want to run using `abagen` and the filename that we will use to store the generated expression data.
  This is useful for running pipelines in batches (see below) to avoid race conditions in writing files.
  Data are saved to `data/derivatives/{dk,dksurf}`.
- [`batch_pipeline.py`](./batch_pipeline.py): Runs the actual abagen pipelines.
  Requires a few command-line arguments in order to be run correctly.
  Each pipeline only takes about 30-60sec to run, but since there are ~750k splitting them up and using an HPC to batch them is strongly encouraged.
  Refer to the walkthrough for more information on how best to do that!
  Data are saved to `data/derivatives/{dk,dksurf}`.
- [`batch_analysis.py`](./batch_analysis.py): Runs the three statistical analyses on the outputs of the abagen pipelines.
  Requires the same command-line arguments as `batch_pipeline.py`; splitting up the analyses on a HPC is strongly encouraged.
  Refer to the walkthrough for more information on how best to do that!
  Data are saved to `data/derivatives/{dk,dksurf}`.
- [`aggregate_analysis.py`](./aggregate_analysis.py): Aggregates the various output CSV files from `batch_analysis.py` (assuming it has been run in batch form) into a single (quite large) CSV file.
  Data are saved to `data/derivatives/{dk,dksurf}` and `data/derivatives/pipelines.csv`.
- [`compute_parameter_impact.py`](./compute_parameter_impact.py): Calculates the impact score of each parameter for each analysis and saves to a CSV file.
  Data are saved to `data/derivatives/impact.csv`.
- [`run_literature_pipelines.py`](./run_literature_pipelines.py): Runs abagen pipeline for the nine processing pipelines reproduced from the literature.
  For obvious reasons we are taking some liberties with these pipelines so they fit into the abagen framework, but all efforts were made to reproduce them faithfully.
  Data are saved to `data/derivatives/literature`
- [`visualization.py`](./visualization.py): Creates visualizations for statistical distributions of all processing pipelines.
  Images are saved to `figures/`.
- [`visualization_literature.py`](./visualization_literature.py): Creates visualizations for outputs of reproduced processing pipelines.
  Images are saved to `figures/`.
