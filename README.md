# Standardizing workflows in imaging transcriptomics with the `abagen` toolbox

## "What's in this repository?"

This repository contains data, code, and results for the manuscript "Standardizing workflows in imaging transcriptomics with the `abagen` toolbox" by Markello et al. *Biorxiv*, 2021.
We investigate how variability in processing of the Allen Human Brain Atlas impacts analyses relating gene expression to neuroimaging data and highlight how functionality from the [`abagen`](https://github.com/rmarkello/abagen) toolbox can help to standardize these workflows.

We've tried to document the various aspects of this repository with a whole bunch of README files, so feel free to jump around and check things out.

## "Just let me run the things!"

Itching to just run the analyses?
You'll need to make sure you have installed the appropriate software packages, have access to the HCP, and have downloaded the appropriate data files (check out our [walkthrough](https://netneurolab.github.io/markello_transcriptome) for more details!).
Once you've done that, you can get going with the following:

```bash
git clone https://github.com/netneurolab/markello_transcriptome
cd markello_transcriptome
conda env create -f environment.yml
conda activate markello_transcriptome
pip install vibecheck/
make all
```

If you don't want to deal with the hassle of creating a new Python environment you can create a Singularity image run things in there:

```bash
git clone https://github.com/netneurolab/markello_transcriptome
cd markello_transcriptome
bash container/gen_simg.sh
singularity run container/markello_transcriptome.simg make all
```

Note, however, that **we don't recommend re-running our analyses in this manner** as it will take a *very* long time to do so!
Instead, we refer to our [walkthrough](https://netneurolab.github.io/markello_transcriptome) for more information on the optimal way to reproduce our results.

## "I'd like more information."

If you want a step-by-step through all the methods + analyses take a look at our [walkthrough](https://netneurolab.github.io/markello_transcriptome).

## "I have some questions..."

[Open an issue](https://github.com/netneurolab/markello_transcriptome/issues) on this repository and someone will try and get back to you as soon as possible!
