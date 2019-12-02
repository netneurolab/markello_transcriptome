# Set-up and installation

## Required software

Reproducing these analyses require the following software (links go to installation instructions for each dependency):

- [Git](https://git-scm.com/),
- [Python 3.7+](https://docs.conda.io/en/latest/miniconda.html),

Alternatively, you can opt to just use:

- [Singularity](https://sylabs.io/guides/3.6/user-guide/quick_start.html)

## Getting the repository with `git`

First, you'll need a copy of all the data, code, and whatnot in the repository.
You can make a copy by running the following command:

```bash
git clone https://github.com/netneurolab/markello_transcriptome
cd markello_transcriptome/
```

## Python dependencies

It is recommended that you create a new Python environment to install all the dependencies for the analyses.
If you'd prefer to install all the dependencies in your current Python environment you can do that, but no guarantees that things will work without issue!
(No guarantee things will work without issue even if you use environments and containers, but we're trying!)

### Using `conda` (recommended)

If you are using [`conda`](https://docs.conda.io/en/latest/miniconda.html) you can create a new environment and install all the required Python dependencies with the following command:

```bash
conda env create -f environment.yml
conda activate markello_transcriptome
```

Alternatively you can add all the dependencies to your current environment with:

```bash
conda env update -f environment.yml
```

### Using `pip`

If you are using `pip` you can install all the dependencies into your current environment with:

```bash
pip install -r requirements.txt
```

### Internal libraries

We've written a small internal package for this project that contains some processing code for the analyses in the manuscript.
Once you've created your Python environment (or installed the dependencies as described above), you can install this package from the root of the repository with the following commands:

```bash
pip install vibecheck
```

## Singularity

If you'd prefer to not install all the Python packages and whatnot you can instead opt to use [Singularity](https://sylabs.io/docs/).
