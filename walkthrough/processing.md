# Running processing pipelines + analyses

We ran ~750k different processing pipelines on the AHBA, so even though each pipeline takes a relatively short amount of time to run we strongly encourage splitting up the work on a HPC.
Here, we provide a few sample batch scripts (written for SLURM) that demonstrate how we split the work up among different jobs.

## Running the processing pipelines

The processing pipelines take the bulk of the time and memory, so we encourage splitting them into a sizeable chunks (which, unfortunately, means a lot of jobs).
You could alternatively run fewer jobs and request more time—totally up to you!
Below we include an example batch script with some estimates (time, memory, number of jobs) that should work for most cases!

Note that you need to specify the `ATLAS` parameter, which means you'll need to submit two jobs: one for 'dk' and one for 'dksurf' to ensure you're running all the different pipelines.

```bash
#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --array=0-1457

# which atlas to run(options: dk, dksurf)
ATLAS=dksurf

# how many analyses should we run in a single process? changing this will
# require modifying the `array` specification above
STEP=256

START=$( echo "${SLURM_ARRAY_TASK_ID} * ${STEP}" | bc )
singularity run --cleanenv -B ${PWD}:/opt/runner --pwd /opt/runner \
                ${PWD}/container/markello_transcriptome.simg \
                python scripts/batch_pipeline.py \
                --n_jobs ${STEP} --atlas ${ATLAS} ${START}
```

## Running the analyses

The statistical analyses are much (much) faster to run than the processing pipelines, so you can group them into fewer jobs and run more analyses per job.
Note that, as before, you will need to submit two separate jobs: one for each atlas ('dk' and 'dksurf').

```bash
#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --array=0-242

# which atlas to run(options: dk, dksurf)
ATLAS=dksurf

# how many analyses should we run in a single process? changing this will
# require modifying the `array` specification above
STEP=1536

START=$( echo "${SLURM_ARRAY_TASK_ID} * ${STEP}" | bc )
singularity run --cleanenv -B ${PWD}:/opt/runner --pwd /opt/runner \
                ${PWD}/container/markello_transcriptome.simg \
                python scripts/batch_analysis.py \
                --n_jobs ${STEP} --atlas ${ATLAS} ${START}
```
