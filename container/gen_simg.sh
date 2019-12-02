#!/usr/bin/env bash
#
# Description:
#
#     This script is used to generate a Singularity container that can be used
#     to run all the analyses reported in our manuscript.
#
#     This script was initially written to be used on a Linux box running
#     Ubuntu 18.04. YMMV if you try it on any other system! At the very least,
#     you can extract the relevant codebits for creating the Singularity
#     container and run that (or use the pre-generated Singularity recipe in
#     the same directory as this script).
#
# Usage:
#
#     $ bash container/gen_simg.sh
#

curr_dir=$PWD && if [ ${curr_dir##*/} = "container" ]; then cd ..; fi
tag=markello_transcriptome

# use neurodocker (<3) to make a Singularity recipe and build the Singularity
# image. this should only happen if the image doesn't already exist
if [ ! -f ./container/${tag}.simg ]; then
  if [ ! -d ./data/raw/abagen-data ]; then
    cmd="python -c \"import abagen; \
         abagen.fetch_microarray('./data/raw/abagen-data', 'all'); \
         abagen.fetch_rnaseq('./data/raw/abagen-data', 'all')\""
    if ! eval ${cmd}; then
        printf "Cannot find data/raw/abagen-data directory. Please 'pip "
        printf "install abagen' to proceed."
        exit 1
    fi
  fi
  singularity --quiet exec docker://repronim/neurodocker:0.7.0                \
    /usr/bin/neurodocker generate singularity                                 \
    --base ubuntu:18.04                                                       \
    --pkg-manager apt                                                         \
    --install git make                                                        \
    --copy ./environment.yml /opt/environment.yml                             \
    --copy ./data/raw/abagen-data /opt/data                                   \
    --copy ./vibecheck /opt/vibecheck                                         \
    --miniconda                                                               \
      create_env=${tag}                                                       \
      yaml_file=/opt/environment.yml                                          \
    --run "bash -c 'source activate ${tag} && pip install /opt/vibecheck'"    \
    --add-to-entrypoint "source activate ${tag}"                              \
    --env ABAGEN_DATA="/opt/data" NUMEXPR_MAX_THREADS=1                       \
          OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1                            \
  > container/Singularity
  sudo singularity build container/${tag}.simg container/Singularity
fi
