# container

This directory contains files to generate a [Singularity container](https://sylabs.io/docs/) that can be used to re-run analyses without the need to install extra software (beyond, of course, Singularity).

- [**`gen_simg.sh`**](./gen_simg.sh): This is a helper script designed to create the Singularity image.
- [**`Singularity`**](./Singularity): The Singularity recipe generated from `gen_simg.sh` and used to build the Singularity image.

Once you've created the Singularity container you _should_ (famous last words) be able to reproduce all analyses with the following command:

```bash
singularity exec --cleanenv                             \
                 --home ${PWD}                          \
                 container/markello_transcriptome.simg   \
                 /neurodocker/startup.sh                \
                 make all
```

(Note: we don't recommend using `make all` to run the analyses because this will take an _incredibly_ long time with default settings.
Please refer to [our walkthrough](https://netneurolab.github.io/markello_transcriptome) for guidelines and suggestions on reproducing our analyses!)
