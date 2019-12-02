# Standardizing workflows in imaging transcriptomics with the `abagen` toolbox

```{toctree}
---
hidden: true
includehidden: true
titlesonly: true
---
Home <self>
setting_up.md
accessing_data.md
processing.md
```

## Overview of this walkthrough

This walkthrough is designed to accompany the manuscript "Standardizing workflows in imaging transcriptomics with the `abagen` toolbox" by RD Markello and colleagues.
While we strove to be detailed in our manuscript, we acknowledge that working through code and data files (however well-documented) to reproduce results from other people can be...frustrating.
This walkthrough attempts to lay out the relevant steps required to install + set up your computing environment, how to get access to the (minimal) data required to run the code, and what commands + scripts you need to run to re-generate our results.
Of course, you'll need to refer to the actual code files in our [GitHub repository](https://github.com/netneurolab/markello_transcriptome) for this to all make sense, but hopefully it helps make this whole reproducibility process a bit less garish.

Note that the *vast majority* of the code that was used in the analyses resides in the [`abagen`](https://github.com/rmarkello/abagen) toolbox; this repository really just contains the wrapper code we used to call `abagen`!

## Questions?

If anything in this walkthrough (or in our repository or our manuscript!) is confusing / unclear / whathaveyou, you can reach out by either (1) [opening an issue](https://github.com/netneurolab/markello_transcriptome/issues) on our GitHub repository, or (2) [emailing Ross](mailto:ross.markello@mail.mcgill.ca).
