.PHONY: all parameters pipelines analysis results visualization help

PYTHON ?= python

all: parameters pipelines analysis results visualization

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  all                 to run everything"
	@echo "  parameters          to generate processing parameters for the pipelines"
	@echo "  pipelines           to run processing pipelines for all parameter combinations"
	@echo "  analysis            to run statistical analyses on pipeline outputs"
	@echo "  results             to aggregate analysis outputs + generate some results files"
	@echo "  visualization       to generate the visualizations + figures from the analyses"
	@echo "  manuscript          to compile a PDF from the manuscript TeX files"
	@echo "  doc                 to create a Jupyter Book of the documentation / walkthrough"

parameters:
	$(PYTHON) scripts/generate_parameters.py

pipelines: parameters
	$(PYTHON) scripts/batch_pipeline.py --n_jobs 373248 --atlas dk 0
	$(PYTHON) scripts/batch_pipeline.py --n_jobs 373248 --atlas dksurf 0

analysis: pipelines
	$(PYTHON) scripts/batch_analysis.py --n_jobs 373248 --atlas dk 0
	$(PYTHON) scripts/batch_analysis.py --n_jobs 373248 --atlas dksurf 0

results: analysis
	$(PYTHON) scripts/aggregate_analyses.py
	$(PYTHON) scripts/run_literature_pipelines.py
	$(PYTHON) scripts/compute_parameter_impact.py

visualization: results
	$(PYTHON) scripts/visualization.py
	$(PYTHON) scripts/visualization_literature.py

manuscript:
	@echo "Generating PDF with pdflatex + bibtex"
	@cd manuscript && \
	 rm -f main.pdf && \
	 pdflatex --interaction=nonstopmode main > /dev/null && \
	 bibtex main > /dev/null && \
	 pdflatex --interaction=nonstopmode main > /dev/null && \
	 pdflatex --interaction=nonstopmode main > /dev/null && \
	 rm -f main.aux main.bbl main.blg main.log main.out mainNotes.bib main.synctex.gz
	@echo "Check ./manuscript/main.pdf for generated file"

doc:
	@cd walkthrough && make clean html
