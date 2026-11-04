LATEXMK ?= latexmk
MAIN := main.tex
BUILD_DIR := build
COMBINED_JOB := main-combined
SUPPLEMENT := supplementary-standalone.tex

.PHONY: all pdf combined supp clean watch watch_combined watch_supp

all: pdf

pdf:
	$(LATEXMK) -outdir=$(BUILD_DIR) -pdf $(MAIN)

combined:
	$(LATEXMK) -outdir=$(BUILD_DIR) -pdf -jobname=$(COMBINED_JOB) \
		-usepretex='\def\combinedbuild{}' $(MAIN)

supp:
	$(LATEXMK) -outdir=$(BUILD_DIR) -pdf $(SUPPLEMENT)

clean:
	$(LATEXMK) -C -outdir=$(BUILD_DIR) $(MAIN)
	$(LATEXMK) -C -outdir=$(BUILD_DIR) -jobname=$(COMBINED_JOB) $(MAIN)
	$(LATEXMK) -C -outdir=$(BUILD_DIR) $(SUPPLEMENT)

watch:
	$(LATEXMK) -outdir=$(BUILD_DIR) -pdf -pvc $(MAIN)

watch_combined:
	$(LATEXMK) -outdir=$(BUILD_DIR) -pdf -pvc -jobname=$(COMBINED_JOB) \
		-usepretex='\def\combinedbuild{}' $(MAIN)

watch_supp:
	$(LATEXMK) -outdir=$(BUILD_DIR) -pdf -pvc $(SUPPLEMENT)
