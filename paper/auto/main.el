;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "10pt" "twocolumn" "letterpaper")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("wacv" "review" "algorithms") ("hyperref" "pagebackref" "breaklinks" "colorlinks" "allcolors=wacvblue")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "preamble"
    "supplementary"
    "article"
    "art10"
    "wacv"
    "hyperref")
   (TeX-add-symbols
    "wacvPaperID"
    "confName"
    "confYear")
   (LaTeX-add-labels
    "sec:introduction"
    "fig:positive-examples"
    "sec:related-work"
    "sec:method"
    "fig:overview"
    "sec:two-branches"
    "sec:match-naming"
    "eq:match-support"
    "sec:pose-selection"
    "eq:selection"
    "sec:implementation"
    "sec:experiments"
    "sec:datasets-metrics"
    "sec:main-results"
    "tab:main-results"
    "tab:segmenter-control"
    "sec:sampling-ablation"
    "tab:sampling"
    "sec:branch-analysis"
    "tab:branch-complementarity"
    "tab:threshold"
    "sec:cross-dataset"
    "tab:cross-dataset"
    "tab:pot-conditions"
    "sec:naming-analysis"
    "tab:naming-ablation"
    "sec:failure-analysis"
    "fig:ucsb-mismatch"
    "fig:mask-pose"
    "sec:efficiency"
    "sec:limitations"
    "sec:conclusion"
    "sec:future")
   (LaTeX-add-bibliographies
    "sjo-bib"))
 :latex)

