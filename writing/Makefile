all: figs
	pdflatex dynamic_model.tex
	bibtex dynamic_model
	pdflatex dynamic_model.tex
	pdflatex dynamic_model.tex
	open dynamic_model.pdf

clean:
	rm *.aux *.bbl *.log *.blg *.out *.toc *.tdo *.synctex.gz

