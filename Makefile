help :
	@echo
	@echo 'Possible commands:'
	@echo
	@echo '  make test         # <-- run tests in *one* python environment (uses hatch)'
	@echo '  make check        # <-- check types in *one* python environment (uses hatch)'
	@echo
	@echo '  make testall      # <-- run tests in *all* python versions (uses hatch)'
	@echo '  make checkall     # <-- check types in *all* python versions (uses hatch)'
	@echo
	@echo '  make quarto-serve # <-- run quarto over the docs'
	@echo '  make docs-check   # <-- check types within the docs'
	@echo

test :
	hatch test -i python=3.8 -vv

check :
	hatch run +python=3.8 types:check

testall :
	hatch test --all --cover --randomize

checkall :
	hatch run types:check

QUARTO_PATH := /usr/local/bin/quarto

$(QUARTO_PATH) :
	wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.7.31/quarto-1.7.31-linux-amd64.deb
	sudo dpkg -i quarto-1.7.31-linux-amd64.deb
	rm -f quarto-1.7.31-linux-amd64.deb

quarto-serve : $(QUARTO_PATH)
	rm -rf docs/_site/ docs/.quarto/
	$(QUARTO_PATH) preview docs

docs-check :
	hatch run docs:check

.PHONY: help test check testall checkall quarto-serve docs-check
