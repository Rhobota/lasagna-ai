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

test :
	hatch test -i python=3.8 -vv

check :
	hatch run +python=3.8 types:check

testall :
	hatch test --all --cover --randomize

checkall :
	hatch run types:check

.PHONY: help test check testall checkall
