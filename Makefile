.PHONY: test test_cov examples package install lint clean all

test:
	pytest tests/unit/
	polars-profiling -h

test_cov:
	pytest --cov=. tests/unit/
	polars-profiling -h

examples:
	find ./examples -maxdepth 2 -type f -name "*.py" -execdir python {} \;

package:
	rm -rf build dist
	echo "$(version)" > VERSION
	python -m build
	twine check dist/*

install:
	pip install -e ".[test]"

lint:
	pre-commit run --all-files

clean:
	rm -rf build dist *.egg-info src/*.egg-info

all:
	make install
	make lint
	make test
