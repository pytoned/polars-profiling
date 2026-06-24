.PHONY: docs examples

docs:
	mkdocs build

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
	python setup.py sdist bdist_wheel
	twine check dist/*

install:
	pip install -e ".[notebook]"

install-docs: install ### Installs regular and docs dependencies
	pip install -e ".[docs]"

publish-docs: examples ### Publishes the documentation
	mkdir docs/examples
	rsync -R examples/*/*.html docs
	mike deploy --push --update-aliases $(version) latest

lint:
	pre-commit run --all-files

clean:
	git rm --cached `git ls-files -i --exclude-from=.gitignore`

all:
	make lint
	make install
	make examples
	make docs
	make test
