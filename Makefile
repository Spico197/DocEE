all: format clean test docs build
	echo 'finished'

.PHONY: format
format:
	isort --profile black --filter-files .
	black .

.PHONY: build
build: clean
	python3 setup.py sdist bdist_wheel

.PHONY: clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -name '*.pyc' -type f -delete
	find . -name '__pycache__' -type d -delete

.PHONY: test
test:
	coverage run -m pytest -vv --durations=0 .
	coverage report -m
	flake8

.PHONY: debug
debug:
	# python -m unittest -v tests/helper/test_arg_rel.py
	python -m unittest -v tests/test_utils.py
	# python -m unittest -v tests/modules/test_adj_decoding.py

.PHONY: docs
docs:
	cd docs && make clean
	cd docs && sphinx-apidoc -o . ../dee && make html
