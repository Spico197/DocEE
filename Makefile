all: format clean test build
	echo 'finished'

.PHONY: format
format:
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
