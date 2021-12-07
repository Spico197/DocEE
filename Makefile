build: clean
	python3 setup.py sdist bdist_wheel

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -name '*.pyc' -type f -delete
	find . -name '__pycache__' -type d -delete

test:
	python -m unittest discover -v tests
	flake8

debug:
	# python -m unittest -v tests/helper/test_arg_rel.py
	python -m unittest -v tests/test_utils.py
	# python -m unittest -v tests/modules/test_adj_decoding.py
