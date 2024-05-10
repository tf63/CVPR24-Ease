.PHONY: test
test:
	isort .
	autopep8 -r -i .
	pflake8
