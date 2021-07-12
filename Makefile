init:
	pip3 install -r requirements.txt

test:
	python -m unittest discover -s tests/unit -v

dist:
	python setup.py sdist bdist_wheel

clean:
	@echo "Running clean up...."
	@rm -rf build dist *.egg-info
	@find . -type d -name  "__pycache__" -exec rm -r {} +
	@rm -rf */logs
	@rm -rf */.venv
	@rm -rf */model/*/
