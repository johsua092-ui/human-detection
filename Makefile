.PHONY: install test run calibrate capabilities clean

install:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

test:
	.venv/bin/python -m pytest tests/ -q

run:
	.venv/bin/python main.py

calibrate:
	.venv/bin/python main.py --calibrate

capabilities:
	.venv/bin/python main.py --capabilities

service:
	sudo bash deploy/install.sh $(shell pwd)

clean:
	rm -rf .venv data/*.csv data/*.json data/*.npz models/*.json .pytest_cache
