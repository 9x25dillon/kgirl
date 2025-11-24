.ONESHELL:
VENV?=.venv
PY?=$(VENV)/bin/python
PIP?=$(VENV)/bin/pip

install:
	python -m venv $(VENV)
	$(PIP) install -U pip
	cd server && $(PIP) install -e .

dev:
	cd server && $(VENV)/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	cd server && $(VENV)/bin/pytest -q

fmt:
	cd server && $(VENV)/bin/black app

lint:
	cd server && $(VENV)/bin/ruff check app