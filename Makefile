.PHONY: help setup clean test run lint

# Variables
VENV = poc-document-assesment
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
STREAMLIT = $(VENV)/bin/streamlit
PYTEST = $(VENV)/bin/pytest

help:
	@echo "Available commands:"
	@echo "make setup    - Create virtual environment and install dependencies"
	@echo "make clean    - Remove virtual environment and cached files"
	@echo "make test     - Run tests"
	@echo "make run      - Run Streamlit app"
	@echo "make lint     - Run code linting"

setup:
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

clean:
	rm -rf $(VENV)

test:
	$(PYTEST) -v test-document-assistant.py

run:
	$(STREAMLIT) run streamlit-chat-interface.py

lint:
	$(PYTHON) -m black .
	$(PYTHON) -m flake8 .

install-dev: setup
	$(PIP) install black flake8 pytest-cov

# Default target
all: setup test run