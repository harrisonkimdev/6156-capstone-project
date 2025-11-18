## Simple environment + Jupyter kernel setup
## Usage:
##   make init                 # create venv, install deps, register kernel
##   make kernel               # (re)register the Jupyter kernel
##   make clean-kernel         # remove the Jupyter kernel
## Variables you can override: VENV=.venv PYTHON=python3.10 KERNEL_NAME=6156-capstone DISPLAY_NAME=6156\ (py3.10)

VENV ?= .venv
PYTHON ?= python3.10
KERNEL_NAME ?= 6156-capstone
DISPLAY_NAME ?= 6156 (py3.10)

PIP := $(VENV)/bin/pip
PY  := $(VENV)/bin/python

.PHONY: init setup deps kernel clean-kernel

init: setup deps kernel

setup:
	@echo "Creating venv with $(PYTHON) at $(VENV) ..."
	$(PYTHON) -m venv $(VENV)
	@echo "Venv ready: $(VENV)"

deps:
	@echo "Upgrading pip/setuptools/wheel ..."
	$(PIP) install --upgrade pip setuptools wheel
	@echo "Installing project requirements ..."
	$(PIP) install -r requirements.txt
	@echo "Installing Jupyter tooling (ipykernel, jupyterlab) ..."
	$(PIP) install ipykernel jupyterlab

kernel:
	@echo "Registering Jupyter kernel: $(KERNEL_NAME) -> '$(DISPLAY_NAME)' ..."
	$(PY) -m ipykernel install --user --name $(KERNEL_NAME) --display-name "$(DISPLAY_NAME)"
	@echo "Kernel registered. Select '$(DISPLAY_NAME)' in VS Code/Jupyter."

clean-kernel:
	@echo "Removing Jupyter kernel: $(KERNEL_NAME) ..."
	jupyter kernelspec remove -y $(KERNEL_NAME) || true
	@echo "Done."
