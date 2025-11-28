## Simple Conda environment + Jupyter kernel setup (uses environment.yml)
## Usage:
##   make init                 # create/update env from environment.yml, register kernel
##   make env                  # sync env only
##   make kernel               # (re)register the Jupyter kernel
##   make clean-kernel         # remove the Jupyter kernel
## Variables you can override: ENV_NAME=6156-capstone CONDA=conda KERNEL_NAME=6156-capstone DISPLAY_NAME=6156\ (py3.10)

ENV_NAME ?= 6156-capstone
CONDA ?= conda
KERNEL_NAME ?= 6156-capstone
DISPLAY_NAME ?= 6156 (py3.10)

CONDA_RUN := $(CONDA) run -n $(ENV_NAME)

.PHONY: init env kernel clean-kernel

init: env kernel

env:
	@if $(CONDA) env list | awk '{print $$1}' | grep -qx "$(ENV_NAME)"; then \
		echo "Updating Conda env $(ENV_NAME) from environment.yml ..."; \
		$(CONDA) env update -n $(ENV_NAME) -f environment.yml --prune; \
	else \
		echo "Creating Conda env $(ENV_NAME) from environment.yml ..."; \
		$(CONDA) env create -n $(ENV_NAME) -f environment.yml; \
	fi

kernel:
	@echo "Registering Jupyter kernel: $(KERNEL_NAME) -> '$(DISPLAY_NAME)' using env $(ENV_NAME) ..."
	$(CONDA_RUN) python -m ipykernel install --user --name $(KERNEL_NAME) --display-name "$(DISPLAY_NAME)"
	@echo "Kernel registered. Select '$(DISPLAY_NAME)' in VS Code/Jupyter."

clean-kernel:
	@echo "Removing Jupyter kernel: $(KERNEL_NAME) ..."
	jupyter kernelspec remove -y $(KERNEL_NAME) || true
	@echo "Done."
