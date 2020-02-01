PROJECT_NAME=polytune

# To run 'source' and 'conda' functions in Makefile
# SHELL=/bin/bash
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh; conda init; conda activate

setup: create-env

create-env:
  # Create a local python environment on top of conda.
	conda env create -f environment.yml -n $(PROJECT_NAME)
	$(CONDA_ACTIVATE) $(PROJECT_NAME)

update-env:
  # Update the local python environment.
	conda env update -f environment.yml -n $(PROJECT_NAME)
	$(CONDA_ACTIVATE) $(PROJECT_NAME)

remove-env:
  # Remove the local python environment in conda.
	conda env remove -y -n $(PROJECT_NAME)

env-name:
	echo "$(PROJECT_NAME)"

lint:
	flake8

test:
	pytest -v tests/
