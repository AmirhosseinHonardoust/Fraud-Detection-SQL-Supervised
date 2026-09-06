PYTHON ?= python3

.PHONY: install lint format format-check typecheck test gate run clean

install:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

lint:
	ruff check src tests

format:
	black src tests

format-check:
	black --check src tests

typecheck:
	mypy src

test:
	pytest -q

# Same checks CI runs, in one command.
gate: lint format-check typecheck test

run:
	$(PYTHON) src/create_db.py --csv data/transactions_labeled.csv --db fraud.db
	$(PYTHON) src/train_supervised.py --db fraud.db --sql src/queries.sql --outdir outputs

clean:
	rm -rf fraud.db outputs/*.json outputs/*.csv outputs/charts/*.png \
		.pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
