.PHONY: install bootstrap-db seed-db docker-up docker-down train test run lint typecheck verify

install:
	python -m pip install -r requirements.txt

bootstrap-db:
	python scripts/bootstrap_database.py

seed-db:
	python scripts/initialize_database.py

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

train:
	python scripts/train_opssat.py

test:
	python -m pytest

run:
	python -m streamlit run app.py

lint:
	ruff check .

typecheck:
	basedpyright --level error

verify:
	ruff check .
	python -m pytest
	basedpyright --level error
