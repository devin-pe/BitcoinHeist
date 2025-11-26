.PHONY: test
test:
	@sudo docker compose -f infra/docker-compose.yaml up -d mysql mlflow
	@echo "Waiting for services to be ready..."
	@sleep 10
	@sudo docker compose -f infra/docker-compose.yaml run --rm -v $(PWD):/nova app sh -c "coverage run -m pytest -v . && coverage report --rcfile=.coveragerc"

.PHONY: format
format:
	ruff format .

.PHONY: build
build:
	docker build . -t nova -f infra/build/Dockerfile