.PHONY: test
test:
	coverage run -m pytest -v -p no:warnings . && coverage report --rcfile=.coveragerc

.PHONY: docker-test
docker-test: build
	docker compose --file infra/docker-compose.yaml run --rm --no-deps \
  -v $(CURDIR):/app app sh -c \
  "coverage run -m pytest -v -p no:warnings . && coverage report -m --rcfile=.coveragerc"

.PHONY: format
format:
	ruff format . --line-length 120

.PHONY: build
build:
	docker build . -t explain -f infra/build/Dockerfile 

.PHONY: up
up: build
	docker compose --file infra/docker-compose.yaml down
	docker compose --file infra/docker-compose.yaml up 