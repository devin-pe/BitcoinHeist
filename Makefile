.PHONY: local_test
local_test:
	coverage run -m pytest -v . && coverage report --rcfile=.coveragerc

.PHONY: format
format:
	ruff format .

.PHONY: build
build:
	docker build . -t nova -f infra/build/Dockerfile