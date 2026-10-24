# Backend Makefile — TSR-CNN-BiLSTM dimensional sentiment service.
# Targets cover local dev, docker build/run, slimming for deploy, and Render deploy.

# ---- Config (override on CLI: `make build IMAGE=foo`) ----
IMAGE       ?= tsr-backend
PORT        ?= 8000
PYTHON      ?= python3.11
VENV        ?= .venv
VENV_PY     := $(VENV)/bin/python
VENV_PIP    := $(VENV)/bin/pip
HOST        ?= 127.0.0.1
API_URL     ?= http://localhost:$(PORT)

.DEFAULT_GOAL := help

# ---- Help ----
.PHONY: help
help: ## Show this help.
	@echo "TSR-CNN-BiLSTM backend — Makefile targets"
	@echo
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo

# ===== Local dev =====

.PHONY: install
install: ## Create local venv and install requirements.
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt

.PHONY: dev
dev: ## Run uvicorn locally with --reload on $(HOST):$(PORT).
	$(VENV)/bin/uvicorn app.main:app --host $(HOST) --port $(PORT) --reload

.PHONY: stanza-en
stanza-en: ## Download Stanza English constituency model.
	$(VENV_PY) -c "import stanza; stanza.download('en')"

.PHONY: stanza-zh
stanza-zh: ## Download Stanza Chinese (Simplified) constituency model.
	$(VENV_PY) -c "import stanza; stanza.download('zh-hans')"

# ===== Deploy prep =====

.PHONY: slim
slim: ## Show training-only files that would be removed for deploy. Pass CONFIRM=1 to delete.
	@echo "Training-only files (will be deleted if CONFIRM=1):"
	@du -sh --apparent-size sgns.merge.word glove.840B.300d.txt models/emobank-reader models/standalone \
	    cache/parsed_trees_*.pkl cache/*standalone* env datasets \
	    CVAT.csv fb_dataset_cleaned.csv stanford_sentiment_assembled.csv \
	    writer_10240.csv reader_10240.csv 2>/dev/null || true
ifeq ($(CONFIRM),1)
	@echo "CONFIRM=1 set, deleting..."
	rm -f sgns.merge.word glove.840B.300d.txt
	rm -rf models/emobank-reader models/standalone
	rm -f cache/parsed_trees_*.pkl cache/*standalone*
	rm -rf env datasets
	rm -f CVAT.csv fb_dataset_cleaned.csv stanford_sentiment_assembled.csv writer_10240.csv reader_10240.csv
	$(MAKE) size
else
	@echo
	@echo "Dry run. Re-run with CONFIRM=1 to actually delete:  make slim CONFIRM=1"
endif

.PHONY: size
size: ## Show top-level dir sizes (verify slim worked).
	@du -sh */ 2>/dev/null | sort -h

.PHONY: clean
clean: ## Remove __pycache__, dist, egg-info.
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf dist build *.egg-info

# ===== Docker =====

.PHONY: build
build: ## Build docker image $(IMAGE).
	docker build -t $(IMAGE) .

.PHONY: run
run: ## Run container locally on port $(PORT).
	docker run --rm -p $(PORT):$(PORT) -e PORT=$(PORT) --name $(IMAGE) $(IMAGE)

.PHONY: run-detached
run-detached: ## Run container in background.
	docker run -d -p $(PORT):$(PORT) -e PORT=$(PORT) --name $(IMAGE) $(IMAGE)

.PHONY: stop
stop: ## Stop and remove running container.
	-docker stop $(IMAGE) 2>/dev/null
	-docker rm $(IMAGE) 2>/dev/null

.PHONY: shell
shell: ## Shell inside a fresh container (for debugging the image).
	docker run -it --rm $(IMAGE) bash

.PHONY: logs
logs: ## Tail logs of running container.
	docker logs -f $(IMAGE)

.PHONY: image-size
image-size: ## Show built image size.
	docker images $(IMAGE) --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}"

# ===== Smoke tests =====

.PHONY: health
health: ## Curl /health endpoint.
	curl -fsS $(API_URL)/health | (command -v jq >/dev/null && jq . || cat)
	@echo

.PHONY: models
models: ## List discovered models (/models endpoint).
	curl -fsS $(API_URL)/models | (command -v jq >/dev/null && jq . || cat)
	@echo

.PHONY: test-predict
test-predict: ## Sample prediction against /predict (emobank_writer + bilstm).
	curl -fsS -X POST $(API_URL)/predict \
	    -H 'Content-Type: application/json' \
	    -d '{"text":"I am incredibly happy with how this turned out.","dataset":"emobank_writer","dimension":"valence","architecture":"bilstm","run_tag":"paper"}' \
	    | (command -v jq >/dev/null && jq . || cat)
	@echo

# ===== Render deploy =====

.PHONY: deploy
deploy: ## Push to GitHub (Render auto-deploys on push).
	git push origin main

.PHONY: render-logs
render-logs: ## Open Render dashboard logs URL (manual: replace <service-id>).
	@echo "Open: https://dashboard.render.com/web/<service-id>/logs"
