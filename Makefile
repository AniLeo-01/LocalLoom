.PHONY: help deploy deploy-smolvlm deploy-whisper test-smolvlm test-whisper \
       serve-smolvlm serve-whisper stop logs-smolvlm logs-whisper \
       status install-deps test test-services

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

install-deps: ## Install Python (modal, aiohttp) and Node dependencies
	pip install modal aiohttp
	npm install

# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------

deploy: deploy-smolvlm deploy-whisper ## Deploy both Modal services

deploy-smolvlm: ## Deploy the SmolVLM2 visual analysis VM
	modal deploy services/modal_smolvlm.py

deploy-whisper: ## Deploy the Whisper ASR transcription VM
	modal deploy services/modal_whisper_asr.py

# ---------------------------------------------------------------------------
# Run (ephemeral / dev)
# ---------------------------------------------------------------------------

serve-smolvlm: ## Run SmolVLM2 VM ephemerally (modal serve)
	modal serve services/modal_smolvlm.py

serve-whisper: ## Run Whisper ASR VM ephemerally (modal serve)
	modal serve services/modal_whisper_asr.py

# ---------------------------------------------------------------------------
# Test Modal services
# ---------------------------------------------------------------------------

test-smolvlm: ## Run the SmolVLM2 health-check entrypoint
	modal run services/modal_smolvlm.py

test-whisper: ## Run the Whisper ASR health-check entrypoint
	modal run services/modal_whisper_asr.py

# ---------------------------------------------------------------------------
# Local tests
# ---------------------------------------------------------------------------

test: ## Run TypeScript and Python unit tests
	npm test
	python3 -m unittest discover -s tests

test-services: ## Run Python unit tests only
	python3 -m unittest discover -s tests

# ---------------------------------------------------------------------------
# Status & logs
# ---------------------------------------------------------------------------

status: ## Show running Modal apps
	modal app list

logs-smolvlm: ## Tail logs for the SmolVLM2 deployment
	modal app logs openloom-smolvlm

logs-whisper: ## Tail logs for the Whisper ASR deployment
	modal app logs openloom-whisper-asr

stop: ## Stop both Modal deployments
	-modal app stop openloom-smolvlm
	-modal app stop openloom-whisper-asr
