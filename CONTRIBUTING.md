# Contributing

Thanks for your interest! Bug reports, model results, and pull requests are all welcome.

## Getting set up

Follow the [Quick start](README.md#quick-start). You don't need a GPU or API keys to work on most of the code: the
unit tests mock the model and every external API.

```bash
cd backend
pip install -r requirements-ci.txt     # lighter than requirements.txt (CPU torch, no GPU stack)
pytest                                 # unit + task-runtime tests

cd ../frontend
npm install
npx tsc -b && npm test
```

Other suites (opt-in):

```bash
cd backend
pytest tests/test_providers_integration.py -m integration -v          # real API calls; needs keys in .env
pytest tests/contract -m contract --model <id> -v                      # does a model work with the agent?
```

`tests/fakes.py` has a scripted provider that runs the real agent loop deterministically. Use it for agent tests.

## Reporting a new model

Did you try a local model that works well (or doesn't)? Run the contract suite against it and open an issue with the
output and your hardware (GPU, VRAM, tokens/sec if you measured it). These reports help decide which models become
the recommended defaults.

## Pull requests

- Keep changes focused, and add or update tests for behavior changes.
- Make sure `pytest` and `npx tsc -b && npm test` pass. CI runs both.
- For a new provider, see [Adding a provider](docs/architecture.md#adding-a-provider). Most OpenAI-compatible
  servers need only a `models.yaml` entry, not code.
- Never commit secrets: `backend/.env`, `backend/connectors.yaml`, and runtime data (`chats/`, `projects/`, `data/`,
  `memory.txt`, `settings.json`) are git-ignored on purpose.

## Security issues

Please don't open a public issue for vulnerabilities. See [SECURITY.md](SECURITY.md).
