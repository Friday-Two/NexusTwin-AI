# Local Ollama + Colab Free training workflow

## Part A — Run a real local LLM (no API budget)

1. Install Ollama for Windows from https://ollama.com/download.
2. Open PowerShell and run:

```powershell
ollama pull llama3.2
ollama run llama3.2
```

Wait until the model responds to a prompt, then exit the interactive session with `/bye`. Ollama should remain running in the background.

3. In the project root, update the local `.env` file (not the tracked example) to:

```dotenv
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.2
LLM_API_KEY=ollama
LLM_TIMEOUT_SECONDS=90
```

The `/v1` setting is retained for compatibility with OpenAI-style clients; NexusTwin's local integration calls Ollama's native `/api/chat` endpoint. No cloud LLM key is used.

4. Restart the backend and sign into the dashboard. In **Talk to your digital twin**, submit a request such as “Find affordable accommodation near the airport under ₹10,000”. The endpoint is `POST /api/v1/assistant/interpret`. It returns structured intent, category, constraints and missing information. If Ollama is offline, the API returns a clear error; it does not fabricate an LLM response.

This first integration interprets requests only. It does not yet replace deterministic negotiation offers; that is a separate, tested integration step.

## Part B — Train the preference ranker in Google Colab Free

1. Open https://colab.research.google.com/ and create a notebook.
2. In Runtime → Change runtime type, select a GPU if one is offered. If not, the script automatically runs on CPU.
3. Upload the repository or the files `scripts/train_preference_ranker.py`, `datasets/ranking/starter_examples.csv`, and `requirements-training.txt`.
4. Run:

```python
!pip install -r requirements-training.txt
!python scripts/train_preference_ranker.py --data datasets/ranking/starter_examples.csv --output-dir artifacts/ranking --epochs 25
```

5. Download `artifacts/ranking/preference_ranker.pt` and `artifacts/ranking/metrics.json` from Colab, or save them to your Drive.

The training script uses a pretrained `all-MiniLM-L6-v2` sentence encoder with a small PyTorch MLP over preference/service embeddings. It splits by `query_id` to reduce leakage and reports accuracy, precision, recall, F1, ROC-AUC and NDCG@5.

**Important data limitation:** the committed starter CSV is synthetic and is only suitable for checking that the training pipeline runs. Do not report its metrics as benchmark performance. Before making a research claim, replace or augment it with a documented real benchmark or consented preference data, preserve the license/source, and evaluate on held-out query groups. A trained checkpoint is not yet connected to the live service-discovery endpoint.
