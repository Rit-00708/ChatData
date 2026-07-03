# ChatData — Talk to Your Data 📊

A local LLM-powered data analysis assistant. Upload any CSV/Excel dataset and ask questions in plain English — powered by your own machine, no cloud APIs required.

> **Privacy-first**: Runs entirely on your computer with Ollama. Your data never leaves your machine.
>
> **Cloud demo**: Swap to a free Groq API key for hosted use.

---

## 🚀 Live Demo

[**Launch the Streamlit App**](https://share.streamlit.io) *(deployed on Streamlit Community Cloud — coming soon)*

<!-- After deployment, replace with actual URL and add screenshot/GIF -->
<!-- ![ChatData Demo](docs/demo.gif) -->

---

## 🤔 Why I Built This

Manually writing pandas code to explore every new dataset is slow and frustrating. You know *what* you want to know, but the detour into code every time kills momentum.

ChatData lets you **talk to your data** instead:

1. Upload a CSV/Excel file
2. Get automatic profiling, cleaning suggestions, and EDA charts
3. Ask questions in plain English → pandas code is generated and executed safely
4. Optionally generate an auto-insights summary

All of this runs locally with **zero cost**, **zero cloud dependency**, and **zero data leaving your machine**.

---

## ✨ Features

### 1. Data Ingestion & Auto-Cleaning
- Accept CSV / Excel uploads
- Automatically detect: missing values, duplicate rows, column types, outliers (IQR method), negative counts, high-cardinality columns
- Suggest and optionally apply cleaning steps: fill/drop missing values, fix dtypes, strip whitespace

### 2. Automated EDA (Exploratory Data Analysis)
- Summary statistics (`describe()`, value counts for categoricals)
- Auto-generated charts: correlation heatmap + distributions per numeric column + bar charts for categorical columns
- Interactive Plotly charts or static matplotlib output (configurable)

### 3. Natural-Language Query Engine *(core AI feature)*
- Text input: "What's the average age of players on team X?"
- LLM generates a pandas expression from your question + dataset schema
- **Sandboxed execution**: no `exec()` with full builtins — restricted namespace, blocked dangerous patterns, timeout protection
- Shows both the generated code and the answer (full transparency)
- Automatically renders a chart when the result is plottable

### 4. Auto-Insights
- One-click button sends dataset summary stats to the LLM
- Returns a bullet-point summary of notable patterns, correlations, or anomalies
- Based on actual statistics — not generic text

### Tech Stack
| Layer | Technology | Why |
|-------|-----------|-----|
| Language | Python 3.11+ | Fast data processing ecosystem |
| Data | pandas + numpy | Industry standard for tabular analysis |
| Visualization | Plotly (interactive) / seaborn + matplotlib (static) | Configurable per deployment |
| LLM (local) | Ollama (`qwen2.5:32b`) | Free, private, runs on consumer hardware |
| LLM (cloud) | Groq (OpenAI-compatible API) | Fast free tier for public demos |
| UI | Streamlit | Single-file web app, zero JS/HTML needed |
| Testing | pytest | Focused on the critical sandbox safety layer |
| Deployment | Streamlit Community Cloud | Free auto-deploy on every `git push` |

---

## 🔧 Setup (Local)

### Prerequisites
1. **Install Ollama**: https://ollama.ai — then pull a model:
   ```bash
   ollama pull qwen2.5:32b
   ```
2. **Clone and install**:
   ```bash
   git clone <your-repo-url>
   cd chatdata
   pip install -r requirements.txt
   ```

### Run locally
```bash
streamlit run app.py
```

The app opens in your browser. Upload a dataset and start asking questions. Everything runs on your machine — no API keys needed for local use.

### Configure (optional)
| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen2.5:32b` | Ollama model name |
| `LLM_PROVIDER` | `ollama` | `ollama` / `groq` / `fake` (testing) |
| `GROQ_API_KEY` | *(none)* | Set for public demo deployment |
| `CHART_ENGINE` | `plotly` | `plotly` / `matplotlib` |

---

## 🌐 Deploying the Public Demo (Streamlit Cloud)

1. Push to GitHub (this repo)
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with your GitHub account
3. Click "New app", select the repo and `app.py`
4. In **Secrets** panel, add:
   - `LLM_PROVIDER = groq`
   - `GROQ_API_KEY = <your-free-key-from-console.groq.com>`
5. Deploy! The hosted version uses Groq's free API instead of local Ollama.

> Get a free Groq API key at https://console.groq.com — no credit card required.

---

## 📋 Usage Examples

### Sample Question → Response (using the NBA demo dataset)

**Question**: *"Who are the top 5 scorers?"*
```python
df.nlargest(10, 'PointsPerGame')[['Name', 'PointsPerGame']]
```
| Name | PointsPerGame |
|------|--------------|
| Player A | 32.4 |
| Player B | 30.1 |
| ... | ... |

**Question**: *"What's the average points by team?"*
```python
df.groupby('Team')['PointsPerGame'].mean().sort_values(ascending=False).head(10)
```

**Question**: *"Show me a chart of rebounds distribution"*

→ Generates histogram + KDE plot automatically.

---

## 📁 Project Structure

```
chatdata/
├── app.py                  # Main Streamlit app (UI: upload, EDA, Q&A, insights)
├── app/                    # Application modules
│   ├── __init__.py
│   ├── config.py           # Environment-driven configuration (LLM provider, model, etc.)
│   ├── generator.py        # Demo dataset generator (NBA 2024-25 player stats)
│   ├── analyzer.py         # Auto-profile + quick stats for LLM context
│   ├── cleaner.py          # Detect issues + auto-clean with strategies
│   ├── eda.py              # EDA chart generation (Plotly/matplotlib)
│   ├── llm_client.py       # Swappable LLM client (Ollama/Groq/fake)
│   └── sandbox.py          # Safe code execution engine (critical security layer)
├── tests/
│   └── test_chatdata.py    # 20+ tests focused on safety + core features
├── data/                   # Generated sample datasets (not committed to git)
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT license
├── .gitignore              # Ignore cache, env files, data files
└── README.md               # This file
```

---

## 🧠 What I Learned

1. **Prompt engineering for code generation** — Constraining an LLM to output *only* valid pandas expressions requires careful system prompts and post-processing (removing markdown fences, handling edge cases).

2. **Safe code execution is hard** — Writing a sandbox that blocks `exec`/`open`/imports while still allowing useful pandas operations taught me about Python's security model (`__builtins__`, namespace isolation, timeout patterns).

3. **Building a data app end-to-end** — From dataset generation → profiling → EDA → LLM integration → Streamlit UI → deployment, this project covered the full pipeline of what real data tools do under the hood.

4. **Swappable backends** — Making Ollama and Groq interchangeable via environment variables (same `chat()` interface) meant zero changes to the app logic when switching providers.

5. **The importance of "fake" mode** — Having a fake LLM mode let me develop and test the entire app without needing a running Ollama instance — crucial for CI/testing workflows.

---

## 🚀 Future Improvements

- [ ] SQL support — import from SQLite/PostgreSQL databases
- [ ] Conversation memory — maintain context across multiple Q&A turns
- [ ] Multi-file datasets — join/merge multiple uploads automatically
- [ ] Custom chart builder — let users specify x/y axes for custom plots
- [ ] Export results — download cleaned data or generated charts as PNG
- [ ] Mobile responsiveness improvements for the Streamlit layout

---

## 📄 License

MIT License — see [`LICENSE`](LICENSE) file.
