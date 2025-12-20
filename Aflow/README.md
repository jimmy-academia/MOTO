# AFlow: Modular CLI-based Agentic Workflow Generation

This repository provides a **modularized and command-line–driven refactoring** of the original **AFlow** system proposed in the following work:

> **AFlow: Automating Agentic Workflow Generation**  
> ICLR 2025 (Oral)  
> Official repository: https://github.com/FoundationAgents/AFlow

The primary goal of this project is to reorganize the original AFlow implementation into a **clean, unified Python entrypoint (`main.py`)** and expose the workflow execution process via a **CLI interface**, making the system easier to use, extend, and integrate into downstream research or applications.

---

## ✨ Key Features

- Unified execution through a single entrypoint: `main.py`
- Modular system design for agentic workflows
- Command-line interface (CLI) for task execution
- Clean separation between schemes, datasets, and model configurations
- Support for multiple LLM backends (e.g., OpenAI, Azure, Ollama, Groq)
- Safer configuration management via external YAML config files

---

## 📁 Project Structure

```text
AFlow/
├── main.py
├── schemes/
│   └── AFlow/
│       ├── config/
│       │   └── config2.example.yaml
│       ├── ...
├── datasets/
├── utils/
└── ...
```

Configuration

Before running the system, you must prepare a configuration file for model settings.

1️⃣ Create the configuration directory

Inside the following path:

``` schemes/AFlow/ ```

Create a folder named:

``` onfig/ ```

2️⃣ Create the configuration file

Inside the config/ folder, create a file named:

``` config2.example.yaml ```

Example content:

```
models:
  "<model_name_1>":
    api_type: "openai"        # openai / azure / ollama / groq / etc.
    base_url: "<your base url>"
    api_key: "<your api key>"
    temperature: 0

  "<model_name_2>":
    api_type: "openai"
    base_url: "<your base url>"
    api_key: "<your api key>"
    temperature: 0
```

Important Notes

- Replace <your api key> with your own API key

- Never commit real API keys to GitHub

- It is strongly recommended to add real config files to .gitignore

Example .gitignore entry:

``` schemes/AFlow/config/*.yaml ```

Usage

Run the system using the unified CLI interface:

```
python main.py \
  --scheme aflow \
  --dataset <dataset_name> \
  --opt_model_name gpt-4o \
  --exec_model_name gpt-4o-mini
```
