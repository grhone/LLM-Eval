# LLM Benchmarking Tool

A flexible tool for benchmarking multiple LLM providers against a dataset of questions with ground truth answers.

## Features

- Supports multiple LLM providers (OpenAI, Anthropic, Gemini, Ollama, OpenRouter)
- Uses a separate "judge" model for unbiased evaluation
- Configurable via simple YAML configuration
- Processes datasets in Parquet format
- Outputs detailed results to CSV
- Calculates and displays accuracy metrics

## Prerequisites

- Python 3.8+
- pip package manager

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to specify:
1. The judge model (used for evaluation)
2. The benchmark models to test

Example configuration:
```yaml
judge_model:
  provider: "gemini"
  model_name: "gemini-2.5-pro"
  api_key: "your_api_key"

benchmark_models:
  - provider: "ollama"
    model_name: "gemma_finetuned:latest"
    base_url: "http://localhost:11434"
  
  - provider: "openai"
    model_name: "gpt-3.5-turbo"
    api_key: "your_openai_key"
```

## Data Format

The tool expects a Parquet file (`data/train-00000-of-00001.parquet`) with these columns:
- `question`: The question to ask
- `ground_truth_answer`: The correct answer
- `document_summary` (optional): Context for evaluation
- `chunks` (optional): Text chunks for context

## Usage

Run the benchmark:
```bash
python app.py
```

## Output

The tool generates:
1. `benchmark_results.csv` with columns:
   - model_provider
   - model_name
   - question
   - gold_answer
   - model_answer
   - judge_score (0 or 1)

2. Console output showing accuracy statistics:
```
Model: gemini-2.5-pro
  - Total Questions: 100
  - Correct Answers: 85
  - Accuracy: 85.00%
```

## License

[MIT License]
