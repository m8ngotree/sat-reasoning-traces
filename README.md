# SAT Reasoning Dataset Generator

A system for generating synthetic datasets of Boolean satisfiability (SAT) problems with detailed reasoning traces for training and evaluating models on logical reasoning.

## Overview

This project produces training data by:
- Creating diverse SAT problem instances (Random 3-SAT, Pigeonhole Principle, Graph Coloring, Scheduling)
- Solving them with detailed step-by-step traces using a DPLL-based solver
- Converting solver traces into compact natural language explanations
- Exporting datasets in multiple formats (HuggingFace, OpenAI, Alpaca, CSV, HDF5, PyTorch)
- Providing validation and basic quality checks

## Features

### Problem Generation
- Random 3-SAT: configurable clause-to-variable ratios
- Pigeonhole Principle: guaranteed UNSAT instances
- Graph Coloring: NP-complete constraint problems
- Scheduling: job/slot allocation with conflicts

### Solver
- DPLL (Davis–Putnam–Logemann–Loveland) with backtracking and unit propagation
- Generates step-by-step execution traces

### Natural Language Formatting
- Tag-structured, concise reasoning traces
- Optional ASCII clause rendering and domain context

### Export Formats
- HuggingFace (instruction/response JSONL)
- OpenAI (chat fine-tuning JSONL)
- Alpaca JSON
- CSV (analysis)
- HDF5 (storage)
- PyTorch tensors (features/labels + metadata)

### Validation
- Schema checks for generated instances
- SAT logic and trace-consistency checks
- Reasoning trace quality heuristics

## Installation

```bash
git clone https://github.com/m8ngotree/sat-rl-environment.git
cd sat-rl-environment
pip install -r requirements.txt
```

## Quick Start

Generate and solve a sample instance:
```bash
python main.py sample
```

Generate a small dataset:
```bash
python main.py generate --num-instances 100 --output-dir sat_dataset
```

Run the end-to-end pipeline (generate, validate, export):
```bash
python main.py pipeline --num-instances 1000 --formats huggingface csv
```

## Usage Examples

### Custom Dataset Generation
```python
from src.dataset.generator import DatasetGenerator, DatasetConfig
from src.core.sat_generator import ProblemType

config = DatasetConfig(
    num_instances=5000,
    max_variables_range=(15, 40),
    problem_type_distribution={
        ProblemType.RANDOM_3SAT: 0.5,
        ProblemType.PIGEONHOLE: 0.2,
        ProblemType.GRAPH_COLORING: 0.2,
        ProblemType.SCHEDULING: 0.1,
    },
    solver_types=["DPLL"],
    max_solve_time_seconds=120,
)

generator = DatasetGenerator(config)
dataset = generator.generate_dataset_sequential()
generator.save_dataset(dataset, "json")
```

### Validation and Export
```python
from src.dataset.validator import DatasetValidator
from src.dataset.exporter import DatasetExporter

validator = DatasetValidator()
result = validator.validate_dataset_file("dataset.json")
print(validator.generate_validation_report(result))

exporter = DatasetExporter()
exported_files = exporter.export_all_formats(dataset)
```

## Repository Structure

```
sat-rl-environment/
├── src/
│   ├── core/
│   │   ├── sat_generator.py      # SAT instance generation
│   │   └── sat_solver.py         # DPLL solver with tracing
│   ├── dataset/
│   │   ├── generator.py          # Dataset creation and saving
│   │   ├── validator.py          # Validation and reporting
│   │   └── exporter.py           # Multi-format export
│   └── formatting/
│       └── trace_formatter.py    # Natural language trace formatting
├── tests/                        # Unit tests
├── main.py                       # CLI entry point
└── requirements.txt              # Dependencies
```

## Output Structure

Datasets are written to run-specific directories under the output directory:

```
sat_dataset/
└── run_YYYYMMDD_HHMMSS/
    ├── sat_dataset_YYYYMMDD_HHMMSS.json
    └── dataset_info_YYYYMMDD_HHMMSS.json
```

## Development

- Python 3.8+
- Run tests: `pytest -q`
- Code style: follow type annotations and keep functions short and readable

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments

- Based on the classic DPLL SAT solving technique
- Designed for research and educational workflows that require explainable reasoning traces