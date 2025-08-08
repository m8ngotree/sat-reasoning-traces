# SAT Reasoning Dataset Generator

A comprehensive system for building large-scale synthetic datasets of detailed reasoning traces from Boolean satisfiability (SAT) problems to enhance logical reasoning capabilities in large language models.

## 🎯 Overview

This project generates high-quality training data by:
- Creating diverse SAT problem instances (Random 3-SAT, Pigeonhole Principle, Graph Coloring, Scheduling)  
- Solving them with detailed step-by-step traces using the DPLL algorithm
- Converting solver traces into natural language reasoning explanations
- Exporting datasets in multiple ML-friendly formats (HuggingFace, OpenAI, Alpaca, etc.)
- Providing comprehensive validation and quality assurance

## 🚀 Features

### Problem Generation
- **Random 3-SAT**: Configurable clause-to-variable ratios for varying difficulty
- **Pigeonhole Principle**: Guaranteed unsatisfiable instances for proof learning
- **Graph Coloring**: NP-complete problems with real-world relevance  
- **Scheduling**: Resource allocation problems with conflict constraints

### Solver Algorithm
- **DPLL (Davis-Putnam-Logemann-Loveland)**: Classic recursive algorithm with backtracking
- Generates detailed step-by-step execution traces for educational purposes

### Natural Language Generation
- Converts solver steps into human-readable explanations
- Problem descriptions with context and background
- Step-by-step reasoning with mathematical notation
- Final result explanations with proof summaries

### Export Formats
- **HuggingFace**: Instruction-response format for fine-tuning
- **OpenAI**: Chat completion format for GPT fine-tuning
- **Alpaca**: Instruction-following format
- **PyTorch**: Tensors for neural network training
- **CSV/HDF5/XML**: Analysis and storage formats

### Quality Assurance
- Schema validation for data structure integrity
- SAT logic validation for correctness
- Reasoning quality assessment
- Statistical analysis and reporting

## 📦 Installation

```bash
git clone https://github.com/yourusername/sat-rl-environment.git
cd sat-rl-environment
pip install -r requirements.txt
```

## 🔧 Quick Start

### Generate a Sample Instance
```bash
python main.py sample
```

### Generate a Small Dataset  
```bash
python main.py generate --num-instances 100 --output-dir test_dataset
```

### Complete Pipeline
```bash
python main.py pipeline --num-instances 1000 --formats huggingface csv
```

## 📖 Usage Examples

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
        ProblemType.SCHEDULING: 0.1
    },
    solver_types=["DPLL"],
    max_solve_time_seconds=120
)

generator = DatasetGenerator(config)
dataset = generator.generate_dataset_sequential()
generator.save_dataset(dataset, "json")
```

### Validation and Export
```python
from src.dataset.validator import DatasetValidator
from src.dataset.exporter import DatasetExporter

# Validate dataset
validator = DatasetValidator()
result = validator.validate_dataset_file("dataset.json")
print(validator.generate_validation_report(result))

# Export to multiple formats
exporter = DatasetExporter()
exported_files = exporter.export_all_formats(dataset)
```

## 🏗️ Architecture

```
sat-rl-environment/
├── src/
│   ├── core/
│   │   ├── sat_generator.py      # SAT problem instance generation
│   │   └── sat_solver.py         # DPLL solver implementation  
│   ├── dataset/
│   │   ├── generator.py          # Dataset creation
│   │   ├── validator.py          # Quality assurance and validation
│   │   └── exporter.py           # Multi-format export functionality
│   └── formatting/
│       └── trace_formatter.py    # Natural language trace generation
├── main.py                       # Command-line interface
└── requirements.txt             # Dependencies
```

## 📁 Output Structure

Generated datasets are organized in run-specific directories:

```
sat_dataset/
├── run_20250106_143022/
│   ├── sat_dataset_20250106_143022.json
│   └── dataset_info_20250106_143022.json
├── run_20250106_151045/
│   ├── sat_dataset_20250106_151045.json
│   └── dataset_info_20250106_151045.json
└── ...
```

## 🎯 Use Cases

### Large Language Model Training
- Fine-tune models on logical reasoning tasks
- Improve step-by-step problem solving capabilities  
- Learn mathematical proof techniques
- Understand satisfiability and constraint satisfaction

### Research Applications  
- Study SAT solver behavior and performance
- Develop new heuristics and algorithms
- Benchmark logical reasoning systems
- Create educational materials for SAT solving

### Educational Content
- Generate textbook examples and exercises
- Create interactive learning materials
- Provide worked solutions for complex problems
- Demonstrate algorithmic thinking patterns

## 📊 Dataset Statistics

A typical dataset contains:
- **Problem Diversity**: 4 different SAT problem types
- **Size Range**: 10-50 variables, 20-300 clauses per instance
- **Solver Coverage**: DPLL algorithm traces with educational explanations
- **Trace Quality**: 500-5000 character natural language explanations
- **Success Rate**: >95% valid instances after quality filtering

## 🔍 Validation Features

- **Structure Validation**: JSON schema compliance
- **Logic Validation**: SAT problem correctness and solver trace consistency  
- **Quality Assessment**: Natural language explanation completeness
- **Statistical Analysis**: Distribution balance and complexity metrics

## 🌟 Example Output

```
# SAT Solving Reasoning Trace

## Problem Description
This is a random 3-SAT problem with 12 variables and 30 clauses.
Each clause contains exactly 3 literals. The clause-to-variable ratio is 2.50.
This ratio suggests the problem is likely satisfiable.

## Step-by-Step Reasoning

Step 1: We make a decision to set variable x1 = true at decision level 1.
This is a branching point where we explore one possible assignment...

Step 2: Unit propagation forces variable x3 = false. This assignment is 
required because clause 'x3 ∨ ¬x7 ∨ x12' becomes a unit clause...

## Final Result
✅ **SATISFIABLE**: A satisfying assignment was found!
**Satisfying assignment**: {x1=true, x2=false, x3=false, ...}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)  
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- SAT solving algorithm based on the classic DPLL technique
- Inspired by the need for high-quality logical reasoning training data
- Built for the machine learning and automated reasoning communities