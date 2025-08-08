#!/usr/bin/env python3
"""
SAT Reasoning Dataset Generation System

This script generates large-scale synthetic datasets of detailed reasoning traces
from Boolean satisfiability (SAT) problems to enhance logical reasoning capabilities
in large language models.
"""

import argparse
import sys
from pathlib import Path
import logging
from typing import Dict, Any

from src.core.sat_generator import SATGenerator, ProblemType
from src.core.sat_solver import DPLLSolver
from src.formatting.trace_formatter import TraceFormatter
from src.dataset.generator import DatasetGenerator, DatasetConfig
from src.dataset.validator import DatasetValidator
from src.dataset.exporter import DatasetExporter, ExportConfig


def setup_logging(output_dir: Path, verbose: bool = False):
    """Set up logging configuration"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'sat_dataset_generation.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def generate_sample_instance():
    """Generate and solve a single SAT instance for demonstration"""
    print("=" * 60)
    print("SAMPLE SAT INSTANCE GENERATION AND SOLVING")
    print("=" * 60)
    
    # Generate a sample instance
    generator = SATGenerator(seed=42)
    instance = generator.generate_random_3sat(num_variables=8, num_clauses=20)
    
    print(f"Generated {instance.problem_type.value} problem:")
    print(f"- Variables: {instance.num_variables}")
    print(f"- Clauses: {len(instance.clauses)}")
    print(f"- Metadata: {instance.metadata}")
    
    print("\nFirst 5 clauses:")
    for i, clause in enumerate(instance.clauses[:5]):
        print(f"  {i+1}. {clause}")
    
    # Solve with DPLL
    print(f"\nSolving with DPLL solver...")
    solver = DPLLSolver(instance)
    result = solver.solve()
    trace = solver.get_trace()
    
    print(f"Result: {'SATISFIABLE' if result else 'UNSATISFIABLE'}")
    print(f"Steps taken: {len(trace.steps)}")
    print(f"Final assignment: {trace.final_assignment}")
    
    # Format reasoning trace
    formatter = TraceFormatter()
    reasoning_trace = formatter.format_reasoning_trace(trace)
    
    print("\n" + "=" * 60)
    print("FORMATTED REASONING TRACE")
    print("=" * 60)
    print(reasoning_trace[:1000] + "..." if len(reasoning_trace) > 1000 else reasoning_trace)
    
    return trace


def generate_dataset(args):
    """Generate the complete dataset"""
    print("=" * 60)
    print("LARGE-SCALE DATASET GENERATION")
    print("=" * 60)
    
    # Create configuration
    config = DatasetConfig(
        num_instances=args.num_instances,
        max_variables_range=(args.min_vars, args.max_vars),
        problem_type_distribution={
            ProblemType.RANDOM_3SAT: 0.4,
            ProblemType.PIGEONHOLE: 0.25,
            ProblemType.GRAPH_COLORING: 0.2,
            ProblemType.SCHEDULING: 0.15
        },
        solver_types=args.solver_types,
        max_solve_time_seconds=args.timeout,
        include_unsatisfiable=True,
        output_directory=args.output_dir,
        num_processes=args.processes,
        random_seed=args.seed
    )
    
    print(f"Configuration:")
    print(f"- Target instances: {config.num_instances}")
    print(f"- Variable range: {config.max_variables_range}")
    print(f"- Processes: {config.num_processes}")
    print(f"- Output directory: {config.output_directory}")
    print(f"- Random seed: {config.random_seed}")
    
    # Generate dataset
    generator = DatasetGenerator(config)
    dataset = generator.generate_dataset_parallel()
    
    # Save dataset
    saved_files = generator.save_dataset(dataset, "json")
    print(f"\nDataset saved to: {saved_files}")
    
    # Print statistics
    stats = dataset["dataset_info"]
    print(f"\nGeneration Statistics:")
    print(f"- Instances generated: {stats['generation_config']['num_instances_generated']}")
    print(f"- Failed instances: {stats['generation_config']['num_failed']}")
    print(f"- Generation time: {stats['generation_config']['generation_time_seconds']:.2f} seconds")
    print(f"- Problem types: {stats['problem_type_stats']}")
    print(f"- Satisfiability: {stats['satisfiability_stats']}")
    print(f"- Average steps: {stats['complexity_stats']['avg_steps']:.1f}")
    
    return dataset, saved_files


def validate_dataset(dataset_file: Path):
    """Validate a dataset file"""
    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)
    
    validator = DatasetValidator()
    result = validator.validate_dataset_file(dataset_file)
    
    # Generate and print report
    report = validator.generate_validation_report(result)
    print(report)
    
    # Save report
    report_file = dataset_file.parent / f"validation_report_{dataset_file.stem}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"Validation report saved to: {report_file}")
    
    return result


def export_dataset(dataset: Dict[str, Any], args):
    """Export dataset in multiple formats"""
    print("=" * 60)
    print("DATASET EXPORT")
    print("=" * 60)
    
    export_config = ExportConfig(
        output_directory=str(Path(args.output_dir) / "exports"),
        include_reasoning_traces=True,
        include_step_details=args.include_steps,
        max_trace_length=args.max_trace_length
    )
    
    exporter = DatasetExporter(export_config)
    
    if args.export_formats == ["all"]:
        exported_files = exporter.export_all_formats(dataset, "sat_reasoning_dataset")
    else:
        exported_files = {}
        for format_name in args.export_formats:
            if hasattr(exporter, f"export_to_{format_name}"):
                method = getattr(exporter, f"export_to_{format_name}")
                exported_files[format_name] = method(dataset, "sat_reasoning_dataset")
    
    print(f"\nExported formats:")
    for format_name, file_path in exported_files.items():
        print(f"- {format_name}: {file_path}")
    
    return exported_files


def main():
    parser = argparse.ArgumentParser(
        description="Generate large-scale SAT reasoning datasets for LLM training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a small dataset for testing
  python main.py generate --num-instances 100 --output-dir test_dataset
  
  # Generate large dataset with specific parameters
  python main.py generate --num-instances 10000 --min-vars 10 --max-vars 50 \\
                          --processes 8 --timeout 120
  
  # Validate an existing dataset
  python main.py validate --dataset-file dataset.json
  
  # Export dataset in multiple formats
  python main.py export --dataset-file dataset.json --formats huggingface openai
  
  # Complete pipeline: generate, validate, and export
  python main.py pipeline --num-instances 1000 --formats huggingface csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Sample command
    sample_parser = subparsers.add_parser('sample', help='Generate and solve a sample SAT instance')
    
    # Generate command  
    gen_parser = subparsers.add_parser('generate', help='Generate dataset')
    gen_parser.add_argument('--num-instances', type=int, default=1000,
                           help='Number of instances to generate (default: 1000)')
    gen_parser.add_argument('--min-vars', type=int, default=10,
                           help='Minimum number of variables (default: 10)')
    gen_parser.add_argument('--max-vars', type=int, default=50,
                           help='Maximum number of variables (default: 50)')
    gen_parser.add_argument('--timeout', type=int, default=300,
                           help='Solver timeout in seconds (default: 300)')
    gen_parser.add_argument('--processes', type=int, default=4,
                           help='Number of parallel processes (default: 4)')
    gen_parser.add_argument('--seed', type=int, default=42,
                           help='Random seed for reproducibility (default: 42)')
    gen_parser.add_argument('--output-dir', default='sat_dataset',
                           help='Output directory (default: sat_dataset)')
    gen_parser.add_argument('--verbose', action='store_true',
                           help='Enable verbose logging')
    gen_parser.add_argument('--solver-types', nargs='+', default=['DPLL'],
                           choices=['DPLL'], 
                           help='Solver types to use (default: DPLL only)')
    
    # Validate command
    val_parser = subparsers.add_parser('validate', help='Validate dataset')
    val_parser.add_argument('--dataset-file', type=Path, required=True,
                           help='Path to dataset file to validate')
    
    # Export command
    exp_parser = subparsers.add_parser('export', help='Export dataset')
    exp_parser.add_argument('--dataset-file', type=Path, required=True,
                           help='Path to dataset file to export')
    exp_parser.add_argument('--formats', nargs='+', 
                           choices=['huggingface', 'openai', 'alpaca', 'csv', 'pytorch', 'hdf5', 'xml', 'all'],
                           default=['huggingface'],
                           help='Export formats (default: huggingface)')
    exp_parser.add_argument('--include-steps', action='store_true',
                           help='Include detailed step-by-step traces')
    exp_parser.add_argument('--max-trace-length', type=int,
                           help='Maximum length of reasoning traces')
    exp_parser.add_argument('--output-dir', default='sat_dataset',
                           help='Output directory (default: sat_dataset)')
    
    # Pipeline command
    pip_parser = subparsers.add_parser('pipeline', help='Complete pipeline: generate, validate, export')
    pip_parser.add_argument('--num-instances', type=int, default=1000,
                           help='Number of instances to generate (default: 1000)')
    pip_parser.add_argument('--min-vars', type=int, default=10,
                           help='Minimum number of variables (default: 10)')
    pip_parser.add_argument('--max-vars', type=int, default=50,
                           help='Maximum number of variables (default: 50)')
    pip_parser.add_argument('--timeout', type=int, default=300,
                           help='Solver timeout in seconds (default: 300)')
    pip_parser.add_argument('--processes', type=int, default=4,
                           help='Number of parallel processes (default: 4)')
    pip_parser.add_argument('--seed', type=int, default=42,
                           help='Random seed for reproducibility (default: 42)')
    pip_parser.add_argument('--output-dir', default='sat_dataset',
                           help='Output directory (default: sat_dataset)')
    pip_parser.add_argument('--formats', nargs='+', dest='export_formats',
                           choices=['huggingface', 'openai', 'alpaca', 'csv', 'pytorch', 'hdf5', 'xml', 'all'],
                           default=['huggingface', 'csv'],
                           help='Export formats (default: huggingface csv)')
    pip_parser.add_argument('--include-steps', action='store_true',
                           help='Include detailed step-by-step traces')
    pip_parser.add_argument('--max-trace-length', type=int,
                           help='Maximum length of reasoning traces')
    pip_parser.add_argument('--verbose', action='store_true',
                           help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set up output directory and logging
    if hasattr(args, 'output_dir'):
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        setup_logging(output_dir, getattr(args, 'verbose', False))
    
    # Execute commands
    if args.command == 'sample':
        generate_sample_instance()
    
    elif args.command == 'generate':
        dataset, saved_file = generate_dataset(args)
        print(f"\n✅ Dataset generation complete! Saved to: {saved_file}")
    
    elif args.command == 'validate':
        if not args.dataset_file.exists():
            print(f"❌ Dataset file not found: {args.dataset_file}")
            return
        
        result = validate_dataset(args.dataset_file)
        if result.is_valid:
            print(f"\n✅ Dataset validation passed!")
        else:
            print(f"\n❌ Dataset validation failed with {len(result.errors)} errors")
    
    elif args.command == 'export':
        if not args.dataset_file.exists():
            print(f"❌ Dataset file not found: {args.dataset_file}")
            return
        
        import json
        with open(args.dataset_file, 'r') as f:
            dataset = json.load(f)
        
        exported_files = export_dataset(dataset, args)
        print(f"\n✅ Dataset export complete! {len(exported_files)} formats exported")
    
    elif args.command == 'pipeline':
        print("Starting complete SAT reasoning dataset pipeline...")
        
        # Step 1: Generate dataset
        print("\n📊 Step 1: Generating dataset...")
        dataset, dataset_file = generate_dataset(args)
        
        # Step 2: Validate dataset
        print("\n🔍 Step 2: Validating dataset...")
        validation_result = validate_dataset(Path(dataset_file))
        
        if not validation_result.is_valid:
            print("⚠️ Dataset validation failed, but continuing with export...")
        
        # Step 3: Export dataset
        print("\n📤 Step 3: Exporting dataset...")
        exported_files = export_dataset(dataset, args)
        
        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"Dataset: {dataset_file}")
        print(f"Exports: {list(exported_files.values())}")
        print(f"Total instances: {len(dataset['instances'])}")


if __name__ == "__main__":
    main()