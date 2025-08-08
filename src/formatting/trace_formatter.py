from typing import List, Dict, Any
import json
from src.core.sat_solver import SolverTrace, SolverStep, DecisionType
from src.core.sat_generator import ProblemType


class TraceFormatter:
    def __init__(self):
        self.problem_descriptions = {
            ProblemType.RANDOM_3SAT: "random 3-SAT problem",
            ProblemType.PIGEONHOLE: "pigeonhole principle problem", 
            ProblemType.GRAPH_COLORING: "graph coloring problem",
            ProblemType.HAMILTONIAN_PATH: "Hamiltonian path problem",
            ProblemType.SCHEDULING: "scheduling problem"
        }
    
    def format_problem_description(self, trace: SolverTrace) -> str:
        problem_type = trace.instance.problem_type
        metadata = trace.instance.metadata
        
        desc = f"This is a {self.problem_descriptions.get(problem_type, 'SAT problem')} "
        desc += f"with {trace.instance.num_variables} variables and {len(trace.instance.clauses)} clauses.\n"
        
        if problem_type == ProblemType.PIGEONHOLE:
            desc += f"It involves placing {metadata.get('num_pigeons', 'N')} pigeons into "
            desc += f"{metadata.get('num_holes', 'M')} holes, where each pigeon must be in exactly one hole "
            desc += "and no two pigeons can share the same hole. Since there are more pigeons than holes, "
            desc += "this problem is unsatisfiable by the pigeonhole principle.\n"
        
        elif problem_type == ProblemType.GRAPH_COLORING:
            desc += f"It involves coloring a graph with {metadata.get('num_vertices', 'N')} vertices "
            desc += f"and {metadata.get('num_edges', 'M')} edges using {metadata.get('num_colors', 'K')} colors, "
            desc += "such that no two adjacent vertices have the same color.\n"
        
        elif problem_type == ProblemType.SCHEDULING:
            desc += f"It involves scheduling {metadata.get('num_jobs', 'N')} jobs into "
            desc += f"{metadata.get('num_time_slots', 'T')} time slots, where each job must be assigned "
            desc += f"to exactly one slot and {metadata.get('num_conflicts', 'C')} pairs of jobs cannot "
            desc += "be scheduled at the same time due to conflicts.\n"
        
        elif problem_type == ProblemType.RANDOM_3SAT:
            ratio = metadata.get('clause_to_variable_ratio', 0)
            desc += f"Each clause contains exactly 3 literals. The clause-to-variable ratio is {ratio:.2f}. "
            if ratio > 4.2:
                desc += "This ratio suggests the problem is likely in the hard/unsatisfiable region.\n"
            else:
                desc += "This ratio suggests the problem is likely satisfiable.\n"
        
        return desc
    
    def format_step_explanation(self, step: SolverStep, step_index: int) -> str:
        explanation = f"Step {step_index + 1}: "
        
        if step.decision_type == DecisionType.DECIDE:
            var = step.assignment.variable
            value = "true" if step.assignment.value else "false"
            explanation += f"We make a decision to set variable x{var} = {value} at decision level {step.decision_level}. "
            explanation += "This is a branching point where we explore one possible assignment to try to satisfy the formula."
        
        elif step.decision_type == DecisionType.PROPAGATE:
            var = step.assignment.variable
            value = "true" if step.assignment.value else "false"
            explanation += f"Unit propagation forces variable x{var} = {value}. "
            explanation += f"This assignment is required because clause '{step.clause}' becomes a unit clause "
            explanation += f"under the current partial assignment, leaving only one way to satisfy it."
        
        elif step.decision_type == DecisionType.CONFLICT:
            explanation += f"A conflict is detected! Clause '{step.clause}' cannot be satisfied "
            explanation += "under the current assignment. All literals in this clause evaluate to false, "
            explanation += "making it impossible to satisfy the entire formula with the current assignments."
        
        elif step.decision_type == DecisionType.BACKTRACK:
            explanation += f"We backtrack to decision level {step.decision_level}. "
            explanation += "This means we undo recent assignments and try a different path through the search space. "
            explanation += "Backtracking helps us explore alternative assignments when conflicts are encountered."
        
        elif step.decision_type == DecisionType.RESTART:
            explanation += "The solver performs a restart, clearing all assignments and beginning the search anew. "
            explanation += "This helps escape from difficult regions of the search space."
        
        # Add assignment state information
        if step.assignments_after:
            assigned_vars = sorted(step.assignments_after.keys())
            if len(assigned_vars) <= 10:  # Only show if manageable number
                assignments_str = ", ".join([f"x{var}={str(step.assignments_after[var]).lower()}" 
                                           for var in assigned_vars])
                explanation += f" Current assignments: {{{assignments_str}}}."
            else:
                explanation += f" Total variables assigned: {len(assigned_vars)}."
        
        return explanation
    
    def format_reasoning_trace(self, trace: SolverTrace) -> str:
        formatted_trace = "# SAT Solving Reasoning Trace\n\n"
        
        # Problem description
        formatted_trace += "## Problem Description\n"
        formatted_trace += self.format_problem_description(trace)
        formatted_trace += "\n"
        
        # Show the clauses in a readable format
        formatted_trace += "## Clauses to Satisfy\n"
        for i, clause in enumerate(trace.instance.clauses[:10]):  # Show first 10 clauses
            formatted_trace += f"{i+1}. {clause}\n"
        if len(trace.instance.clauses) > 10:
            formatted_trace += f"... and {len(trace.instance.clauses) - 10} more clauses\n"
        formatted_trace += "\n"
        
        # Solving process
        formatted_trace += "## Solving Process\n"
        formatted_trace += "The solver will systematically assign truth values to variables while maintaining "
        formatted_trace += "the satisfiability of all clauses. When conflicts arise, the solver backtracks and "
        formatted_trace += "tries alternative assignments.\n\n"
        
        # Step-by-step reasoning
        formatted_trace += "## Step-by-Step Reasoning\n\n"
        
        key_steps = self._select_key_steps(trace.steps)
        
        for i, step in enumerate(key_steps):
            formatted_trace += self.format_step_explanation(step, i) + "\n\n"
        
        # Final result
        formatted_trace += "## Final Result\n"
        if trace.final_result is True:
            formatted_trace += "**SATISFIABLE**: A satisfying assignment was found!\n\n"
            if trace.final_assignment:
                assigned_vars = sorted(trace.final_assignment.keys())
                if len(assigned_vars) <= 20:
                    assignment_str = ", ".join([f"x{var}={str(trace.final_assignment[var]).lower()}" 
                                              for var in assigned_vars])
                    formatted_trace += f"**Satisfying assignment**: {{{assignment_str}}}\n\n"
                else:
                    formatted_trace += f"**Satisfying assignment found** with {len(assigned_vars)} variables assigned.\n\n"
            
            formatted_trace += "This assignment makes all clauses evaluate to true, proving that "
            formatted_trace += "the Boolean formula is satisfiable.\n"
        
        elif trace.final_result is False:
            formatted_trace += "**UNSATISFIABLE**: No satisfying assignment exists.\n\n"
            formatted_trace += "The solver has exhaustively explored all possible variable assignments "
            formatted_trace += "and determined that no assignment can make all clauses true simultaneously. "
            formatted_trace += "This proves the Boolean formula is unsatisfiable.\n"
        
        else:
            formatted_trace += "**UNKNOWN**: The solving process was incomplete.\n"
        
        # Summary statistics
        formatted_trace += f"\n## Solving Statistics\n"
        formatted_trace += f"- Total steps: {len(trace.steps)}\n"
        formatted_trace += f"- Decision steps: {sum(1 for s in trace.steps if s.decision_type == DecisionType.DECIDE)}\n"
        formatted_trace += f"- Propagation steps: {sum(1 for s in trace.steps if s.decision_type == DecisionType.PROPAGATE)}\n"
        formatted_trace += f"- Conflicts encountered: {sum(1 for s in trace.steps if s.decision_type == DecisionType.CONFLICT)}\n"
        formatted_trace += f"- Backtrack operations: {sum(1 for s in trace.steps if s.decision_type == DecisionType.BACKTRACK)}\n"
        
        return formatted_trace
    
    def _select_key_steps(self, steps: List[SolverStep], max_steps: int = 50) -> List[SolverStep]:
        if len(steps) <= max_steps:
            return steps
        
        key_steps = []
        
        # Always include first few steps
        key_steps.extend(steps[:5])
        
        # Include all conflicts and backtracking
        conflicts_and_backtracks = [s for s in steps if s.decision_type in [DecisionType.CONFLICT, DecisionType.BACKTRACK]]
        key_steps.extend(conflicts_and_backtracks)
        
        # Include some decision points
        decisions = [s for s in steps if s.decision_type == DecisionType.DECIDE]
        if len(decisions) > 10:
            # Sample decisions evenly
            step_size = len(decisions) // 10
            key_steps.extend(decisions[::step_size])
        else:
            key_steps.extend(decisions)
        
        # Include some unit propagations 
        propagations = [s for s in steps if s.decision_type == DecisionType.PROPAGATE]
        if propagations:
            key_steps.extend(propagations[:5])  # First few propagations
            if len(propagations) > 10:
                key_steps.extend(propagations[-5:])  # Last few propagations
        
        # Always include last few steps
        key_steps.extend(steps[-3:])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_steps = []
        for step in key_steps:
            if step.step_number not in seen:
                seen.add(step.step_number)
                unique_steps.append(step)
        
        # Sort by step number
        unique_steps.sort(key=lambda s: s.step_number)
        
        return unique_steps[:max_steps]
    
    def format_trace_for_training(self, trace: SolverTrace) -> Dict[str, Any]:
        """Format trace for machine learning training data"""
        
        reasoning_trace = self.format_reasoning_trace(trace)
        
        # Create a structured format suitable for LLM training
        training_data = {
            "problem": {
                "type": trace.instance.problem_type.value,
                "num_variables": trace.instance.num_variables,
                "num_clauses": len(trace.instance.clauses),
                "clauses": [str(clause) for clause in trace.instance.clauses],
                "dimacs": trace.instance.to_dimacs(),
                "metadata": trace.instance.metadata
            },
            "reasoning_trace": reasoning_trace,
            "solution": {
                "satisfiable": trace.final_result,
                "assignment": trace.final_assignment,
                "steps_taken": len(trace.steps),
                "conflicts_encountered": sum(1 for s in trace.steps if s.decision_type == DecisionType.CONFLICT),
                "decisions_made": sum(1 for s in trace.steps if s.decision_type == DecisionType.DECIDE)
            },
            "step_by_step": [step.to_dict() for step in trace.steps]
        }
        
        return training_data
    
    def format_multiple_traces(self, traces: List[SolverTrace]) -> List[Dict[str, Any]]:
        """Format multiple traces for batch training data generation"""
        return [self.format_trace_for_training(trace) for trace in traces]