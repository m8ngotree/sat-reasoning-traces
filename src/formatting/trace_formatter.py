from typing import List, Dict, Any, Tuple
import json
from src.core.sat_solver import SolverTrace, SolverStep, DecisionType
from src.core.sat_generator import ProblemType


class TraceFormatter:
    def __init__(self,
                 ascii_mode: bool = True,
                 max_steps_in_response: int = 200,
                 max_propagations_head: int = 5,
                 max_propagations_tail: int = 5,
                 include_backtracks: bool = True,
                 show_problem_type_name: bool = False,
                 show_domain_constraints: bool = False):
        self.problem_descriptions = {
            ProblemType.RANDOM_3SAT: "random 3-SAT problem",
            ProblemType.PIGEONHOLE: "pigeonhole principle problem", 
            ProblemType.GRAPH_COLORING: "graph coloring problem",
            ProblemType.SCHEDULING: "scheduling problem"
        }
        self.ascii_mode = ascii_mode
        self.max_steps_in_response = max_steps_in_response
        self.max_propagations_head = max_propagations_head
        self.max_propagations_tail = max_propagations_tail
        self.include_backtracks = include_backtracks
        self.show_problem_type_name = show_problem_type_name
        self.show_domain_constraints = show_domain_constraints
    
    def _to_ascii_clause(self, text: str) -> str:
        """Convert pretty symbols (¬, ∨, ∧) into ASCII-friendly tokens."""
        if text is None:
            return ""
        # Replace negations ¬x42 -> NOT x42
        # Do a simple pass: replace '¬x' with 'NOT x'
        text = text.replace("¬x", "NOT x")
        # Replace OR/AND separators
        text = text.replace(" ∨ ", " OR ")
        text = text.replace(" ∧ ", " AND ")
        return text
    
    def format_problem_description(self, trace: SolverTrace) -> str:
        problem_type = trace.instance.problem_type
        metadata = trace.instance.metadata
        
        intro = self.problem_descriptions.get(problem_type, 'SAT problem') if self.show_problem_type_name else 'SAT problem'
        desc = f"This is a {intro} "
        desc += f"with {trace.instance.num_variables} variables and {len(trace.instance.clauses)} clauses.\n"
        
        if self.show_domain_constraints and problem_type == ProblemType.PIGEONHOLE:
            desc += f"It involves {metadata.get('num_pigeons', 'N')} pigeons and {metadata.get('num_holes', 'M')} holes; "
            desc += "each pigeon must occupy exactly one hole, and no hole may hold two pigeons.\n"
        
        elif self.show_domain_constraints and problem_type == ProblemType.GRAPH_COLORING:
            desc += f"Graph with {metadata.get('num_vertices', 'N')} vertices and {metadata.get('num_edges', 'M')} edges, "
            desc += f"using {metadata.get('num_colors', 'K')} colors with adjacency constraints.\n"
        
        elif self.show_domain_constraints and problem_type == ProblemType.SCHEDULING:
            desc += f"Schedule {metadata.get('num_jobs', 'N')} jobs into {metadata.get('num_time_slots', 'T')} time slots "
            desc += f"with {metadata.get('num_conflicts', 'C')} job-conflict pairs.\n"
        
        elif self.show_domain_constraints and problem_type == ProblemType.RANDOM_3SAT:
            ratio = metadata.get('clause_to_variable_ratio', 0)
            desc += f"Clause/variable ratio {ratio:.2f}.\n"
        
        return desc
    
    def _format_step_delta(self, step: SolverStep) -> str:
        """Render a concise delta-only step line using ASCII clauses when needed."""
        if step.decision_type == DecisionType.DECIDE and step.assignment:
            return f"D: decide x{step.assignment.variable}={'true' if step.assignment.value else 'false'} @L{step.decision_level}"
        if step.decision_type == DecisionType.PROPAGATE and step.assignment:
            clause_str = str(step.clause) if step.clause else ""
            if self.ascii_mode:
                clause_str = self._to_ascii_clause(clause_str)
            return f"P: x{step.assignment.variable}={'true' if step.assignment.value else 'false'} (from {clause_str})"
        if step.decision_type == DecisionType.CONFLICT:
            clause_str = str(step.clause) if step.clause else ""
            if self.ascii_mode:
                clause_str = self._to_ascii_clause(clause_str)
            return f"C: conflict on ({clause_str}) @L{step.decision_level}"
        if step.decision_type == DecisionType.BACKTRACK and self.include_backtracks:
            return f"B: backtrack to L{step.decision_level}"
        if step.decision_type == DecisionType.RESTART:
            return "R: restart"
        return ""  # Fallback for steps we choose not to print
    
    def format_reasoning_trace(self, trace: SolverTrace) -> str:
        """Produce a concise, tag-structured reasoning trace suitable for LLM finetuning."""
        # Problem
        problem_desc = self.format_problem_description(trace).strip()
        problem_block = f"<problem>\n{problem_desc}\n</problem>\n\n"

        # Steps (condensed)
        key_steps = self._select_key_steps(trace.steps, self.max_steps_in_response)
        step_lines: List[str] = []
        for step in key_steps:
            line = self._format_step_delta(step)
            if line:
                step_lines.append(line)
        steps_block = "<steps>\n" + "\n".join(step_lines) + "\n</steps>\n\n"

        # Result + optional proof outline
        outline_block = ""
        if trace.final_result is False:
            outline = self._build_unsat_outline(trace.steps)
            if outline:
                outline_block = "<proof_outline>\n" + "\n".join(outline) + "\n</proof_outline>\n\n"

        final_answer = "SAT" if trace.final_result is True else ("UNSAT" if trace.final_result is False else "UNKNOWN")
        final_block = f"<final_answer>{final_answer}</final_answer>"

        return problem_block + steps_block + outline_block + final_block
    
    def _select_key_steps(self, steps: List[SolverStep], max_steps: int = 200) -> List[SolverStep]:
        """Select a compact subset of steps prioritizing decisions, conflicts, and backtracks,
        with sampled propagations (head and tail)."""
        if len(steps) <= max_steps:
            return steps

        decisions = [s for s in steps if s.decision_type == DecisionType.DECIDE]
        conflicts = [s for s in steps if s.decision_type == DecisionType.CONFLICT]
        backtracks = [s for s in steps if s.decision_type == DecisionType.BACKTRACK]
        propagations = [s for s in steps if s.decision_type == DecisionType.PROPAGATE]

        key_steps: List[SolverStep] = []
        key_steps.extend(decisions)
        key_steps.extend(conflicts)
        if self.include_backtracks:
            key_steps.extend(backtracks)

        # Sample propagations
        if propagations:
            head = propagations[: self.max_propagations_head]
            tail = propagations[-self.max_propagations_tail :] if len(propagations) > self.max_propagations_head else []
            key_steps.extend(head)
            key_steps.extend(tail)

        # Always include first and last few raw steps for context
        key_steps.extend(steps[:3])
        key_steps.extend(steps[-3:])

        # De-dup and sort
        seen = set()
        unique_steps: List[SolverStep] = []
        for s in key_steps:
            if s.step_number not in seen:
                seen.add(s.step_number)
                unique_steps.append(s)
        unique_steps.sort(key=lambda s: s.step_number)

        # Ensure hard cap
        if len(unique_steps) > max_steps:
            # Keep all decisions/conflicts/backtracks first, then truncate propagations as needed
            priority = {
                DecisionType.DECIDE: 0,
                DecisionType.CONFLICT: 0,
                DecisionType.BACKTRACK: 1 if self.include_backtracks else 2,
                DecisionType.PROPAGATE: 2,
                DecisionType.RESTART: 3,
            }
            unique_steps.sort(key=lambda s: (priority.get(s.decision_type, 9), s.step_number))
            unique_steps = unique_steps[:max_steps]
            unique_steps.sort(key=lambda s: s.step_number)

        return unique_steps

    def _build_unsat_outline(self, steps: List[SolverStep]) -> List[str]:
        """Construct a brief outline over top-level branches for UNSAT proofs."""
        outline: List[str] = []
        # Find top-level decisions (decision_level == 1)
        top_decisions: List[Tuple[int, int, bool]] = []  # (index, var, value)
        for idx, s in enumerate(steps):
            if s.decision_type == DecisionType.DECIDE and s.decision_level == 1 and s.assignment:
                top_decisions.append((idx, s.assignment.variable, s.assignment.value))

        if not top_decisions:
            return outline

        for i, (start_idx, var, val) in enumerate(top_decisions):
            # Determine end of this top-level branch segment
            end_idx = len(steps)
            for j in range(start_idx + 1, len(steps)):
                sj = steps[j]
                if (sj.decision_type == DecisionType.DECIDE and sj.decision_level == 1) or \
                   (sj.decision_type == DecisionType.BACKTRACK and sj.decision_level == 0):
                    end_idx = j
                    break

            conflicts = sum(1 for s in steps[start_idx:end_idx] if s.decision_type == DecisionType.CONFLICT)
            outline.append(
                f"Branch x{var}={'true' if val else 'false'} ⇒ {conflicts} conflict(s)"
            )

        outline.append("All top-level branches lead to conflict ⇒ UNSAT")
        return outline
    
    def format_trace_for_training(self, trace: SolverTrace) -> Dict[str, Any]:
        """Format trace for machine learning training data with ASCII-friendly clauses and compact reasoning."""
        reasoning_trace = self.format_reasoning_trace(trace)

        if self.ascii_mode:
            clauses_out = [self._to_ascii_clause(str(clause)) for clause in trace.instance.clauses]
            dimacs_out = trace.instance.to_dimacs()  # DIMACS is already ASCII
        else:
            clauses_out = [str(clause) for clause in trace.instance.clauses]
            dimacs_out = trace.instance.to_dimacs()

        training_data = {
            "problem": {
                "type": trace.instance.problem_type.value,
                "num_variables": trace.instance.num_variables,
                "num_clauses": len(trace.instance.clauses),
                "clauses": clauses_out,
                "dimacs": dimacs_out,
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