from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import copy
import time
from .sat_generator import SATInstance, Clause


class DecisionType(Enum):
    DECIDE = "decide"
    PROPAGATE = "propagate" 
    CONFLICT = "conflict"
    BACKTRACK = "backtrack"
    RESTART = "restart"


@dataclass
class Assignment:
    variable: int
    value: bool
    decision_level: int
    reason: Optional[Clause] = None  # Clause that forced this assignment
    
    def __str__(self):
        sign = "" if self.value else "¬"
        reason_str = f" (reason: {self.reason})" if self.reason else ""
        return f"{sign}x{self.variable}@{self.decision_level}{reason_str}"


@dataclass
class SolverStep:
    step_number: int
    decision_type: DecisionType
    assignment: Optional[Assignment]
    clause: Optional[Clause]
    assignments_before: Dict[int, bool]
    assignments_after: Dict[int, bool]
    decision_level: int
    explanation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "decision_type": self.decision_type.value,
            "assignment": str(self.assignment) if self.assignment else None,
            "clause": str(self.clause) if self.clause else None,
            "assignments_before": dict(self.assignments_before),
            "assignments_after": dict(self.assignments_after),
            "decision_level": self.decision_level,
            "explanation": self.explanation
        }


@dataclass
class SolverTrace:
    instance: SATInstance
    steps: List[SolverStep] = field(default_factory=list)
    final_result: Optional[bool] = None
    final_assignment: Optional[Dict[int, bool]] = None
    
    def add_step(self, step: SolverStep):
        self.steps.append(step)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_type": self.instance.problem_type.value,
            "num_variables": self.instance.num_variables,
            "num_clauses": len(self.instance.clauses),
            "metadata": self.instance.metadata,
            "steps": [step.to_dict() for step in self.steps],
            "final_result": self.final_result,
            "final_assignment": self.final_assignment,
            "total_steps": len(self.steps)
        }



class DPLLSolver:
    def __init__(self, instance: SATInstance, time_limit_seconds: Optional[float] = None):
        self.instance = instance
        self.trace = SolverTrace(instance)
        self.step_counter = 0
        self._time_limit_seconds = time_limit_seconds
        self._start_time: Optional[float] = None

    def _dpll_recursive(self, clauses: List[Clause], assignments: Dict[int, bool], 
                       level: int) -> Tuple[bool, Optional[Dict[int, bool]]]:
        # Preemptive timeout check
        if self._time_limit_seconds is not None and self._start_time is not None:
            if time.time() - self._start_time > self._time_limit_seconds:
                raise TimeoutError("Solver time limit exceeded")
        assignments_before = copy.deepcopy(assignments)

        # Simplify clauses based on current assignments
        simplified_clauses = []
        for clause in clauses:
            satisfied = False
            new_literals = []

            for lit in clause.literals:
                var = abs(lit)
                if var in assignments:
                    expected_value = lit > 0
                    if assignments[var] == expected_value:
                        satisfied = True
                        break
                    # Literal is false, don't add it
                else:
                    new_literals.append(lit)

            if satisfied:
                continue  # Clause is satisfied, ignore it

            if not new_literals:
                # Empty clause - conflict
                step = SolverStep(
                    step_number=self.step_counter,
                    decision_type=DecisionType.CONFLICT,
                    assignment=None,
                    clause=clause,
                    assignments_before=assignments_before,
                    assignments_after=copy.deepcopy(assignments),
                    decision_level=level,
                    explanation=f"Conflict: clause {clause} becomes empty"
                )
                self.trace.add_step(step)
                self.step_counter += 1
                return False, None

            simplified_clauses.append(Clause(new_literals))

        if not simplified_clauses:
            # All clauses satisfied
            return True, assignments

        # Unit propagation (one assignment per step to maintain trace consistency)
        while True:
            # Recompute simplified clauses based on current assignments
            simplified_clauses = []
            unit_candidates = []
            for clause in clauses:
                satisfied = False
                new_literals = []
                for lit in clause.literals:
                    var = abs(lit)
                    if var in assignments:
                        expected_value = lit > 0
                        if assignments[var] == expected_value:
                            satisfied = True
                            break
                        # else literal is false
                    else:
                        new_literals.append(lit)
                if satisfied:
                    continue
                if not new_literals:
                    # Empty clause - conflict
                    step = SolverStep(
                        step_number=self.step_counter,
                        decision_type=DecisionType.CONFLICT,
                        assignment=None,
                        clause=clause,
                        assignments_before=copy.deepcopy(assignments),
                        assignments_after=copy.deepcopy(assignments),
                        decision_level=level,
                        explanation=f"Conflict: clause {clause} becomes empty"
                    )
                    self.trace.add_step(step)
                    self.step_counter += 1
                    return False, None
                simplified = Clause(new_literals)
                simplified_clauses.append(simplified)
                if len(simplified.literals) == 1:
                    unit_candidates.append(simplified)

            if not unit_candidates:
                break

            # Pick the first unit clause whose variable is unassigned
            selected = None
            for uc in unit_candidates:
                lit = uc.literals[0]
                var = abs(lit)
                if var not in assignments:
                    selected = (uc, var, lit > 0)
                    break

            if selected is None:
                # No applicable unit clauses remain
                break

            clause, var, value = selected
            assignments_before = copy.deepcopy(assignments)
            assignments[var] = value
            assignment = Assignment(var, value, level, clause)
            step = SolverStep(
                step_number=self.step_counter,
                decision_type=DecisionType.PROPAGATE,
                assignment=assignment,
                clause=clause,
                assignments_before=assignments_before,
                assignments_after=copy.deepcopy(assignments),
                decision_level=level,
                explanation=f"Unit clause {clause} forces {var} = {value}"
            )
            self.trace.add_step(step)
            self.step_counter += 1

        # Choose a variable to branch on
        unassigned_vars = set(range(1, self.instance.num_variables + 1)) - set(assignments.keys())
        if not unassigned_vars:
            return True, assignments

        var = min(unassigned_vars)  # Simple heuristic

        # Try True first
        assignments_true = copy.deepcopy(assignments)
        assignments_true[var] = True
        assignment_true = Assignment(var, True, level + 1)

        decide_true_before = copy.deepcopy(assignments)
        step = SolverStep(
            step_number=self.step_counter,
            decision_type=DecisionType.DECIDE,
            assignment=assignment_true,
            clause=None,
            assignments_before=decide_true_before,
            assignments_after=copy.deepcopy(assignments_true),
            decision_level=level + 1,
            explanation=f"Decision: try {var} = True at level {level + 1}"
        )
        self.trace.add_step(step)
        self.step_counter += 1

        sat, solution = self._dpll_recursive(clauses, assignments_true, level + 1)
        if sat:
            return True, solution

        # Emit a backtrack step to lower decision level before trying the opposite branch
        prev_after = copy.deepcopy(self.trace.steps[-1].assignments_after) if self.trace.steps else copy.deepcopy(assignments_true)
        backtrack_after = copy.deepcopy(assignments)  # revert to state before branching
        backtrack_step = SolverStep(
            step_number=self.step_counter,
            decision_type=DecisionType.BACKTRACK,
            assignment=None,
            clause=None,
            assignments_before=prev_after,
            assignments_after=backtrack_after,
            decision_level=level,
            explanation=f"Backtrack to level {level} before exploring {var} = False"
        )
        self.trace.add_step(backtrack_step)
        self.step_counter += 1

        # Try False - new decision at the same next level
        assignments_false = copy.deepcopy(assignments)
        assignments_false[var] = False
        assignment_false = Assignment(var, False, level + 1)

        decide_false_before = copy.deepcopy(assignments)
        step = SolverStep(
            step_number=self.step_counter,
            decision_type=DecisionType.DECIDE,
            assignment=assignment_false,
            clause=None,
            assignments_before=decide_false_before,
            assignments_after=copy.deepcopy(assignments_false),
            decision_level=level + 1,
            explanation=f"Backtrack and try {var} = False at level {level + 1}"
        )
        self.trace.add_step(step)
        self.step_counter += 1

        return self._dpll_recursive(clauses, assignments_false, level + 1)

    def solve(self) -> bool:
        self._start_time = time.time()
        sat, solution = self._dpll_recursive(self.instance.clauses, {}, 0)
        self.trace.final_result = sat
        self.trace.final_assignment = solution
        return sat

    def get_trace(self) -> SolverTrace:
        return self.trace