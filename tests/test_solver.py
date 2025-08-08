import unittest
from typing import Dict

from src.core.sat_generator import Clause, SATInstance, ProblemType, SATGenerator
from src.core.sat_solver import DPLLSolver, DecisionType


def is_clause_satisfied(clause: Clause, assignment: Dict[int, bool]) -> bool:
    for lit in clause.literals:
        var = abs(lit)
        expected = lit > 0
        if var in assignment and assignment[var] == expected:
            return True
    return False


def is_formula_satisfied(instance: SATInstance, assignment: Dict[int, bool]) -> bool:
    return all(is_clause_satisfied(c, assignment) for c in instance.clauses)


class TestDPLLSolverBasic(unittest.TestCase):
    def test_simple_sat(self):
        # (x1) AND (x2)
        inst = SATInstance(
            num_variables=2,
            clauses=[Clause([1]), Clause([2])],
            problem_type=ProblemType.RANDOM_3SAT,
            metadata={}
        )
        solver = DPLLSolver(inst)
        result = solver.solve()
        trace = solver.get_trace()

        self.assertTrue(result)
        self.assertIsNotNone(trace.final_assignment)
        self.assertTrue(is_formula_satisfied(inst, trace.final_assignment))
        self.assertGreater(len(trace.steps), 0)

    def test_simple_unsat(self):
        # (x1) AND (¬x1)
        inst = SATInstance(
            num_variables=1,
            clauses=[Clause([1]), Clause([-1])],
            problem_type=ProblemType.RANDOM_3SAT,
            metadata={}
        )
        solver = DPLLSolver(inst)
        result = solver.solve()
        trace = solver.get_trace()

        self.assertFalse(result)
        self.assertIsNone(trace.final_assignment)
        self.assertGreater(len(trace.steps), 0)

    def test_pigeonhole_unsat_small(self):
        gen = SATGenerator(seed=0)
        inst = gen.generate_pigeonhole_principle(num_pigeons=3, num_holes=2)
        solver = DPLLSolver(inst)
        result = solver.solve()
        self.assertFalse(result)

    def test_trace_consistency(self):
        # A small SAT instance that triggers both decide and propagate
        # (x1 ∨ x2) AND (¬x1 ∨ x2) AND (x1)
        inst = SATInstance(
            num_variables=2,
            clauses=[Clause([1, 2]), Clause([-1, 2]), Clause([1])],
            problem_type=ProblemType.RANDOM_3SAT,
            metadata={}
        )
        solver = DPLLSolver(inst)
        _ = solver.solve()
        steps = solver.get_trace().steps

        # assignments_before should equal previous assignments_after
        for i in range(1, len(steps)):
            self.assertEqual(steps[i].assignments_before, steps[i-1].assignments_after)

        # Decision levels: DECIDE should increment by 1 vs previous level
        prev_level = 0
        for step in steps:
            if step.decision_type == DecisionType.DECIDE:
                self.assertEqual(step.decision_level, prev_level + 1)
                prev_level = step.decision_level
            elif step.decision_type == DecisionType.BACKTRACK:
                self.assertLess(step.decision_level, prev_level)
                prev_level = step.decision_level


if __name__ == "__main__":
    unittest.main()


