"""Procedural mathematics task generator with continuous difficulty scaling."""

from __future__ import annotations

import math
import random

from orbit.environments.base import TaskSpec


class MathTaskGenerator:
    """Procedurally generates verified mathematics tasks with parameterizable difficulty [0.0, 1.0]."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def seed(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def generate_task(
        self,
        difficulty: float = 0.5,
        task_id: str | None = None,
    ) -> TaskSpec:
        """Generates a math problem corresponding to the given difficulty level [0.0, 1.0]."""
        diff = max(0.0, min(1.0, float(difficulty)))
        tid = task_id or f"math_{self.rng.randint(100000, 999999)}"

        if diff < 0.25:
            return self._generate_arithmetic_task(tid, diff)
        elif diff < 0.50:
            return self._generate_linear_equation_task(tid, diff)
        elif diff < 0.75:
            return self._generate_quadratic_task(tid, diff)
        else:
            return self._generate_discrete_math_task(tid, diff)

    def _generate_arithmetic_task(
        self, task_id: str, difficulty: float
    ) -> TaskSpec:
        # Scale magnitude of numbers with difficulty [0.0 - 0.25]
        scale = int(10 + (difficulty / 0.25) * 40)
        a = self.rng.randint(2, scale)
        b = self.rng.randint(2, scale)
        c = self.rng.randint(2, scale)
        d = self.rng.randint(1, scale)

        op = self.rng.choice(["mult_sub", "add_mult", "chained"])
        if op == "mult_sub":
            prompt = f"Calculate the exact value of ({a} * {b}) - ({c} + {d})."
            answer = (a * b) - (c + d)
        elif op == "add_mult":
            prompt = f"Calculate the exact value of ({a} + {b}) * ({c} - {d})."
            answer = (a + b) * (c - d)
        else:
            prompt = f"Calculate the exact value of {a} * {b} + {c} * {d}."
            answer = (a * b) + (c * d)

        return TaskSpec(
            task_id=task_id,
            family="math_arithmetic",
            prompt=prompt,
            ground_truth=str(answer),
            difficulty=difficulty,
            metadata={"category": "arithmetic", "answer": answer},
        )

    def _generate_linear_equation_task(
        self, task_id: str, difficulty: float
    ) -> TaskSpec:
        # Construct solvable linear equation with integer root x
        root = self.rng.randint(-20, 20)
        a = self.rng.choice([-5, -4, -3, -2, 2, 3, 4, 5, 6, 7])
        b = self.rng.randint(-30, 30)

        # a*x + b = c -> c = a*root + b
        c = a * root + b

        b_sign = "+" if b >= 0 else "-"
        b_abs = abs(b)
        prompt = (
            f"Solve the linear equation for x: {a}x {b_sign} {b_abs} = {c}. "
            "Provide the final value as \\boxed{x}."
        )

        return TaskSpec(
            task_id=task_id,
            family="math_algebra",
            prompt=prompt,
            ground_truth=str(root),
            difficulty=difficulty,
            metadata={"category": "linear_equation", "root": root},
        )

    def _generate_quadratic_task(
        self, task_id: str, difficulty: float
    ) -> TaskSpec:
        # Construct quadratic equation with integer roots: (x - r1)(x - r2) = x^2 - (r1+r2)x + r1*r2 = 0
        r1 = self.rng.randint(1, 15)
        r2 = self.rng.randint(r1 + 1, 20)  # Distinct positive roots

        b = -(r1 + r2)
        c = r1 * r2

        b_sign = "+" if b >= 0 else "-"
        b_abs = abs(b)
        c_sign = "+" if c >= 0 else "-"
        c_abs = abs(c)

        prompt = (
            f"Find the largest positive root of the quadratic equation: "
            f"x^2 {b_sign} {b_abs}x {c_sign} {c_abs} = 0. "
            "Write your final answer in \\boxed{answer}."
        )

        return TaskSpec(
            task_id=task_id,
            family="math_quadratic",
            prompt=prompt,
            ground_truth=str(r2),
            difficulty=difficulty,
            metadata={"category": "quadratic", "roots": [r1, r2], "largest_root": r2},
        )

    def _generate_discrete_math_task(
        self, task_id: str, difficulty: float
    ) -> TaskSpec:
        choice = self.rng.choice(["mod_exp", "combinations"])

        if choice == "mod_exp":
            base = self.rng.randint(2, 9)
            exp = self.rng.randint(3, 12)
            mod = self.rng.choice([5, 7, 11, 13, 17, 19])
            answer = pow(base, exp, mod)

            prompt = (
                f"Calculate the remainder when {base}^{exp} is divided by {mod} "
                f"({base}^{exp} mod {mod}). Put your final answer in \\boxed{{answer}}."
            )
        else:
            n = self.rng.randint(5, 12)
            k = self.rng.randint(2, min(5, n))
            answer = math.comb(n, k)

            prompt = (
                f"Calculate the number of ways to choose {k} items from a set of {n} items "
                f"(i.e., C({n}, {k})). Put your final answer in \\boxed{{answer}}."
            )

        return TaskSpec(
            task_id=task_id,
            family="math_discrete",
            prompt=prompt,
            ground_truth=str(answer),
            difficulty=difficulty,
            metadata={"category": choice, "answer": answer},
        )
