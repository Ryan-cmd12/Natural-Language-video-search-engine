from dataclasses import dataclass, field
from typing import Any


'''
Explains execution plan
'''

@dataclass
class PlanStep:
    step_id: str

    operation: str

    description: str

    depends_on: list[str] = field(
        default_factory=list
    )

    params: dict[str, Any] = field(
        default_factory=dict
    )

    output: str | None = None

    def to_dict(self):

        return {
            "step_id": self.step_id,
            "operation": self.operation,
            "description": self.description,
            "depends_on": self.depends_on,
            "params": self.params,
            "output": self.output,
        }


@dataclass
class QueryPlan:

    query: str

    steps: list[PlanStep] = field(
        default_factory=list
    )

    requires_vlm_verification: bool = False

    def to_dict(self):

        return {
            "query": self.query,
            "requires_vlm_verification":
                self.requires_vlm_verification,
            "steps": [
                step.to_dict()
                for step in self.steps
            ],
        }