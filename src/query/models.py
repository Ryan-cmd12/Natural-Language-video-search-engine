from dataclasses import (
    dataclass,
    asdict,
    field,
)


@dataclass
class EntitySpec:
    id: str
    concept: str

    attributes: dict[
        str,
        str
    ] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ActionSpec:
    id: str
    verb: str

    actor: str | None = None
    object: str | None = None
    target: str | None = None

    attributes: dict[
        str,
        str
    ] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RelationshipSpec:
    subject: str
    predicate: str
    object: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TemporalConstraint:
    relation: str

    first: str
    second: str

    value: float | None = None
    unit: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CompiledQuery:
    original_query: str

    target: str

    entities: list[
        EntitySpec
    ]

    actions: list[
        ActionSpec
    ]

    relationships: list[
        RelationshipSpec
    ]

    temporal_constraints: list[
        TemporalConstraint
    ]

    def to_dict(self) -> dict:

        return {
            "original_query":
                self.original_query,

            "target":
                self.target,

            "entities": [
                entity.to_dict()
                for entity
                in self.entities
            ],

            "actions": [
                action.to_dict()
                for action
                in self.actions
            ],

            "relationships": [
                relationship.to_dict()
                for relationship
                in self.relationships
            ],

            "temporal_constraints": [
                constraint.to_dict()
                for constraint
                in self.temporal_constraints
            ],
        }