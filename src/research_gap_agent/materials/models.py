"""Stable contracts for evidence-grounded materials knowledge extraction."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from research_gap_agent.evidence import (
    EvidenceRef,
    EvidenceResearchResult,
    EvidenceResearchStatus,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class MaterialKnowledgeStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    NO_EVIDENCE = "no_evidence"
    NO_KNOWLEDGE = "no_knowledge"
    FAILED = "failed"


class StructureKind(StrEnum):
    CRYSTAL_STRUCTURE = "crystal_structure"
    PHASE = "phase"
    SPACE_GROUP = "space_group"
    MORPHOLOGY = "morphology"
    MICROSTRUCTURE = "microstructure"
    OTHER = "other"


class MaterialRelationType(StrEnum):
    COMPOSITION_PROPERTY = "composition_property"
    STRUCTURE_PROPERTY = "structure_property"
    PROCESS_STRUCTURE = "process_structure"
    PROCESS_PROPERTY = "process_property"
    MATERIAL_MATERIAL = "material_material"
    OTHER = "other"


class CompositionComponent(_StrictModel):
    """One source-reported constituent and its amount, when available."""

    species: str = Field(min_length=1, max_length=300)
    amount: float | None = None
    unit: str | None = Field(default=None, min_length=1, max_length=100)
    raw_amount: str | None = Field(default=None, min_length=1, max_length=300)

    @model_validator(mode="after")
    def require_amount(self) -> CompositionComponent:
        if self.amount is None and self.raw_amount is None:
            raise ValueError("amount or raw_amount is required")
        return self


class StructureDescriptor(_StrictModel):
    kind: StructureKind
    value: str = Field(min_length=1, max_length=1000)


class MaterialCondition(_StrictModel):
    """A source-reported experimental, synthesis, or simulation condition."""

    name: str = Field(min_length=1, max_length=300)
    value: str = Field(min_length=1, max_length=1000)
    unit: str | None = Field(default=None, min_length=1, max_length=100)


class ReportedValue(_StrictModel):
    """A value preserving source text while optionally exposing a number."""

    raw_value: str = Field(min_length=1, max_length=1000)
    numeric_value: float | None = None
    unit: str | None = Field(default=None, min_length=1, max_length=100)


class _EvidenceBoundProposal(_StrictModel):
    evidence_ids: list[str] = Field(min_length=1, max_length=50)

    @field_validator("evidence_ids")
    @classmethod
    def unique_evidence_ids(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))


class MaterialEntityProposal(_EvidenceBoundProposal):
    name: str = Field(min_length=1, max_length=500)
    formula: str | None = Field(default=None, min_length=1, max_length=300)
    aliases: list[str] = Field(default_factory=list, max_length=30)
    composition: list[CompositionComponent] = Field(
        default_factory=list,
        max_length=100,
    )
    structures: list[StructureDescriptor] = Field(
        default_factory=list,
        max_length=50,
    )

    @field_validator("aliases")
    @classmethod
    def unique_aliases(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))


class PropertyObservationProposal(_EvidenceBoundProposal):
    material_name: str = Field(min_length=1, max_length=500)
    property_name: str = Field(min_length=1, max_length=500)
    value: ReportedValue
    conditions: list[MaterialCondition] = Field(
        default_factory=list,
        max_length=50,
    )
    measurement_method: str | None = Field(
        default=None,
        min_length=1,
        max_length=1000,
    )


class SynthesisProcedureProposal(_EvidenceBoundProposal):
    material_name: str = Field(min_length=1, max_length=500)
    method: str = Field(min_length=1, max_length=1000)
    precursors: list[str] = Field(default_factory=list, max_length=100)
    steps: list[str] = Field(default_factory=list, max_length=100)
    conditions: list[MaterialCondition] = Field(
        default_factory=list,
        max_length=100,
    )


class SimulationMethodProposal(_EvidenceBoundProposal):
    material_name: str | None = Field(default=None, min_length=1, max_length=500)
    method: str = Field(min_length=1, max_length=1000)
    software: str | None = Field(default=None, min_length=1, max_length=500)
    theory_or_model: str | None = Field(
        default=None,
        min_length=1,
        max_length=1000,
    )
    parameters: list[MaterialCondition] = Field(
        default_factory=list,
        max_length=100,
    )
    outputs: list[str] = Field(default_factory=list, max_length=100)


class MaterialRelationProposal(_EvidenceBoundProposal):
    relation_type: MaterialRelationType
    subject: str = Field(min_length=1, max_length=1000)
    predicate: str = Field(min_length=1, max_length=500)
    object: str = Field(min_length=1, max_length=1000)
    conditions: list[MaterialCondition] = Field(
        default_factory=list,
        max_length=50,
    )


class MaterialKnowledgeDraft(_StrictModel):
    """Strict JSON contract returned by the extraction reasoning step."""

    materials: list[MaterialEntityProposal] = Field(
        default_factory=list,
        max_length=100,
    )
    properties: list[PropertyObservationProposal] = Field(
        default_factory=list,
        max_length=100,
    )
    synthesis_procedures: list[SynthesisProcedureProposal] = Field(
        default_factory=list,
        max_length=100,
    )
    simulation_methods: list[SimulationMethodProposal] = Field(
        default_factory=list,
        max_length=100,
    )
    relations: list[MaterialRelationProposal] = Field(
        default_factory=list,
        max_length=100,
    )


class MaterialEntity(MaterialEntityProposal):
    material_id: str = Field(pattern=r"^material-[1-9][0-9]*$")


class PropertyObservation(PropertyObservationProposal):
    property_id: str = Field(pattern=r"^property-[1-9][0-9]*$")


class SynthesisProcedure(SynthesisProcedureProposal):
    synthesis_id: str = Field(pattern=r"^synthesis-[1-9][0-9]*$")


class SimulationMethod(SimulationMethodProposal):
    simulation_id: str = Field(pattern=r"^simulation-[1-9][0-9]*$")


class MaterialRelation(MaterialRelationProposal):
    relation_id: str = Field(pattern=r"^relation-[1-9][0-9]*$")


class MaterialExtractionLimits(_StrictModel):
    max_input_evidence: int = Field(default=20, ge=1, le=100)
    max_records_per_type: int = Field(default=30, ge=1, le=100)


class MaterialKnowledgeExtractionRequest(_StrictModel):
    """Input: a Module 2 result plus bounded extraction limits."""

    evidence_result: EvidenceResearchResult
    limits: MaterialExtractionLimits = Field(
        default_factory=MaterialExtractionLimits
    )


class MaterialExtractionModelTrace(_StrictModel):
    sequence: int = Field(default=1, ge=1)
    stage: Literal["extraction"] = "extraction"
    ok: bool
    error_type: str | None = None
    error_message: str | None = None


class MaterialKnowledgeExtractionResult(_StrictModel):
    status: MaterialKnowledgeStatus
    question: str = Field(min_length=3, max_length=4096)
    model: str = Field(min_length=1)
    evidence_status: EvidenceResearchStatus
    source_evidence: list[EvidenceRef] = Field(default_factory=list)
    materials: list[MaterialEntity] = Field(default_factory=list)
    properties: list[PropertyObservation] = Field(default_factory=list)
    synthesis_procedures: list[SynthesisProcedure] = Field(default_factory=list)
    simulation_methods: list[SimulationMethod] = Field(default_factory=list)
    relations: list[MaterialRelation] = Field(default_factory=list)
    model_traces: list[MaterialExtractionModelTrace] = Field(
        default_factory=list
    )
    warnings: list[str] = Field(default_factory=list)
