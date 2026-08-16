"""Local orchestration and grounding policy for materials extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from research_gap_agent.agent import DeepSeekChatModel
from research_gap_agent.evidence import EvidenceRef, EvidenceResearchStatus

from .deepseek import (
    DeepSeekMaterialKnowledgeModel,
    MaterialKnowledgeReasoningModel,
)
from .models import (
    MaterialEntity,
    MaterialExtractionModelTrace,
    MaterialKnowledgeExtractionRequest,
    MaterialKnowledgeExtractionResult,
    MaterialKnowledgeStatus,
    MaterialRelation,
    PropertyObservation,
    SimulationMethod,
    SynthesisProcedure,
)

ProposalT = TypeVar("ProposalT", bound=BaseModel)
RecordT = TypeVar("RecordT", bound=BaseModel)


class MaterialKnowledgeExtractor:
    """Extract domain records and reject ungrounded LLM proposals locally."""

    def __init__(self, reasoning_model: MaterialKnowledgeReasoningModel) -> None:
        self._reasoning_model = reasoning_model

    def extract(
        self,
        request: MaterialKnowledgeExtractionRequest | dict[str, Any],
    ) -> MaterialKnowledgeExtractionResult:
        validated = MaterialKnowledgeExtractionRequest.model_validate(request)
        evidence_result = validated.evidence_result
        source_evidence = evidence_result.evidence[
            : validated.limits.max_input_evidence
        ]
        question = evidence_result.request.question
        warnings = [f"Module 2: {item}" for item in evidence_result.warnings]

        if not source_evidence:
            warnings.append(
                "Module 2 supplied no evidence; material extraction was not attempted"
            )
            return self._result(
                MaterialKnowledgeStatus.NO_EVIDENCE,
                question,
                evidence_result.status,
                source_evidence,
                warnings=warnings,
            )

        try:
            draft = self._reasoning_model.extract(
                question,
                source_evidence,
                max_records_per_type=validated.limits.max_records_per_type,
            )
        except Exception as exc:
            trace = MaterialExtractionModelTrace(
                ok=False,
                error_type=type(exc).__name__,
                error_message=str(exc)[:500],
            )
            warnings.append("Structured material knowledge extraction failed")
            return self._result(
                MaterialKnowledgeStatus.FAILED,
                question,
                evidence_result.status,
                source_evidence,
                traces=[trace],
                warnings=warnings,
            )

        known_evidence = {item.evidence_id for item in source_evidence}
        limit = validated.limits.max_records_per_type
        materials, material_warnings = _materialize(
            draft.materials,
            MaterialEntity,
            "material_id",
            "material",
            "Material",
            known_evidence,
            limit,
        )
        properties, property_warnings = _materialize(
            draft.properties,
            PropertyObservation,
            "property_id",
            "property",
            "Property",
            known_evidence,
            limit,
        )
        synthesis, synthesis_warnings = _materialize(
            draft.synthesis_procedures,
            SynthesisProcedure,
            "synthesis_id",
            "synthesis",
            "Synthesis",
            known_evidence,
            limit,
        )
        simulations, simulation_warnings = _materialize(
            draft.simulation_methods,
            SimulationMethod,
            "simulation_id",
            "simulation",
            "Simulation",
            known_evidence,
            limit,
        )
        relations, relation_warnings = _materialize(
            draft.relations,
            MaterialRelation,
            "relation_id",
            "relation",
            "Relation",
            known_evidence,
            limit,
        )
        local_warnings = [
            *material_warnings,
            *property_warnings,
            *synthesis_warnings,
            *simulation_warnings,
            *relation_warnings,
        ]
        warnings.extend(local_warnings)
        trace = MaterialExtractionModelTrace(ok=True)
        record_count = sum(
            len(items)
            for items in (
                materials,
                properties,
                synthesis,
                simulations,
                relations,
            )
        )
        incomplete_input = evidence_result.status != EvidenceResearchStatus.COMPLETE
        if record_count == 0 and not local_warnings and not incomplete_input:
            status = MaterialKnowledgeStatus.NO_KNOWLEDGE
        elif local_warnings or incomplete_input:
            status = MaterialKnowledgeStatus.PARTIAL
        else:
            status = MaterialKnowledgeStatus.COMPLETE

        return self._result(
            status,
            question,
            evidence_result.status,
            source_evidence,
            materials=materials,
            properties=properties,
            synthesis=synthesis,
            simulations=simulations,
            relations=relations,
            traces=[trace],
            warnings=warnings,
        )

    def close(self) -> None:
        close = getattr(self._reasoning_model, "close", None)
        if callable(close):
            close()

    def _result(
        self,
        status: MaterialKnowledgeStatus,
        question: str,
        evidence_status: EvidenceResearchStatus,
        source_evidence: list[EvidenceRef],
        *,
        materials: list[MaterialEntity] | None = None,
        properties: list[PropertyObservation] | None = None,
        synthesis: list[SynthesisProcedure] | None = None,
        simulations: list[SimulationMethod] | None = None,
        relations: list[MaterialRelation] | None = None,
        traces: list[MaterialExtractionModelTrace] | None = None,
        warnings: list[str] | None = None,
    ) -> MaterialKnowledgeExtractionResult:
        return MaterialKnowledgeExtractionResult(
            status=status,
            question=question,
            model=self._reasoning_model.model_name,
            evidence_status=evidence_status,
            source_evidence=source_evidence,
            materials=materials or [],
            properties=properties or [],
            synthesis_procedures=synthesis or [],
            simulation_methods=simulations or [],
            relations=relations or [],
            model_traces=traces or [],
            warnings=warnings or [],
        )


def build_material_knowledge_extractor(
    *,
    env_file: str | Path = ".env",
) -> MaterialKnowledgeExtractor:
    """Build the extraction module from the existing DeepSeek adapter."""

    chat_model = DeepSeekChatModel.from_env(env_file)
    return MaterialKnowledgeExtractor(DeepSeekMaterialKnowledgeModel(chat_model))


def _materialize(
    proposals: list[ProposalT],
    record_type: type[RecordT],
    id_field: str,
    id_prefix: str,
    label: str,
    known_evidence: set[str],
    limit: int,
) -> tuple[list[RecordT], list[str]]:
    records: list[RecordT] = []
    keys: dict[str, int] = {}
    warnings: list[str] = []
    if len(proposals) > limit:
        warnings.append(f"{label} proposals exceeded the configured limit")

    for proposal_number, proposal in enumerate(proposals[:limit], start=1):
        evidence_ids = list(getattr(proposal, "evidence_ids"))
        unknown = [item for item in evidence_ids if item not in known_evidence]
        if unknown:
            warnings.append(
                f"{label} proposal {proposal_number} referenced unknown evidence "
                "and was rejected"
            )
            continue
        key = json.dumps(
            proposal.model_dump(mode="json", exclude={"evidence_ids"}),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if key in keys:
            index = keys[key]
            existing = records[index]
            merged = list(
                dict.fromkeys(
                    [*getattr(existing, "evidence_ids"), *evidence_ids]
                )
            )
            if len(merged) > 50:
                warnings.append(
                    f"{label} duplicate evidence references exceeded 50 and "
                    "were truncated"
                )
                merged = merged[:50]
            merged_payload = existing.model_dump(mode="python")
            merged_payload["evidence_ids"] = merged
            records[index] = record_type.model_validate(merged_payload)
            continue
        payload = proposal.model_dump(mode="python")
        payload[id_field] = f"{id_prefix}-{len(records) + 1}"
        records.append(record_type.model_validate(payload))
        keys[key] = len(records) - 1
    return records, warnings
