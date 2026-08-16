from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from research_gap_agent.evidence import (
    EvidenceRef,
    EvidenceResearchResult,
    EvidenceResearchStatus,
    ResearchLimits,
    ResearchRequest,
)
from research_gap_agent.materials import (
    CompositionComponent,
    MaterialCondition,
    MaterialEntityProposal,
    MaterialExtractionLimits,
    MaterialKnowledgeDraft,
    MaterialKnowledgeExtractionRequest,
    MaterialKnowledgeExtractor,
    MaterialKnowledgeStatus,
    MaterialRelationProposal,
    MaterialRelationType,
    PropertyObservationProposal,
    ReportedValue,
    SimulationMethodProposal,
    StructureDescriptor,
    StructureKind,
    SynthesisProcedureProposal,
)


def _evidence(evidence_id: str) -> EvidenceRef:
    rank = int(evidence_id.rsplit("-", 1)[-1])
    return EvidenceRef(
        evidence_id=evidence_id,
        source_tool_call_id="evidence-call-002",
        rank=rank,
        doc_id=f"doc-{rank}",
        chunk_id=f"chunk-{rank}",
        offset=rank * 100,
        title="LLZO source",
        publication_year=2025,
        score=0.9,
        quoted_text="Explicitly reported materials knowledge.",
        raw_chunk={"unchanged": True},
    )


def _research_result(
    *,
    status: EvidenceResearchStatus = EvidenceResearchStatus.COMPLETE,
    evidence: list[EvidenceRef] | None = None,
) -> EvidenceResearchResult:
    return EvidenceResearchResult(
        status=status,
        request=ResearchRequest(
            question="Which LLZO compositions, processes, and properties are reported?",
            limits=ResearchLimits(context_expansions=0),
        ),
        evidence=(
            [_evidence("evidence-1"), _evidence("evidence-2")]
            if evidence is None
            else evidence
        ),
    )


def _draft() -> MaterialKnowledgeDraft:
    material = {
        "name": "Ta-doped LLZO",
        "formula": "Li6.4La3Zr1.4Ta0.6O12",
        "composition": [
            CompositionComponent(
                species="Ta",
                amount=0.6,
                raw_amount="x = 0.6",
            )
        ],
        "structures": [
            StructureDescriptor(
                kind=StructureKind.CRYSTAL_STRUCTURE,
                value="cubic garnet",
            )
        ],
    }
    return MaterialKnowledgeDraft(
        materials=[
            MaterialEntityProposal(**material, evidence_ids=["evidence-1"]),
            MaterialEntityProposal(**material, evidence_ids=["evidence-2"]),
        ],
        properties=[
            PropertyObservationProposal(
                material_name="Ta-doped LLZO",
                property_name="ionic conductivity",
                value=ReportedValue(
                    raw_value="1.2 mS cm-1",
                    numeric_value=1.2,
                    unit="mS cm-1",
                ),
                conditions=[
                    MaterialCondition(name="temperature", value="25", unit="°C")
                ],
                measurement_method="electrochemical impedance spectroscopy",
                evidence_ids=["evidence-1"],
            )
        ],
        synthesis_procedures=[
            SynthesisProcedureProposal(
                material_name="Ta-doped LLZO",
                method="solid-state reaction",
                precursors=["Li2CO3", "La2O3", "ZrO2", "Ta2O5"],
                steps=["mix", "calcine", "sinter"],
                conditions=[
                    MaterialCondition(
                        name="sintering temperature",
                        value="1230",
                        unit="°C",
                    )
                ],
                evidence_ids=["evidence-1"],
            )
        ],
        simulation_methods=[
            SimulationMethodProposal(
                material_name="Ta-doped LLZO",
                method="density functional theory",
                software="VASP",
                theory_or_model="PBE",
                parameters=[
                    MaterialCondition(
                        name="plane-wave cutoff",
                        value="520",
                        unit="eV",
                    )
                ],
                outputs=["formation energy"],
                evidence_ids=["evidence-2"],
            )
        ],
        relations=[
            MaterialRelationProposal(
                relation_type=MaterialRelationType.STRUCTURE_PROPERTY,
                subject="cubic garnet structure",
                predicate="is associated with",
                object="higher ionic conductivity",
                evidence_ids=["evidence-1", "evidence-2"],
            )
        ],
    )


@dataclass
class FakeMaterialReasoningModel:
    draft: MaterialKnowledgeDraft
    error: Exception | None = None
    model_name: str = "fake-deepseek"
    calls: list[tuple[str, list[EvidenceRef], int]] = field(default_factory=list)

    def extract(
        self,
        question: str,
        evidence: list[EvidenceRef],
        *,
        max_records_per_type: int,
    ) -> MaterialKnowledgeDraft:
        self.calls.append((question, evidence, max_records_per_type))
        if self.error:
            raise self.error
        return self.draft


def test_extraction_builds_all_grounded_record_types_and_merges_duplicates() -> None:
    reasoning = FakeMaterialReasoningModel(_draft())

    result = MaterialKnowledgeExtractor(reasoning).extract(
        MaterialKnowledgeExtractionRequest(evidence_result=_research_result())
    )

    assert result.status == MaterialKnowledgeStatus.COMPLETE
    assert result.model == "fake-deepseek"
    assert len(reasoning.calls) == 1
    assert len(result.materials) == 1
    assert result.materials[0].material_id == "material-1"
    assert result.materials[0].evidence_ids == ["evidence-1", "evidence-2"]
    assert result.properties[0].property_id == "property-1"
    assert result.synthesis_procedures[0].synthesis_id == "synthesis-1"
    assert result.simulation_methods[0].simulation_id == "simulation-1"
    assert result.relations[0].relation_id == "relation-1"
    assert result.model_traces[0].ok is True
    assert '"property_id":"property-1"' in result.model_dump_json()


def test_unknown_evidence_reference_rejects_only_the_ungrounded_record() -> None:
    draft = _draft()
    draft.relations[0].evidence_ids = ["evidence-999"]

    result = MaterialKnowledgeExtractor(
        FakeMaterialReasoningModel(draft)
    ).extract(MaterialKnowledgeExtractionRequest(evidence_result=_research_result()))

    assert result.status == MaterialKnowledgeStatus.PARTIAL
    assert result.materials
    assert result.relations == []
    assert "unknown evidence" in result.warnings[-1]


def test_empty_evidence_bypasses_reasoning_model() -> None:
    reasoning = FakeMaterialReasoningModel(_draft())

    result = MaterialKnowledgeExtractor(reasoning).extract(
        MaterialKnowledgeExtractionRequest(
            evidence_result=_research_result(evidence=[])
        )
    )

    assert result.status == MaterialKnowledgeStatus.NO_EVIDENCE
    assert reasoning.calls == []
    assert result.model_traces == []


def test_empty_valid_draft_returns_no_knowledge() -> None:
    result = MaterialKnowledgeExtractor(
        FakeMaterialReasoningModel(MaterialKnowledgeDraft())
    ).extract(MaterialKnowledgeExtractionRequest(evidence_result=_research_result()))

    assert result.status == MaterialKnowledgeStatus.NO_KNOWLEDGE
    assert result.materials == []
    assert result.model_traces[0].ok is True


def test_reasoning_failure_returns_structured_failed_result() -> None:
    result = MaterialKnowledgeExtractor(
        FakeMaterialReasoningModel(
            MaterialKnowledgeDraft(),
            error=RuntimeError("invalid structured reply"),
        )
    ).extract(MaterialKnowledgeExtractionRequest(evidence_result=_research_result()))

    assert result.status == MaterialKnowledgeStatus.FAILED
    assert result.materials == []
    assert result.model_traces[0].ok is False
    assert result.model_traces[0].stage == "extraction"


def test_partial_evidence_result_cannot_produce_complete_extraction() -> None:
    result = MaterialKnowledgeExtractor(
        FakeMaterialReasoningModel(_draft())
    ).extract(
        MaterialKnowledgeExtractionRequest(
            evidence_result=_research_result(status=EvidenceResearchStatus.PARTIAL)
        )
    )

    assert result.status == MaterialKnowledgeStatus.PARTIAL
    assert result.evidence_status == EvidenceResearchStatus.PARTIAL


def test_record_limit_truncates_output_and_marks_result_partial() -> None:
    draft = MaterialKnowledgeDraft(
        materials=[
            MaterialEntityProposal(name="LLZO", evidence_ids=["evidence-1"]),
            MaterialEntityProposal(name="LAGP", evidence_ids=["evidence-2"]),
        ]
    )

    result = MaterialKnowledgeExtractor(
        FakeMaterialReasoningModel(draft)
    ).extract(
        MaterialKnowledgeExtractionRequest(
            evidence_result=_research_result(),
            limits=MaterialExtractionLimits(max_records_per_type=1),
        )
    )

    assert result.status == MaterialKnowledgeStatus.PARTIAL
    assert [item.name for item in result.materials] == ["LLZO"]
    assert "configured limit" in result.warnings[-1]


def test_bounded_input_rejects_reference_to_evidence_not_sent_to_model() -> None:
    draft = MaterialKnowledgeDraft(
        materials=[
            MaterialEntityProposal(name="LAGP", evidence_ids=["evidence-2"])
        ]
    )
    reasoning = FakeMaterialReasoningModel(draft)

    result = MaterialKnowledgeExtractor(reasoning).extract(
        MaterialKnowledgeExtractionRequest(
            evidence_result=_research_result(),
            limits=MaterialExtractionLimits(max_input_evidence=1),
        )
    )

    assert len(reasoning.calls[0][1]) == 1
    assert result.materials == []
    assert result.status == MaterialKnowledgeStatus.PARTIAL
    assert "unknown evidence" in result.warnings[-1]


def test_material_contracts_reject_invalid_bounds_and_empty_composition_amount() -> None:
    with pytest.raises(ValidationError):
        MaterialExtractionLimits(max_records_per_type=101)
    with pytest.raises(ValidationError):
        CompositionComponent(species="Ta")
