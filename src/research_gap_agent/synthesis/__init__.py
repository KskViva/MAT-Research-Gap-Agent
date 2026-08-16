"""Evidence-grounded synthesis and research-gap verification."""

from .analyzer import (
    EvidenceGroundedGapAnalyzer,
    EvidenceResearcher,
    build_gap_analyzer,
)
from .deepseek import (
    DeepSeekGapReasoningModel,
    GapReasoningError,
    GapReasoningModel,
)
from .models import (
    Claim,
    ClaimProposal,
    CounterResearchRecord,
    GapAnalysisLimits,
    GapAnalysisRequest,
    GapAnalysisResult,
    GapAnalysisStatus,
    GapAssessmentDraft,
    GapCandidate,
    GapCandidateProposal,
    GapCategory,
    GapVerificationStatus,
    ModelReasoningTrace,
    SynthesisDraft,
    VerifiedGap,
)

__all__ = [
    "Claim",
    "ClaimProposal",
    "CounterResearchRecord",
    "DeepSeekGapReasoningModel",
    "EvidenceGroundedGapAnalyzer",
    "EvidenceResearcher",
    "GapAnalysisLimits",
    "GapAnalysisRequest",
    "GapAnalysisResult",
    "GapAnalysisStatus",
    "GapAssessmentDraft",
    "GapCandidate",
    "GapCandidateProposal",
    "GapCategory",
    "GapReasoningError",
    "GapReasoningModel",
    "GapVerificationStatus",
    "ModelReasoningTrace",
    "SynthesisDraft",
    "VerifiedGap",
    "build_gap_analyzer",
]
