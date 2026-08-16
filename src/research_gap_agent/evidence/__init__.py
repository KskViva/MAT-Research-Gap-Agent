"""SciVerse evidence acquisition contracts and orchestration."""

from .models import (
    EvidenceRef,
    EvidenceResearchResult,
    EvidenceResearchStatus,
    EvidenceToolTrace,
    OriginalTextExcerpt,
    PaperScope,
    ResearchLimits,
    ResearchRequest,
)
from .researcher import (
    SciVerseEvidenceResearcher,
    build_sciverse_evidence_researcher,
)

__all__ = [
    "EvidenceRef",
    "EvidenceResearchResult",
    "EvidenceResearchStatus",
    "EvidenceToolTrace",
    "OriginalTextExcerpt",
    "PaperScope",
    "ResearchLimits",
    "ResearchRequest",
    "SciVerseEvidenceResearcher",
    "build_sciverse_evidence_researcher",
]
