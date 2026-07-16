"""
Freelancer matching agent

Reranks the freelancer candidates the backend provides for a given role. It is a
deterministic, explainable hybrid reranker over the bounded pool the backend
assembles.

("Deterministic" = same input always gives the same output; there is NO LLM and
no randomness here. Despite the name "agent", this file is a plain scoring
function — it ranks candidates the backend hands it, it does not "think".)

`projectFit` fuses two brief-relevance signals with Reciprocal Rank Fusion (RRF),
the scale-free way to combine dense and sparse retrieval:

  - Dense:  `candidate.embeddingSimilarity` — pgvector cosine of the brief vs.
            the freelancer profile embedding, computed by the backend (this is
            the only "AI" number, and it is made elsewhere, not in this file).
  - Sparse: Okapi BM25 over the candidate profile text (skills + headline +
            profileSummary) against the brief topic. BM25 = pure word-counting,
            no model.

The structured components (skills, availability, experience, rate) are scored
separately and pool-relative. The response shape matches the
`POST /agents/match-freelancers` API contract.
"""

import math
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# --- Scoring weights ---------------------------------------------------------
# Each freelancer earns points in 5 categories. Each scorer below returns a
# value in [0, 1]; we multiply by the weight here. The weights sum to 100, so
# the final score is 0-100. Change these to change what "a good match" means.
WEIGHTS = {
    "skills": 35,       # required-skill overlap + skill scores + reliability
    "projectFit": 20,   # brief relevance: dense + sparse fused via RRF
    "availability": 15,  # weekly hours (pool-relative)
    "experience": 15,   # years of experience (pool-relative)
    "rateFit": 15,      # hourly rate (pool-relative)
}

# --- Tuning knobs (standard defaults; you rarely change these) ---------------
RRF_K = 60            # RRF smoothing: higher = the gap between rank #1/#2/#3 matters less
DENSE_WEIGHT = 0.6    # how much we trust the embedding (meaning) signal
SPARSE_WEIGHT = 0.4   # how much we trust the BM25 (keyword) signal   (0.6 + 0.4 = 1.0)
BM25_K1 = 1.5         # how fast repeated words stop helping (term-frequency saturation)
BM25_B = 0.75         # how much to penalize long profiles (length normalization)

# Boring words to ignore when comparing text, so "the"/"and" don't count as matches.
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "are", "was", "will", "can",
    "has", "have", "from", "into", "all", "any", "our", "your", "their", "who",
    "web", "app", "apps", "using", "use", "used", "need", "needs", "want",
}


# --- Request models ----------------------------------------------------------
# Pydantic validates the incoming JSON and turns it into Python objects.
# `extra="ignore"` means: if the backend sends fields we don't list here, just
# drop them silently. So we only declare the fields this file actually reads.
class _Loose(BaseModel):
    model_config = ConfigDict(extra="ignore")


# `str | None = None` means "a string OR nothing, defaulting to nothing" (optional).
# `Field(default_factory=list)` means "if missing, default to an empty list []".
class ProjectForMatching(_Loose):
    title: str | None = None
    description: str | None = None
    requiredSkills: list[str] = Field(default_factory=list)


class BriefForMatching(_Loose):
    summary: str | None = None
    briefText: str | None = None
    requiredSkills: str | list[str] | None = None  # may arrive as a list OR a "a, b, c" string


class SkillScore(_Loose):
    skill: str
    score: float  # 0-5, from the freelancer's assessment


class FreelancerCandidate(_Loose):
    freelancerProfileId: str
    name: str | None = None
    headline: str | None = None
    skills: list[str] = Field(default_factory=list)
    skillScores: list[SkillScore] = Field(default_factory=list)
    averageSkillScore: float | None = None       # 0-5, mean of their skill scores
    availabilityHours: float | None = 0           # free hours per week
    hourlyRate: float | None = None
    yearsExperience: float | None = 0
    profileSummary: str | None = None             # short bio text
    embeddingSimilarity: float | None = None      # 0-1 "meaning" score, computed by the backend


class MatchFreelancersRequest(_Loose):
    matchingRunId: str                 # id of this run (echoed back so the backend can match it up)
    targetRoleKey: str | None = None   # e.g. "architect" or "ui_ux"
    limit: int | None = None           # how many top candidates to return
    project: ProjectForMatching = Field(default_factory=ProjectForMatching)
    brief: BriefForMatching | None = Field(default_factory=BriefForMatching)
    candidates: list[FreelancerCandidate] = Field(default_factory=list)

    # A method on the request: "which skills does this role need?"
    def required_skills(self) -> list[str]:
        """Required skills from `project.requiredSkills`, else a delimited
        `brief.requiredSkills` string."""
        if self.project.requiredSkills:
            return self.project.requiredSkills
        raw = self.brief.requiredSkills if self.brief else None
        if isinstance(raw, list):
            return [str(s) for s in raw if str(s).strip()]
        # A string like "NestJS, PostgreSQL; Figma" -> split on commas/newlines/semicolons.
        if isinstance(raw, str) and raw.strip():
            return [s.strip() for s in re.split(r"[,\n;]+", raw) if s.strip()]
        return []


# --- Generic helpers ---------------------------------------------------------
def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    # Keep a number inside [0, 1]:  1.3 -> 1.0,  -0.2 -> 0.0.
    return max(low, min(high, value))


def _norm(text: str) -> str:
    # Lowercase + trim spaces, so "NestJS " and "nestjs" are treated as equal.
    return text.strip().lower()


def _tokenize(text: str) -> list[str]:
    # Turn a blob of text into a clean list of words.
    # "Build a Web App!" -> ["build", "app"]  (lowercased, stopwords dropped).
    tokens = re.findall(r"[a-z0-9]+", text.lower())  # grab word chunks
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


# --- Hybrid relevance: sparse (BM25) + dense (vector) fused via RRF -----------
class BM25:
    """Keyword-relevance scorer. No AI: it measures how many *important* words a
    profile shares with the brief. Built once over the whole candidate pool."""

    def __init__(self, corpus: list[list[str]], k1: float = BM25_K1, b: float = BM25_B):
        # corpus = one tokenized profile (a list of words) per candidate.
        self.k1 = k1
        self.b = b
        self.docs = corpus
        self.doc_len = [len(doc) for doc in corpus]        # word count of each profile
        n = len(corpus)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0  # average profile length

        # Count how many profiles contain each word ("document frequency").
        doc_freq: dict[str, int] = {}
        for doc in corpus:
            for term in set(doc):  # set() so a word counts once per profile, not per repeat
                doc_freq[term] = doc_freq.get(term, 0) + 1
        # IDF = a "rarity weight". A word found in few profiles is a strong signal;
        # a word in almost every profile means little. (This log(...) form stays
        # non-negative even for very common words.)
        self.idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def score(self, query: list[str], index: int) -> float:
        # How relevant is profile `index` to the `query` (the brief words)?
        doc = self.docs[index]
        if not doc or self.avgdl == 0:
            return 0.0
        # Count how often each word appears in THIS profile ("term frequency").
        freqs: dict[str, int] = {}
        for term in doc:
            freqs[term] = freqs.get(term, 0) + 1

        # Length adjustment: discount long profiles so they don't win by size (knob b).
        length_norm = 1 - self.b + self.b * self.doc_len[index] / self.avgdl
        total = 0.0
        for term in set(query):
            tf = freqs.get(term)
            if not tf:                 # this brief-word isn't in the profile
                continue
            # Add points = rarity(idf) x saturated word-count. k1 caps how much
            # repeating the same word keeps helping.
            total += self.idf.get(term, 0.0) * (tf * (self.k1 + 1)) / (tf + self.k1 * length_norm)
        return total


def _build_query(request: MatchFreelancersRequest) -> list[str]:
    # The "query" = the project's words (what we're matching AGAINST).
    # Required skills are excluded on purpose: they're already scored by the
    # structured `skills` component, so including them here would count twice.
    brief = request.brief
    parts = [request.project.title or "", request.project.description or ""]
    if brief is not None:
        parts += [brief.summary or "", brief.briefText or ""]
    return _tokenize(" ".join(parts))  # join into one string, then split into words


def _candidate_document(candidate: FreelancerCandidate) -> list[str]:
    # The "document" = one freelancer's words (what we're matching each candidate BY).
    parts = [*candidate.skills, candidate.headline or "", candidate.profileSummary or ""]
    return _tokenize(" ".join(parts))


def _rrf_ranks(values: list[float | None]) -> list[int | None]:
    """Turn a list of scores into ranks (1 = best). Ties get the same rank.
    A `None` score means "no value for this signal" and gets no rank."""
    # Indices of candidates that HAVE a value, sorted from highest score to lowest.
    present = sorted(
        (i for i, v in enumerate(values) if v is not None),
        key=lambda i: values[i],  # type: ignore[index]
        reverse=True,
    )
    ranks: list[int | None] = [None] * len(values)  # start everyone at "no rank"
    prev_val: float | None = None
    prev_rank = 0
    for position, i in enumerate(present, start=1):  # position = 1, 2, 3, ...
        val = values[i]
        if prev_val is not None and val == prev_val:
            ranks[i] = prev_rank            # same score as the previous -> same rank (tie)
        else:
            ranks[i] = position
            prev_rank = position
            prev_val = val
    return ranks


def _compute_project_fit(request: MatchFreelancersRequest) -> list[float]:
    """Produce one `projectFit` number (0-1) per candidate by fusing the dense
    (embedding) and sparse (BM25) signals with Reciprocal Rank Fusion."""
    candidates = request.candidates
    if not candidates:
        return []

    # 1) Dense signal: the backend's "meaning" score (may be missing -> None).
    dense_vals: list[float | None] = [
        _clamp(c.embeddingSimilarity) if c.embeddingSimilarity is not None else None
        for c in candidates
    ]

    # 2) Sparse signal: BM25 keyword score of the brief vs. each profile.
    query = _build_query(request)
    bm25 = BM25([_candidate_document(c) for c in candidates])
    sparse_vals: list[float | None] = [
        (score if score > 0 else None)  # 0 overlap -> treat as "no signal" (None)
        for score in (bm25.score(query, i) for i in range(len(candidates)))
    ]

    # 3) Rank candidates by each signal separately.
    dense_ranks = _rrf_ranks(dense_vals)
    sparse_ranks = _rrf_ranks(sparse_vals)

    # 4) Fuse the two RANKINGS (not the raw scores, which live on different scales).
    #    Being ranked #1 by a signal contributes weight/(K+1); #2 slightly less, etc.
    raw: list[float] = []
    for dense_rank, sparse_rank in zip(dense_ranks, sparse_ranks):
        fused = 0.0
        if dense_rank is not None:
            fused += DENSE_WEIGHT / (RRF_K + dense_rank)
        if sparse_rank is not None:
            fused += SPARSE_WEIGHT / (RRF_K + sparse_rank)
        raw.append(fused)

    # 5) Normalize to [0, 1] so the most brief-relevant candidate scores a full 1.0.
    max_raw = max(raw, default=0.0)
    return [(value / max_raw) if max_raw > 0 else 0.0 for value in raw]


# --- Structured component scoring (each returns a number in [0, 1]) ----------
def _score_skills(cand: FreelancerCandidate, required: list[str]) -> tuple[float, list[str], list[str]]:
    # Which required skills does the candidate have / not have?
    cand_skills = {_norm(skill) for skill in cand.skills}  # a set for fast lookup
    matched = [req for req in required if _norm(req) in cand_skills]
    missing = [req for req in required if _norm(req) not in cand_skills]

    match_ratio = len(matched) / len(required) if required else 0.0     # e.g. 3 of 5 -> 0.6
    scores = [s.score for s in cand.skillScores if s.score is not None]
    skill_quality = (sum(scores) / len(scores) / 5.0) if scores else 0.5  # avg skill score, 0-5 -> 0-1
    reliability = cand.averageSkillScore / 5.0 if cand.averageSkillScore is not None else 0.5

    # Blend: mostly "do they have the skills", plus how good/reliable they are.
    if required:
        norm = 0.6 * match_ratio + 0.3 * _clamp(skill_quality) + 0.1 * _clamp(reliability)
    else:
        # No required-skill list -> fall back to overall quality only.
        norm = 0.7 * _clamp(skill_quality) + 0.3 * _clamp(reliability)
    return _clamp(norm), matched, missing


def _score_availability(cand: FreelancerCandidate, pool_max: float) -> float:
    hours = cand.availabilityHours or 0
    if hours <= 0:
        return 0.0                       # no free time -> worst score
    if not pool_max:
        return 1.0
    # Baseline 0.6, plus up to 0.4 more, scaled against the most-available person.
    return _clamp(0.6 + 0.4 * hours / pool_max)


def _score_experience(cand: FreelancerCandidate, pool_max: float) -> float:
    # Your years / the most experienced person's years. Most experienced -> 1.0.
    return _clamp((cand.yearsExperience or 0) / pool_max) if pool_max else 0.0


def _score_rate_fit(cand: FreelancerCandidate, pool_min: float, pool_max: float) -> float:
    # Cheaper candidates rank higher within the pool. (The backend already
    # removed anyone over budget, so everyone here is affordable.)
    rate = cand.hourlyRate
    if rate is None:
        return 0.5                       # unknown rate -> neutral
    if pool_max > pool_min:
        # cheapest -> ~1.0, priciest -> ~0.5
        return _clamp(0.5 + 0.5 * (1 - (rate - pool_min) / (pool_max - pool_min)))
    return 0.75                          # everyone charges the same -> neutral-ish


def _build_rationale(cand: FreelancerCandidate, matched: list[str], required: list[str], project_fit: float) -> str:
    # Builds the human sentence shown to the admin, e.g.
    # "Matches 3 of 5 required skills; 20h availability; 4 years experience; rate 25."
    parts: list[str] = []
    if required:
        parts.append(f"Matches {len(matched)} of {len(required)} required skills")
    elif matched:
        parts.append("Relevant skill set")
    if project_fit >= 0.6:
        parts.append("strong relevance to the project brief")
    parts.append(f"{cand.availabilityHours or 0:g}h availability")
    parts.append(f"{cand.yearsExperience or 0:g} years experience")
    if cand.hourlyRate is not None:
        parts.append(f"rate {cand.hourlyRate:g}")
    if (cand.availabilityHours or 0) <= 0:
        parts.append("but currently has no availability")
    return "; ".join(parts) + "."       # glue the pieces with "; "


class _PoolStats:
    """Numbers about the WHOLE group, computed once. The scorers above are
    "pool-relative" (they compare each person to the group), so they need these."""

    def __init__(self, candidates: list[FreelancerCandidate]):
        self.max_exp = max((c.yearsExperience or 0 for c in candidates), default=0)
        self.max_avail = max((c.availabilityHours or 0 for c in candidates), default=0)
        rates = [c.hourlyRate for c in candidates if c.hourlyRate is not None]
        self.min_rate = min(rates, default=0)   # default=0 avoids crashing on an empty list
        self.max_rate = max(rates, default=0)


def _score_candidate(cand: FreelancerCandidate, required: list[str], project_fit: float, pool: _PoolStats) -> dict[str, Any]:
    """Score ONE freelancer: run all 5 scorers, weight them, and package the
    result (score + per-category breakdown + rationale + evidence)."""
    skills_norm, matched, missing = _score_skills(cand, required)

    # Each 0-1 sub-score x its weight = points for that category.
    breakdown = {
        "skills": round(skills_norm * WEIGHTS["skills"], 2),
        "availability": round(_score_availability(cand, pool.max_avail) * WEIGHTS["availability"], 2),
        "experience": round(_score_experience(cand, pool.max_exp) * WEIGHTS["experience"], 2),
        "rateFit": round(_score_rate_fit(cand, pool.min_rate, pool.max_rate) * WEIGHTS["rateFit"], 2),
        "projectFit": round(project_fit * WEIGHTS["projectFit"], 2),
    }
    score = round(sum(breakdown.values()), 2)  # add the 5 categories -> 0-100

    # Warnings the admin sees on the candidate card.
    hours = cand.availabilityHours or 0
    risk_flags: list[str] = []
    if hours <= 0:
        risk_flags.append("no_availability")
    if required and missing:
        risk_flags.append("missing_required_skills")
    if cand.averageSkillScore is not None and cand.averageSkillScore < 3.0:
        risk_flags.append("low_assessment_score")

    # This dict is exactly what the backend stores and the UI shows.
    return {
        "freelancerProfileId": cand.freelancerProfileId,
        "score": score,
        "scoreBreakdown": breakdown,
        "rationale": _build_rationale(cand, matched, required, project_fit),
        "evidence": {
            "matchedSkills": matched,
            "missingSkills": missing,
            "riskFlags": risk_flags,
            "hourlyRate": cand.hourlyRate,
            "availabilityHours": hours,
            "yearsExperience": cand.yearsExperience or 0,
        },
    }


def match_freelancers(request: MatchFreelancersRequest) -> dict[str, Any]:
    """The entry point. Score every candidate, sort, keep the top N, return them."""
    project_fits = _compute_project_fit(request)   # one projectFit per candidate
    pool = _PoolStats(request.candidates)          # group stats, computed once
    required = request.required_skills()           # skills this role needs, computed once

    # Score each candidate. zip(...) pairs each candidate with its projectFit.
    scored = [
        _score_candidate(cand, required, fit, pool)
        for cand, fit in zip(request.candidates, project_fits)
    ]

    # Sort best-first. On a score tie, prefer higher projectFit, then availability.
    scored.sort(
        key=lambda c: (c["score"], c["scoreBreakdown"]["projectFit"], c["evidence"]["availabilityHours"]),
        reverse=True,
    )

    ranked = scored[: (request.limit or 10)]        # keep the top N (default 10)
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank                    # number them 1, 2, 3, ...

    # The response the backend expects.
    return {
        "matchingRunId": request.matchingRunId,
        "status": "completed",
        "summary": _build_summary(ranked, request),
        "candidates": ranked,
    }


def _build_summary(ranked: list[dict[str, Any]], request: MatchFreelancersRequest) -> str:
    # One-line human summary shown at the top of the results.
    role = request.targetRoleKey or "role"
    count = len(ranked)
    if count == 0:
        return f"No eligible {role} candidates were available to rank."

    top = ranked[0]
    # Find the top candidate's name (the ranked dict only has the id, not the name).
    top_name = next(
        (c.name for c in request.candidates if c.freelancerProfileId == top["freelancerProfileId"] and c.name),
        "the top candidate",
    )
    return (
        f"Ranked {count} approved {role} candidate{'s' if count != 1 else ''} by brief "
        f"relevance (vector + BM25 via reciprocal rank fusion), skills, availability, "
        f"experience, and rate fit. {top_name} leads with a score of {top['score']:g}."
    )
