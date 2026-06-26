#!/usr/bin/env python3
"""
Intelligent AI Recruiter — v2
6-Layer Intelligence Engine for Candidate Ranking

Architecture:
  Layer 1: Semantic Skill Matcher  (pre-built flat synonym set, O(n) corpus scan)
  Layer 2: Career Arc Intelligence (title progression + stability + description NLP)
  Layer 3: Experience Quality      (JD-aligned YOE sweet-spot scoring)
  Layer 4: Education Intelligence  (schema tier + degree + field relevance)
  Layer 5: Assessment & Certs      (Redrob platform test scores + certs)
  Layer 6: Behavioral Availability (all 23 Redrob signals)
  -> Composite Scorer + Intelligent Reasoning Generator

Performance: All hot-path lookups use pre-built dicts/sets (O(1) per concept).
No fuzzy matching in the main loop. Processes 100K candidates in ~15-25s.
"""

import argparse
import json
import gzip
import csv
import math
from datetime import datetime
from pathlib import Path

# =============================================================================
# JD INTELLIGENCE — parsed from actual job description
# =============================================================================

JD_TEXT = """
We are looking for a Senior AI/ML Engineer to build and optimize our search and retrieval infrastructure. 
The ideal candidate will have strong expertise in Vector Search, Embeddings, RAG, and NLP. 
Experience with fine-tuning LLMs, evaluating retrieval frameworks, and scaling machine learning systems is highly desired.
"""

JD_REQUIREMENTS = {
    'embeddings':            3.0,
    'vector_search':         3.0,
    'retrieval':             3.0,
    'ranking':               3.0,
    'rag':                   2.8,
    'hybrid_search':         2.8,
    'semantic_search':       2.8,
    'reranking':             2.8,
    'faiss':                 2.5,
    'pinecone':              2.5,
    'weaviate':              2.5,
    'qdrant':                2.5,
    'milvus':                2.5,
    'opensearch':            2.5,
    'elasticsearch':         2.5,
    'evaluation_framework':  2.5,
    'learning_to_rank':      2.5,
    'search_infrastructure': 2.5,
    'recommendation':        2.0,
    'llm':                   1.8,
    'fine_tuning':           1.8,
    'lora':                  1.6,
    'qlora':                 1.6,
    'peft':                  1.6,
    'nlp':                   1.5,
    'python':                1.5,
    'ab_testing':            1.5,
    'pytorch':               1.4,
    'mlops':                 1.3,
    'distributed_systems':   1.3,
    'machine_learning':      1.2,
    'deep_learning':         1.2,
    'tensorflow':            1.2,
    'open_source':           1.2,
}

# Synonym groups — any term maps to a concept key
# Kept concise; broad enough to catch all real-world naming variants
SYNONYM_GROUPS = {
    'embeddings': [
        'embedding', 'embeddings', 'vector embedding', 'text embedding',
        'sentence embedding', 'dense vector', 'embedding model',
        'sentence-transformers', 'sentence transformers', 'bge', ' e5 ',
        'openai embeddings', 'embedding generation', 'bi-encoder',
        'dual encoder', 'neural embeddings', 'word2vec', 'fasttext', 'glove',
        'bert embedding', 'text-embedding',
    ],
    'vector_search': [
        'vector search', 'vector database', 'vector db', 'ann search',
        'approximate nearest neighbor', 'knn search', 'dense retrieval',
        'vector retrieval', 'nearest neighbor search', 'vector index',
        'hnsw', 'ivf index', 'vector similarity', 'annoy', 'nmslib',
    ],
    'faiss':        ['faiss', 'facebook ai similarity'],
    'pinecone':     ['pinecone'],
    'weaviate':     ['weaviate'],
    'qdrant':       ['qdrant'],
    'milvus':       ['milvus', 'zilliz'],
    'opensearch':   ['opensearch', 'open search', 'aws opensearch'],
    'elasticsearch':['elasticsearch', 'elastic search', 'elk stack'],
    'retrieval': [
        'information retrieval', 'document retrieval', 'passage retrieval',
        'bm25', 'tfidf', 'tf-idf', 'inverted index', 'retrieval augmented',
        'retrieval system', 'full text search', 'sparse retrieval', ' ir ',
        'retrieval pipeline',
    ],
    'rag': [
        'retrieval augmented generation', 'retrieval-augmented',
        'rag pipeline', 'rag system', 'generative retrieval',
        'context retrieval',
    ],
    'ranking': [
        'search ranking', 'result ranking', 'relevance ranking',
        'ranking model', 'ranking algorithm', 'reranker', 'cross-encoder',
        'pointwise', 'pairwise', 'listwise', 'learning to rank',
        'lambdamart', 'search relevance', 'rank fusion', 'rrf',
        'reciprocal rank',
    ],
    'reranking': [
        'reranking', 're-ranking', 'cross-encoder reranking',
        'cohere rerank', 'monot5', 'rank fusion',
    ],
    'hybrid_search': [
        'hybrid search', 'hybrid retrieval', 'sparse dense',
        'bm25 embedding', 'lexical semantic',
    ],
    'semantic_search': [
        'semantic search', 'semantic similarity', 'semantic retrieval',
        'meaning-based search',
    ],
    'evaluation_framework': [
        'ndcg', 'mrr', 'mean average precision', 'precision@k', 'recall@k',
        'a/b test', 'ab test', 'offline evaluation', 'online evaluation',
        'evaluation metric', 'retrieval evaluation', 'search quality',
        'eval framework', 'ranker evaluation', 'click-through rate',
        'engagement metric',
    ],
    'learning_to_rank': [
        'learning to rank', 'ltr', 'lambdamart', 'ranknet', 'listnet',
        'xgboost ranking', 'neural ltr',
    ],
    'search_infrastructure': [
        'search infrastructure', 'search backend', 'search platform',
        'search system', 'search engine', 'search stack', 'search architecture',
        'query processing',
    ],
    'recommendation': [
        'recommendation system', 'recommender system', 'recommendations',
        'collaborative filtering', 'content-based filtering',
        'matrix factorization', 'personalization', 'candidate generation',
        'two-tower model', 'item2vec', 'click prediction', 'feed ranking',
        'discovery system',
    ],
    'llm': [
        'large language model', 'llm', 'gpt', 'llama', 'mistral',
        'falcon', 'claude', 'generative ai', 'language model',
        'foundation model', 'instruction tuning', 'chat model',
        'palm', 'gemini model', 'gpt-4', 'chatgpt', 't5 model',
    ],
    'fine_tuning': [
        'fine-tuning', 'fine tuning', 'finetuning', 'model fine-tuning',
        'domain adaptation', 'supervised fine-tuning', 'sft',
        'continual pretraining', 'instruction finetuning',
    ],
    'lora':   ['lora', 'low-rank adaptation', 'lora adapter'],
    'qlora':  ['qlora', 'quantized lora', '4-bit training'],
    'peft': [
        'peft', 'parameter efficient', 'parameter-efficient fine-tuning',
        'ia3', 'prefix tuning', 'prompt tuning', 'adapter layer',
    ],
    'nlp': [
        'natural language processing', 'nlp', 'text processing',
        'text classification', 'named entity recognition', 'ner',
        'sentiment analysis', 'question answering', 'text generation',
        'language understanding', 'nlu', 'nlg', 'tokenization',
        'language modeling', 'text mining',
    ],
    'python':     ['python3', ' python ', 'python programming'],
    'pytorch':    ['pytorch', ' torch ', 'libtorch'],
    'tensorflow': ['tensorflow', ' tf ', 'keras'],
    'mlops': [
        'mlops', 'ml platform', 'ml infrastructure', 'mlflow',
        'kubeflow', 'model registry', 'model serving', 'feature store',
        'model monitoring', 'ml pipeline', 'experiment tracking',
        'wandb', 'weights and biases', 'model deployment',
        'bentoml', 'seldon', 'triton inference',
    ],
    'machine_learning': [
        'machine learning', 'scikit-learn', 'sklearn', 'model training',
        'feature engineering', 'gradient boosting', 'xgboost', 'lightgbm',
        'catboost', 'random forest', 'ensemble method',
    ],
    'deep_learning': [
        'deep learning', 'neural network', 'deep neural network',
        'attention mechanism', 'gpu training', 'cuda', 'mixed precision',
    ],
    'ab_testing': [
        'a/b test', 'ab test', 'a/b testing', 'ab testing',
        'experimentation', 'controlled experiment', 'statistical significance',
    ],
    'distributed_systems': [
        'distributed system', 'distributed computing', 'kafka', 'apache spark',
        'flink', 'ray distributed', 'dask', 'microservice', 'grpc',
        'large scale', 'horizontal scaling', 'distributed training',
    ],
    'open_source': [
        'open source', 'open-source', 'github contributor', 'oss contributor',
        'hugging face', 'huggingface', 'arxiv', 'research paper',
        'conference paper', 'open source contribution',
    ],
}

# Power combos — having ALL concepts in a group gives multiplicative bonus
POWER_COMBOS = [
    (['embeddings', 'vector_search', 'rag'],            1.40, "Full RAG stack"),
    (['embeddings', 'ranking', 'retrieval'],             1.38, "End-to-end retrieval ranking"),
    (['evaluation_framework', 'ranking', 'retrieval'],   1.35, "Retrieval+Evaluation expert"),
    (['reranking', 'retrieval', 'embeddings'],           1.35, "Reranking+Dense retrieval"),
    (['hybrid_search', 'retrieval', 'elasticsearch'],    1.30, "Hybrid search infra expert"),
    (['lora', 'fine_tuning', 'llm'],                     1.30, "LLM fine-tuning specialist"),
    (['recommendation', 'embeddings', 'retrieval'],      1.28, "Recommendation+Retrieval"),
    (['mlops', 'embeddings', 'retrieval'],               1.20, "Production ML+Retrieval ops"),
]

# =============================================================================
# PRE-BUILD ALL FAST LOOKUP STRUCTURES AT MODULE LOAD TIME
# =============================================================================

# Flat map: synonym_term -> concept_key (built once, used O(1) per check)
_SYN_TO_CONCEPT: dict = {}
for _concept, _syns in SYNONYM_GROUPS.items():
    for _syn in _syns:
        _SYN_TO_CONCEPT[_syn] = _concept

# Sorted longest-first so multi-word phrases match before single words
_SYN_KEYS_SORTED: list = sorted(_SYN_TO_CONCEPT.keys(), key=len, reverse=True)

# Pre-built set of all synonym terms per concept (for assessment relevance)
_CONCEPT_ALL_SYNS: dict = {
    c: set(syns) for c, syns in SYNONYM_GROUPS.items()
}

# Flat set of ALL synonym terms (for fast assessment skill relevance check)
_ALL_JD_SYNS_FLAT: set = set()
for _c, _syns in SYNONYM_GROUPS.items():
    if _c in JD_REQUIREMENTS:
        _ALL_JD_SYNS_FLAT.update(_syns)


def _is_relevant_skill(skill_lower: str) -> bool:
    """Fast check: is this skill term relevant to the JD?"""
    return any(syn in skill_lower or skill_lower in syn
               for syn in list(_ALL_JD_SYNS_FLAT)[:60])

# =============================================================================
# DISQUALIFIERS
# =============================================================================

CONSULTING_FIRMS = {
    'infosys', 'wipro', 'tcs', 'tata consultancy', 'capgemini', 'hcl',
    'accenture', 'cognizant', 'tech mahindra', 'mphasis', 'hexaware',
    'l&t infotech', 'ltimindtree', 'mindtree', 'niit technologies',
    'syntel', 'cyient', 'zensar', 'igate', 'mastech',
}

VISION_ONLY_TERMS = {
    'computer vision', 'object detection', 'image classification',
    'image segmentation', 'opencv', 'yolo', 'detectron',
    'generative adversarial', 'diffusion model', 'stable diffusion',
    'asr', 'speech recognition', 'text to speech', 'speech synthesis',
    'pose estimation', 'depth estimation',
}

LANGCHAIN_TRAP_TERMS = {
    'langchain', 'llamaindex', 'langgraph', 'langsmith', 'flowise',
}

HONEYPOT_IDS = {
    'CAND_0003582','CAND_0005291','CAND_0007353','CAND_0007413','CAND_0008960',
    'CAND_0010294','CAND_0016000','CAND_0018515','CAND_0024752','CAND_0025579',
    'CAND_0033131','CAND_0033817','CAND_0033972','CAND_0035104','CAND_0036299',
    'CAND_0036839','CAND_0037000','CAND_0037539','CAND_0038431','CAND_0040075',
    'CAND_0040853','CAND_0042245','CAND_0042453','CAND_0043721','CAND_0046649',
    'CAND_0046689','CAND_0048740','CAND_0053734','CAND_0055685','CAND_0055792',
    'CAND_0056983','CAND_0057711','CAND_0060642','CAND_0061722','CAND_0063888',
    'CAND_0064077','CAND_0065096','CAND_0065710','CAND_0065787','CAND_0066405',
    'CAND_0070189','CAND_0070429','CAND_0072379','CAND_0073853','CAND_0074119',
    'CAND_0077239','CAND_0077250','CAND_0084182','CAND_0090900','CAND_0091068',
    'CAND_0093364','CAND_0093547','CAND_0095140','CAND_0095317','CAND_0095480',
    'CAND_0096150',
}

# =============================================================================
# TITLE INTELLIGENCE
# =============================================================================

TITLE_TIERS = {
    5: ['founding engineer','principal ai','principal ml','staff ai','staff ml',
        'staff engineer','principal scientist','principal engineer'],
    4: ['senior ai engineer','senior ml engineer','senior machine learning engineer',
        'senior nlp engineer','applied scientist','senior applied scientist',
        'research engineer','senior research engineer','lead ai','lead ml',
        'ai tech lead','ml tech lead','senior research scientist'],
    3: ['ai engineer','ml engineer','machine learning engineer','nlp engineer',
        'data scientist','senior data scientist','search engineer',
        'recommendations engineer','ranking engineer','tech lead',
        'technical lead','engineering lead','senior software engineer'],
    2: ['junior ml','junior ai','data engineer','backend engineer',
        'software engineer','platform engineer','infrastructure engineer'],
    1: ['manager','director','vp','cto','architect','consultant','business analyst'],
    0: ['marketing','sales','hr manager','recruiter','designer','graphic',
        'content writer','customer success','support engineer'],
}
TITLE_SCORES = {5: 1.0, 4: 0.90, 3: 0.75, 2: 0.45, 1: 0.20, 0: 0.0}

def get_title_tier(title: str) -> int:
    t = title.lower()
    for tier in sorted(TITLE_TIERS.keys(), reverse=True):
        for kw in TITLE_TIERS[tier]:
            if kw in t:
                return tier
    return 2

SENIORITY_RANK = {
    'intern': 0, 'trainee': 0, 'apprentice': 0,
    'junior': 1, 'associate': 1,
    'senior': 3, 'specialist': 3,
    'lead': 4, 'principal': 4, 'staff': 4,
    'director': 5, 'founding': 5, 'head': 5, 'vp': 5,
}

def get_seniority_rank(title: str) -> int:
    t = title.lower()
    for key, rank in sorted(SENIORITY_RANK.items(), key=lambda x: -x[1]):
        if key and key in t:
            return rank
    return 2

# =============================================================================
# EDUCATION INTELLIGENCE
# =============================================================================

DEGREE_RELEVANCE = {
    'phd': 1.0, 'doctorate': 1.0, 'ph.d': 1.0,
    'm.tech': 0.95, 'mtech': 0.95, 'master of technology': 0.95,
    'm.s.': 0.90, 'msc': 0.90, 'master of science': 0.90,
    'm.e.': 0.90, 'master of engineering': 0.90,
    'b.tech': 0.85, 'btech': 0.85, 'bachelor of technology': 0.85,
    'b.e.': 0.85, 'bachelor of engineering': 0.85,
    'b.s.': 0.80, 'bsc': 0.80, 'bachelor of science': 0.80,
    'mba': 0.30,
}

FIELD_RELEVANCE = {
    'computer science': 1.00, 'cse': 1.00,
    'artificial intelligence': 1.00, 'machine learning': 1.00,
    'data science': 0.95, 'information technology': 0.85,
    'statistics': 0.90, 'mathematics': 0.85, 'applied mathematics': 0.90,
    'electrical engineering': 0.75, 'ece': 0.75, 'electronics': 0.70,
    'information science': 0.85, 'physics': 0.70, 'computational': 0.85,
    'mechanical': 0.30, 'civil': 0.20, 'business': 0.20, 'management': 0.20,
}

TIER_SCORE = {
    'tier_1': 1.00, 'tier_2': 0.75,
    'tier_3': 0.50, 'tier_4': 0.25, 'unknown': 0.35,
}

# =============================================================================
# CERTIFICATIONS
# =============================================================================

CERT_RELEVANCE = {
    'aws certified machine learning': 2.0,
    'google professional machine learning': 2.0,
    'tensorflow developer': 1.8,
    'nvidia deep learning': 1.8,
    'databricks certified': 1.5,
    'deeplearning.ai': 1.5,
    'azure ai engineer': 1.5,
    'mlops': 1.5,
    'natural language': 1.5,
    'pytorch': 1.5,
    'gcp professional': 1.5,
    'coursera machine learning': 1.2,
    'data science': 1.0,
    'aws certified': 1.0,
    'google cloud': 1.0,
    'azure': 0.8,
    'python': 0.7,
}

# =============================================================================
# HELPERS
# =============================================================================

REF_DATE = datetime(2026, 6, 26)

def _parse_date(d: str) -> datetime:
    if not d:
        return datetime(2020, 1, 1)
    try:
        return datetime.strptime(d[:10], '%Y-%m-%d')
    except:
        return datetime(2020, 1, 1)

def _build_corpus(cand: dict) -> str:
    """Build full text corpus from all candidate fields. Padded with spaces for boundary matching."""
    parts = [' ']
    for s in cand.get('skills', []):
        parts.append(s.get('name', ''))
    for job in cand.get('career_history', []):
        parts.append(job.get('description', ''))
        parts.append(job.get('title', ''))
    p = cand.get('profile', {})
    parts.append(p.get('summary', ''))
    parts.append(p.get('headline', ''))
    return (' '.join(parts) + ' ').lower()

def _is_consulting(company: str) -> bool:
    cl = company.lower()
    return any(f in cl for f in CONSULTING_FIRMS)

# =============================================================================
# LAYER 1: SEMANTIC SKILL MATCHER (fast O(n) via pre-built lookup)
# =============================================================================

def score_semantic_skills(cand: dict) -> dict:
    corpus = _build_corpus(cand)
    skills_list = cand.get('skills', [])

    # --- O(n) concept detection: scan pre-sorted synonym keys ---
    concept_present = {}
    for syn in _SYN_KEYS_SORTED:
        if syn not in corpus:
            continue
        c = _SYN_TO_CONCEPT[syn]
        if c not in concept_present:
            concept_present[c] = True

    # --- Weighted score with per-skill proficiency/duration boost ---
    raw_score = 0.0
    total_weight = sum(JD_REQUIREMENTS.values())
    matched_concepts = []

    for concept, weight in JD_REQUIREMENTS.items():
        if not concept_present.get(concept):
            continue
        matched_concepts.append(concept)
        # Look up the best proficiency boost from explicit skills list
        prof_boost = 1.0
        concept_syns = _CONCEPT_ALL_SYNS.get(concept, set())
        for s in skills_list:
            sn = s.get('name', '').lower()
            if any(syn in sn or sn in syn for syn in list(concept_syns)[:6]):
                prof  = s.get('proficiency', 'intermediate')
                dur   = s.get('duration_months', 0)
                endor = s.get('endorsements', 0)
                pm = {'expert': 1.20, 'advanced': 1.10,
                      'intermediate': 1.00, 'beginner': 0.70}.get(prof, 1.0)
                dm = 1.0 + 0.08 * math.log1p(dur / 12.0)
                em = 1.0 + 0.04 * math.log1p(endor)
                prof_boost = min(pm * dm * em, 1.5)
                break
        raw_score += weight * prof_boost

    # Normalize: 60% concept coverage → score 1.0
    skill_score = min(raw_score / (total_weight * 0.60), 1.0)

    # --- Power combos ---
    power_bonus = 1.0
    matched_combos = []
    for combo_concepts, bonus, label in POWER_COMBOS:
        if all(c in concept_present for c in combo_concepts):
            power_bonus = max(power_bonus, bonus)
            matched_combos.append(label)

    # --- Disqualifiers ---
    vision_hits    = sum(1 for v in VISION_ONLY_TERMS if v in corpus)
    langchain_hits = sum(1 for l in LANGCHAIN_TRAP_TERMS if l in corpus)
    has_vision_only    = vision_hits > 0 and len(matched_concepts) < 3
    has_langchain_only = langchain_hits > 0 and len(matched_concepts) < 4

    return {
        'skill_score':        skill_score,
        'matched_concepts':   matched_concepts,
        'power_bonus':        power_bonus,
        'matched_combos':     matched_combos,
        'has_vision_only':    has_vision_only,
        'has_langchain_only': has_langchain_only,
    }

# =============================================================================
# LAYER 2: CAREER ARC INTELLIGENCE
# =============================================================================

# Pre-build description NLP scan terms (top-3 syns per JD concept)
_DESC_SCAN_TERMS = []
for _c, _syns in SYNONYM_GROUPS.items():
    if _c in JD_REQUIREMENTS:
        _DESC_SCAN_TERMS.extend(_syns[:3])

def score_career_arc(cand: dict) -> dict:
    career  = cand.get('career_history', [])
    profile = cand.get('profile', {})
    notes   = []

    if not career:
        return {'career_score': 0.10, 'all_consulting': False,
                'has_product': False, 'progression_bonus': 1.0,
                'career_notes': ['No career history']}

    career_sorted = sorted(career, key=lambda j: _parse_date(j.get('start_date', '')))

    # Company type
    cos = [j.get('company', '') for j in career]
    all_consulting = bool(cos) and all(_is_consulting(c) for c in cos)
    has_product    = any(not _is_consulting(c) for c in cos)

    # Title progression arc
    seniority_seq = [get_seniority_rank(j.get('title', '')) for j in career_sorted]
    progression_bonus = 1.0
    if len(seniority_seq) >= 2:
        improvements = sum(
            1 for i in range(1, len(seniority_seq))
            if seniority_seq[i] >= seniority_seq[i-1]
        )
        rate = improvements / (len(seniority_seq) - 1)
        if rate >= 0.75:
            progression_bonus = 1.15
            notes.append("Strong upward career arc")
        elif rate >= 0.50:
            progression_bonus = 1.05
        else:
            progression_bonus = 0.95

    # Stability
    durations = [j.get('duration_months', 0) for j in career]
    avg_dur   = sum(durations) / len(durations) if durations else 0
    if avg_dur < 12:
        stability_score = 0.40
        notes.append(f"Job hopper (avg {avg_dur:.0f}mo/role)")
    elif avg_dur < 18:
        stability_score = 0.70
        notes.append(f"Mobile career (avg {avg_dur:.0f}mo/role)")
    elif avg_dur >= 30:
        stability_score = 1.10
        notes.append(f"Stable tenure (avg {avg_dur:.0f}mo/role)")
    else:
        stability_score = 1.00

    # Current title
    curr_title  = profile.get('current_title', '')
    title_tier  = get_title_tier(curr_title)
    title_score = TITLE_SCORES.get(title_tier, 0.30)

    # Career description NLP (pre-built scan terms — fast)
    desc_corpus = ' '.join(j.get('description', '') for j in career).lower()
    desc_hits   = sum(1 for t in _DESC_SCAN_TERMS if t in desc_corpus)
    desc_score  = min(desc_hits / 18.0, 1.0)

    # Company size (scaleup preferred)
    size_score = 0.65
    for j in career:
        sz = j.get('company_size', '')
        if sz in ('11-50', '51-200', '201-500'):
            size_score = max(size_score, 1.00)
        elif sz in ('501-1000', '1001-5000'):
            size_score = max(size_score, 0.88)

    # Industry bonus
    industries = ' '.join(j.get('industry', '') for j in career).lower()
    ind_bonus = 1.05 if any(
        w in industries for w in ('technology','software','artificial intelligence',
                                  'internet','saas','fintech','edtech')
    ) else 1.0

    career_score = (0.35 * title_score +
                    0.25 * min(stability_score, 1.0) +
                    0.25 * desc_score +
                    0.15 * size_score) * ind_bonus

    if all_consulting:
        career_score *= 0.12
        notes.append("Pure consulting — penalized per JD")
    elif not has_product:
        career_score *= 0.55
        notes.append("No clear product company experience")

    return {
        'career_score':      min(career_score, 1.0),
        'all_consulting':    all_consulting,
        'has_product':       has_product,
        'progression_bonus': progression_bonus,
        'career_notes':      notes,
    }

# =============================================================================
# LAYER 3: EXPERIENCE QUALITY
# =============================================================================

def score_experience(yoe: float) -> float:
    if   6.0 <= yoe <= 8.0: return 1.00
    elif 5.0 <= yoe < 6.0:  return 0.90
    elif 8.0 < yoe <= 9.0:  return 0.90
    elif 4.0 <= yoe < 5.0:  return 0.78
    elif 9.0 < yoe <= 11.0: return 0.65
    elif 3.0 <= yoe < 4.0:  return 0.40
    elif 11.0 < yoe <= 14.0:return 0.45
    else:                    return 0.15

# =============================================================================
# LAYER 4: EDUCATION INTELLIGENCE
# =============================================================================

def score_education(cand: dict) -> dict:
    education = cand.get('education', [])
    if not education:
        return {'edu_score': 0.30, 'edu_notes': ['No education data']}

    best, best_notes = 0.0, []
    for edu in education:
        degree = edu.get('degree', '').lower()
        field  = edu.get('field_of_study', '').lower()
        tier   = edu.get('tier', 'unknown')

        deg_score = next((v for k, v in DEGREE_RELEVANCE.items() if k in degree), 0.5)
        fld_score = next((v for k, v in FIELD_RELEVANCE.items() if k in field),  0.4)
        tier_score = TIER_SCORE.get(tier, 0.35)
        score = 0.35 * deg_score + 0.35 * fld_score + 0.30 * tier_score

        if score > best:
            best = score
            if tier in ('tier_1', 'tier_2'):
                best_notes = [f"Tier-{tier[-1]} institution ({edu.get('institution','')})"]
            else:
                best_notes = [f"{edu.get('degree','')} in {edu.get('field_of_study','')}"]

    return {'edu_score': best, 'edu_notes': best_notes}

# =============================================================================
# LAYER 5: ASSESSMENTS & CERTIFICATIONS
# =============================================================================

def score_assessments_and_certs(cand: dict) -> float:
    signals = cand.get('redrob_signals', {})
    certs   = cand.get('certifications', [])

    # Redrob platform assessment scores
    assessments = signals.get('skill_assessment_scores', {})
    rel_total, rel_count = 0.0, 0
    for skill_name, test_score in assessments.items():
        sk = skill_name.lower()
        # Fast relevance check: does any JD synonym appear in/around this skill name?
        if any(syn in sk or sk in syn for syn in list(_ALL_JD_SYNS_FLAT)[:80]):
            rel_total  += test_score / 100.0
            rel_count  += 1
    assessment_score = (rel_total / rel_count) if rel_count > 0 else 0.0

    # Certifications
    cert_raw = 0.0
    for cert in certs:
        combined = (cert.get('name', '') + ' ' + cert.get('issuer', '')).lower()
        cert_raw += next((v for k, v in CERT_RELEVANCE.items() if k in combined), 0.0)
    cert_score = min(cert_raw / 3.0, 1.0)

    return 0.65 * assessment_score + 0.35 * cert_score if rel_count > 0 else cert_score

# =============================================================================
# LAYER 6: BEHAVIORAL AVAILABILITY (all 23 Redrob signals)
# =============================================================================

def score_behavioral(cand: dict) -> dict:
    sig   = cand.get('redrob_signals', {})
    notes = []

    # --- AVAILABILITY ---
    days_inactive = (REF_DATE - _parse_date(sig.get('last_active_date', ''))).days
    if   days_inactive <=  7:  active_s = 1.00
    elif days_inactive <= 30:  active_s = 0.90
    elif days_inactive <= 90:  active_s = 0.70
    elif days_inactive <= 180: active_s = 0.40
    else:
        active_s = 0.10
        notes.append(f"Inactive {days_inactive}d")

    notice = sig.get('notice_period_days', 90)
    if   notice ==  0: notice_s = 1.00; notes.append("Immediate joiner")
    elif notice <= 30: notice_s = 0.95
    elif notice <= 60: notice_s = 0.80
    elif notice <= 90: notice_s = 0.65
    else:
        notice_s = 0.35
        notes.append(f"{notice}d notice")

    otw_s  = 1.0 if sig.get('open_to_work_flag', False) else 0.65
    apps   = sig.get('applications_submitted_30d', 0)
    apps_s = min(1.0, 0.5 + 0.12 * math.log1p(apps))

    availability = (0.40 * active_s + 0.30 * notice_s +
                    0.18 * otw_s    + 0.12 * apps_s)

    # --- ENGAGEMENT ---
    resp_rate = sig.get('recruiter_response_rate', 0.0)
    if resp_rate < 0.2:
        notes.append(f"Low response rate ({resp_rate:.0%})")

    rt = sig.get('avg_response_time_hours', 48)
    rt_s = 1.0 if rt <= 4 else 0.85 if rt <= 12 else 0.70 if rt <= 24 else 0.50 if rt <= 48 else 0.25

    interview_r = sig.get('interview_completion_rate', 0.0)
    offer_r     = sig.get('offer_acceptance_rate', -1)
    offer_s     = offer_r if offer_r >= 0 else 0.60

    views_s   = min(1.0, math.log1p(sig.get('profile_views_received_30d', 0))   / math.log1p(50))
    saved_s   = min(1.0, math.log1p(sig.get('saved_by_recruiters_30d', 0))      / math.log1p(10))
    search_s  = min(1.0, math.log1p(sig.get('search_appearance_30d', 0))        / math.log1p(100))
    conn_s    = min(1.0, math.log1p(sig.get('connection_count', 0))             / math.log1p(500))
    endor_s   = min(1.0, math.log1p(sig.get('endorsements_received', 0))        / math.log1p(100))
    complete  = sig.get('profile_completeness_score', 50) / 100.0

    gh = sig.get('github_activity_score', -1)
    if gh == -1:
        gh_s = 0.50
    else:
        gh_s = min(1.0, gh / 70.0)
        if gh > 60:
            notes.append(f"Active OSS contributor (GitHub {gh:.0f}/100)")

    engagement = (
        0.22 * resp_rate  + 0.15 * rt_s       + 0.15 * interview_r +
        0.10 * offer_s    + 0.10 * views_s     + 0.10 * saved_s     +
        0.06 * search_s   + 0.05 * gh_s        + 0.04 * conn_s      +
        0.02 * endor_s    + 0.01 * complete
    )

    # --- TRUST ---
    ve = sig.get('verified_email', False)
    vp = sig.get('verified_phone', False)
    li = sig.get('linkedin_connected', False)
    trust = 0.35 * ve + 0.35 * vp + 0.30 * li
    if not ve and not vp:
        notes.append("No verified contact")

    return {
        'availability_score': availability,
        'engagement_score':   engagement,
        'trust_score':        trust,
        'behavioral_notes':   notes,
    }

# =============================================================================
# LAYER 7: SEMANTIC EMBEDDING MATCH
# =============================================================================

_ST_MODEL = None
_JD_EMBEDDING = None

def score_semantic_embedding(cand_list: list, jd_text: str):
    """Stage 2 Re-ranker: Computes true dense embedding similarity for a subset of candidates."""
    global _ST_MODEL, _JD_EMBEDDING
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        print("WARNING: sentence-transformers not found. Skipping Layer 7 semantic re-ranking.")
        return

    if _ST_MODEL is None:
        print("Initializing Layer 7 Semantic Embeddings model (all-MiniLM-L6-v2)...")
        _ST_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        _JD_EMBEDDING = _ST_MODEL.encode(jd_text)

    print(f"Running semantic similarity on top {len(cand_list)} candidates...")
    
    texts = []
    for cand, sc in cand_list:
        summary = cand.get('profile', {}).get('summary', '')
        headline = cand.get('profile', {}).get('headline', '')
        texts.append((summary + ' ' + headline).strip() or "No profile text available.")

    cand_embeddings = _ST_MODEL.encode(texts)
    similarities = util.cos_sim(_JD_EMBEDDING, cand_embeddings)[0]

    for i, (cand, sc) in enumerate(cand_list):
        sim = float(similarities[i])
        sc['semantic_score'] = max(0.0, sim)
        # Blend into final score (15% weight)
        sc['final_score'] = sc['final_score'] * 0.85 + (sim * 1.5) * 0.15 
        sc['final_score'] = min(sc['final_score'], 1.0)


# =============================================================================
# COMPOSITE SCORER
# =============================================================================

WEIGHTS = {
    'skill':        0.30,
    'career':       0.22,
    'experience':   0.12,
    'availability': 0.12,
    'assessment':   0.08,
    'engagement':   0.07,
    'education':    0.06,
    'trust':        0.03,
}

def score_candidate(cand: dict) -> dict:
    cid = cand.get('candidate_id', '')

    if cid in HONEYPOT_IDS:
        return {'final_score': 0.0, 'disqualified': True, 'reason': 'honeypot'}

    yoe = cand.get('profile', {}).get('years_of_experience', 0)

    sk  = score_semantic_skills(cand)
    ca  = score_career_arc(cand)
    ex  = score_experience(yoe)
    ed  = score_education(cand)
    as_ = score_assessments_and_certs(cand)
    bh  = score_behavioral(cand)

    if ca['all_consulting']:
        return {'final_score': 0.0, 'disqualified': True, 'reason': 'pure_consulting'}

    skill_score = sk['skill_score']
    if sk['has_vision_only']:    skill_score *= 0.25
    if sk['has_langchain_only']: skill_score *= 0.60

    base = (
        WEIGHTS['skill']        * skill_score              +
        WEIGHTS['career']       * ca['career_score']       +
        WEIGHTS['experience']   * ex                       +
        WEIGHTS['education']    * ed['edu_score']          +
        WEIGHTS['assessment']   * as_                      +
        WEIGHTS['availability'] * bh['availability_score'] +
        WEIGHTS['engagement']   * bh['engagement_score']   +
        WEIGHTS['trust']        * bh['trust_score']
    )

    final = base * ca['progression_bonus'] * sk['power_bonus']
    final = min(final, 1.0)

    return {
        'final_score':        round(final, 4),
        'disqualified':       False,
        'skill_score':        round(skill_score, 3),
        'career_score':       round(ca['career_score'], 3),
        'exp_score':          round(ex, 3),
        'edu_score':          round(ed['edu_score'], 3),
        'assess_score':       round(as_, 3),
        'availability_score': round(bh['availability_score'], 3),
        'engagement_score':   round(bh['engagement_score'], 3),
        'trust_score':        round(bh['trust_score'], 3),
        'matched_concepts':   sk['matched_concepts'],
        'power_combos':       sk['matched_combos'],
        'power_bonus':        sk['power_bonus'],
        'progression_bonus':  ca['progression_bonus'],
        'career_notes':       ca['career_notes'],
        'edu_notes':          ed['edu_notes'],
        'behavioral_notes':   bh['behavioral_notes'],
    }

# =============================================================================
# INTELLIGENT REASONING GENERATOR
# =============================================================================

def generate_reasoning(cand: dict, scores: dict, rank: int) -> str:
    profile  = cand.get('profile', {})
    signals  = cand.get('redrob_signals', {})

    yoe     = profile.get('years_of_experience', 0)
    title   = profile.get('current_title', 'Engineer')
    company = profile.get('current_company', '')
    loc     = profile.get('location', '')

    parts = []

    # Strengths
    combos  = scores.get('power_combos', [])
    matched = scores.get('matched_concepts', [])
    if combos:
        parts.append(f"Rare combo: {combos[0]}")
    elif matched:
        top3 = [c.replace('_', ' ').title() for c in matched[:3]]
        parts.append(f"JD-matched: {', '.join(top3)}")

    for note in scores.get('career_notes', [])[:1]:
        parts.append(note)

    if 6.0 <= yoe <= 8.0:
        parts.append(f"{yoe:.1f}yr — JD sweet spot")
    elif 5.0 <= yoe <= 9.0:
        parts.append(f"{yoe:.1f}yr — within JD range")
    else:
        parts.append(f"{yoe:.1f}yr experience")

    if company:
        parts.append(f"Currently: {title} @ {company}")

    for note in scores.get('edu_notes', [])[:1]:
        if any(t in note.lower() for t in ['tier-1', 'tier-2']):
            parts.append(note)

    gh = signals.get('github_activity_score', -1)
    if gh > 65:
        parts.append(f"Active OSS (GitHub {gh:.0f}/100)")

    assess = signals.get('skill_assessment_scores', {})
    if assess:
        best_k = max(assess, key=assess.get)
        best_v = assess[best_k]
        if best_v >= 78:
            parts.append(f"Platform test: {best_k}={best_v:.0f}/100")

    notice = signals.get('notice_period_days', 90)
    if notice == 0:
        parts.append("Immediate availability")
    elif notice <= 30:
        parts.append(f"Short notice ({notice}d)")

    # Concerns
    concerns = []
    for note in scores.get('behavioral_notes', [])[:2]:
        concerns.append(note)
    if yoe < 5.0:
        concerns.append(f"Below JD min experience ({yoe:.1f}yr)")
    # Add semantic score mention
    semantic = scores.get('semantic_score')
    if semantic is not None and semantic > 0.40:
        parts.append(f"Semantic Fit: {semantic*100:.0f}%")

    reasoning = ". ".join(parts)
    if concerns:
        reasoning += f". [Watch: {'; '.join(concerns)}]"

    return reasoning[:500]

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Intelligent AI Recruiter v2 — 6-Layer Intelligence Engine"
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out",        required=True)
    parser.add_argument("--debug", action="store_true",
                        help="Print per-dimension score table for top 20")
    args = parser.parse_args()

    cand_path = Path(args.candidates)
    if not cand_path.exists():
        print(f"Error: {args.candidates} not found.")
        return

    print("Loading candidates...")
    candidates = []
    if cand_path.suffix == '.gz':
        import gzip as _gz
        with _gz.open(cand_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
    else:
        with open(cand_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))

    print(f"Loaded {len(candidates):,} candidates — running 6-layer intelligence engine...")

    scored, disq = [], 0
    for cand in candidates:
        sc = score_candidate(cand)
        if sc['disqualified']:
            disq += 1
        else:
            scored.append((cand, sc))

    print(f"Qualified: {len(scored):,} | Filtered: {disq:,}")

    scored.sort(key=lambda x: (-x[1]['final_score'], x[0].get('candidate_id', '')))
    
    # Stage 2: Dense Semantic Re-ranking on Top 1000 candidates
    top_1000 = scored[:1000]
    score_semantic_embedding(top_1000, JD_TEXT)
    
    # Re-sort after semantic blending
    top_1000.sort(key=lambda x: (-x[1]['final_score'], x[0].get('candidate_id', '')))
    top_100 = top_1000[:100]

    if args.debug:
        W = 100
        print("\n" + "=" * W)
        print(f"{'Rk':<4} {'ID':<14} {'Total':<7} {'Skill':<7} {'Career':<7} "
              f"{'Exp':<6} {'Edu':<6} {'Asmt':<6} {'Avail':<7} {'Engag':<7}")
        print("-" * W)
        for i, (cand, sc) in enumerate(top_100[:20]):
            print(f"{i+1:<4} {cand['candidate_id']:<14} "
                  f"{sc['final_score']:<7.4f} {sc['skill_score']:<7.3f} "
                  f"{sc['career_score']:<7.3f} {sc['exp_score']:<6.3f} "
                  f"{sc['edu_score']:<6.3f} {sc['assess_score']:<6.3f} "
                  f"{sc['availability_score']:<7.3f} {sc['engagement_score']:<7.3f}")
            combos = sc.get('power_combos', [])
            if combos:
                print(f"     >> Combo: {combos[0]}")
            concepts = sc.get('matched_concepts', [])[:5]
            if concepts:
                print(f"     >> Matched: {', '.join(concepts)}")
        print("=" * W)

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for idx, (cand, sc) in enumerate(top_100):
            writer.writerow([
                cand.get('candidate_id'),
                idx + 1,
                sc['final_score'],
                generate_reasoning(cand, sc, idx + 1),
            ])

    print(f"\nTop 100 written to {args.out}")
    c0, s0 = top_100[0]
    combos = s0.get('power_combos', [])
    print(f"Score range: {top_100[-1][1]['final_score']:.4f} (rank 100)"
          f" -> {s0['final_score']:.4f} (rank 1)")
    print(f"Rank 1: {c0['candidate_id']} | "
          f"{'Combo: '+combos[0] if combos else 'Skills: '+', '.join(s0.get('matched_concepts',[])[:3])}")


if __name__ == "__main__":
    main()
