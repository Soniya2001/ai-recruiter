# AI Recruiter - Team Antigravity

[![Redrob Hackathon](https://img.shields.io/badge/Hackathon-Redrob-blue)](https://redrob.com)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

This repository contains Team Antigravity's submission for the Redrob Hackathon. It features a blazing-fast, 6-Layer Intelligence Engine designed to rank candidates based on nuanced job descriptions, semantic skill match, career trajectory, and behavioral signals.

## 🚀 The 6-Layer Intelligence Engine

Instead of relying on a slow, generic LLM-as-a-judge approach, our system compiles nuanced recruiter intuition into a highly optimized heuristic algorithm.

1. **Semantic Skill Matcher:** Uses pre-built ontologies to group synonyms and detect "Power Combos" (e.g., RAG + Embeddings + Vector Search).
2. **Career Arc Intelligence:** Rewards upward title progression and stability; penalizes job hopping and pure consulting backgrounds lacking product experience.
3. **Experience Quality:** Models the "sweet spot" of years of experience to ensure the candidate is neither too junior nor overly senior for the role.
4. **Education Intelligence:** Scores university tiers, degree types, and fields of study.
5. **Assessments & Certs:** Dynamically evaluates Redrob platform test scores mapped specifically to the required job skills.
6. **Behavioral Availability:** Fuses 23 distinct signals including notice period, GitHub activity, open-to-work flags, and recruiter response rates.

## ⚡ Performance

- **Lightning Fast:** Processes 100,000 candidates in ~15-25 seconds on a standard CPU.
- **Explainable:** Dynamically generates unique, fact-based, hallucination-free reasonings for the top candidates.
- **Robust Filter:** Blacklists 56 logically inconsistent honeypot candidates and penalizes applicants with anti-patterns (e.g., pure LangChain wrappers without core ML fundamentals).

## 🛠 Usage

To reproduce our rankings locally, you need the `candidates.jsonl` file from the project statement.

```bash
# Clone the repository
git clone https://github.com/Soniya2001/ai-recruiter.git
cd ai-recruiter

# Run the 6-Layer Intelligence Engine
python rank.py --candidates ./project_statement/candidates.jsonl --out ./team_antigravity.csv
```

To see a detailed breakdown of the scoring per dimension for the top candidates, you can use the `--debug` flag:
```bash
python rank.py --candidates ./project_statement/candidates.jsonl --out ./team_antigravity.csv --debug
```

## 👥 Team
- **B Soniya** - ML Engineer (soniyab2001@gmail.com)
