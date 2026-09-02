"""Curatogether's AI-curation workflow for a local LLM."""

from llm_belief.ai_curation import ERROR_CATEGORIES


CURATOGETHER_SCHEMA = {
    "type": "object",
    "properties": {
        "original": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "decision": {
                    "type": "string",
                    "enum": ["accepted", "rejected", "uncertain"],
                },
                "errorCategory": {
                    "anyOf": [
                        {"type": "string", "enum": list(ERROR_CATEGORIES)},
                        {"type": "null"},
                    ]
                },
            },
            "required": ["reasoning", "decision", "errorCategory"],
            "additionalProperties": False,
        },
    },
    "required": ["original"],
    "additionalProperties": False,
}


def _name(statement, attribute):
    agent = getattr(statement, attribute, None)
    return agent.name if agent is not None else None


def _statement_text(statement):
    statement_type = type(statement).__name__
    lower_type = statement_type.lower()

    if lower_type == "complex":
        members = [member.name for member in statement.members]
        subject = members[0] if members else "Unknown"
        object_name = " + ".join(members[1:]) or "Unknown"
    elif hasattr(statement, "enz") and hasattr(statement, "sub"):
        subject = _name(statement, "enz") or _name(statement, "sub") or "Unknown"
        object_name = _name(statement, "sub") or "Unknown"
    else:
        subject = _name(statement, "subj") or _name(statement, "enz") or "Unknown"
        object_name = _name(statement, "obj") or _name(statement, "sub") or "Unknown"

    relations = {
        "activation": "activates",
        "inhibition": "inhibits",
        "binding": "binds to",
        "complex": "forms complex with",
        "increaseamount": "increases the amount of",
        "decreaseamount": "decreases the amount of",
    }
    modifications = {
        "phosphorylation": ("phosphorylates", "is phosphorylated"),
        "dephosphorylation": ("dephosphorylates", "is dephosphorylated"),
        "glycosylation": ("glycosylates", "is glycosylated"),
        "ubiquitination": ("ubiquitinates", "is ubiquitinated"),
        "methylation": ("methylates", "is methylated"),
        "acetylation": ("acetylates", "is acetylated"),
        "palmitoylation": ("palmitoylates", "is palmitoylated"),
    }
    if lower_type in modifications:
        relation = modifications[lower_type][0 if _name(statement, "enz") else 1]
    else:
        relation = relations.get(lower_type, f'has relationship "{lower_type}" with')
    return f'"{subject} {relation} {object_name}"', statement_type


def build_prompt(statement, evidence, abstract=None, uniprot_context=None, mesh_terms=None):
    statement_text, statement_type = _statement_text(statement)
    position = getattr(statement, "position", None)
    residue = getattr(statement, "residue", None)
    details = f"Type: {statement_type}"
    if position:
        details += f" | Position: {position}"
    if residue:
        details += f" | Residue: {residue}"

    supporting = ""
    if abstract:
        supporting += f"Abstract:\n{abstract}\n"
    if mesh_terms:
        supporting += f"MeSH terms: {', '.join(mesh_terms)}\n"
    uniprot = ""
    if uniprot_context:
        uniprot = f"""
ENTITY NORMALIZATION CONTEXT (UniProt; optional)
We attached only the first UniProt search result for each statement gene token.
It may not be the exact entity mentioned in the evidence. Use it only as
tentative grounding context and do not infer relations from it.

{uniprot_context}
"""

    return f"""You are an expert biological curator evaluating whether a machine-extracted INDRA statement is supported by the provided evidence.

TASK
Decide whether the ORIGINAL statement is supported by the PRIMARY EVIDENCE.

STATEMENT (machine-extracted)
Text: {statement_text}
{details}

PRIMARY EVIDENCE (most important)
Evidence sentence:
{evidence}

SUPPORTING CONTEXT
{supporting}
{uniprot}

EVALUATION RULES (apply strictly)

A) Critical Distinctions
- Activity vs. Amount:
  * FUNCTIONAL evidence supports activates/inhibits.
  * EXPRESSION evidence alone does NOT support functional claims.
- Mutation logic:
  * Protein-to-Protein: mutant effects may invert implied wild-type function.
  * Protein-to-Disease: pathogenic variants causing disease can support activation of disease.

B) Error Categories (use when rejecting)
1 Entity Boundaries
2 Grounding
3 No Relation
4 Wrong Relation
5 Activity vs. Amount
6 Polarity
7 Negative Result
8 Hypothesis
9 Agent Conditions
10 Modification Site
11 Other

C) Relationship Flexibility
- Protein-to-Disease activation is valid if evidence indicates promotes, drives, or contributes to disease.

ANALYSIS INSTRUCTIONS
- Base the decision primarily on PRIMARY EVIDENCE.
- Use SUPPORTING CONTEXT only to resolve ambiguity, not to invent relations.
- Be explicit about whether evidence is functional vs expression.
- If uncertain, say what is missing.

Return only JSON:
{{
  "original": {{
    "reasoning": "short explanation",
    "decision": "accepted | rejected | uncertain",
    "errorCategory": "category or null"
  }}
}}"""


def curate_curatogether(
    client,
    statement,
    evidence_text,
    abstract=None,
    uniprot_context=None,
    mesh_terms=None,
):
    prompt = build_prompt(
        statement, evidence_text, abstract, uniprot_context, mesh_terms
    )
    result, metadata = client.complete_json(
        [{"role": "user", "content": prompt}],
        name="curatogether_curation",
        schema=CURATOGETHER_SCHEMA,
    )
    original = result.get("original", {})
    decision = original.get("decision")
    if decision not in {"accepted", "rejected", "uncertain"}:
        raise ValueError(f"Invalid curation decision: {decision!r}")

    error_category = original.get("errorCategory")
    if decision == "rejected" and error_category not in ERROR_CATEGORIES:
        raise ValueError("A rejected result must contain a valid errorCategory")

    return {
        "decision": decision,
        "reasoning": original.get("reasoning", ""),
        "error_category": error_category,
        "model": client.model,
        **metadata,
    }
