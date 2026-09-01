"""Ask an LLM whether an INDRA statement is supported by its evidence."""


ERROR_CATEGORIES = (
    "Entity Boundaries",
    "Grounding",
    "No Relation",
    "Wrong Relation",
    "Activity vs. Amount",
    "Polarity",
    "Negative Result",
    "Hypothesis",
    "Agent Conditions",
    "Modification Site",
    "Other",
)

CURATION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["accepted", "rejected", "uncertain"],
        },
        "reasoning": {"type": "string"},
        "error_category": {
            "anyOf": [
                {"type": "string", "enum": list(ERROR_CATEGORIES)},
                {"type": "null"},
            ]
        },
    },
    "required": ["decision", "reasoning", "error_category"],
    "additionalProperties": False,
}


def build_prompt(
    statement,
    evidence_text,
    abstract=None,
    uniprot_context=None,
    mesh_terms=None,
):
    """Build the essential part of curatogether's AI-curation prompt."""
    categories = "\n".join(f"- {category}" for category in ERROR_CATEGORIES)
    abstract_context = f"Abstract:\n{abstract}" if abstract else ""
    entity_context = (
        f"UniProt entity context (for grounding only):\n{uniprot_context}"
        if uniprot_context
        else ""
    )
    mesh_context = f"MeSH terms: {', '.join(mesh_terms)}" if mesh_terms else ""
    context_parts = [abstract_context, mesh_context, entity_context]
    context_parts = [part for part in context_parts if part]
    supporting_context = (
        "SUPPORTING CONTEXT\n" + "\n".join(context_parts)
        if context_parts
        else ""
    )
    context_rule = (
        "- Use supporting context only to resolve ambiguity, not to infer a relation."
        if context_parts
        else ""
    )

    return f"""You are an expert biological curator. Decide whether the INDRA
statement is supported by the evidence sentence.

STATEMENT
{statement}

EVIDENCE
{evidence_text}

{supporting_context}

RULES
- Judge only the relation expressed by this evidence.
{context_rule}
- Expression or amount changes do not by themselves prove activity changes.
- Check entity identity, relation type, polarity, negation, hypothesis language,
  experimental conditions, and modification sites.
- Use accepted when the extraction is supported, rejected when it is not, and
  uncertain only when the evidence is genuinely insufficient.
- When rejected, error_category must be one of:
{categories}

Return only this JSON shape:
{{
  "decision": "accepted | rejected | uncertain",
  "reasoning": "short explanation",
  "error_category": "category or null"
}}"""


def curate(
    client,
    statement,
    evidence_text,
    abstract=None,
    uniprot_context=None,
    mesh_terms=None,
):
    """Run one AI curation and return its structured decision."""
    prompt = build_prompt(
        statement, evidence_text, abstract, uniprot_context, mesh_terms
    )
    result, metadata = client.complete_json(
        [{"role": "user", "content": prompt}],
        name="indra_curation",
        schema=CURATION_SCHEMA,
    )

    decision = result.get("decision")
    if decision not in {"accepted", "rejected", "uncertain"}:
        raise ValueError(f"Invalid curation decision: {decision!r}")

    if decision == "rejected" and result.get("error_category") not in ERROR_CATEGORIES:
        raise ValueError("A rejected result must contain a valid error_category")

    return {
        "decision": decision,
        "reasoning": result.get("reasoning", ""),
        "error_category": result.get("error_category"),
        "model": client.model,
        **metadata,
    }
