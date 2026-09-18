"""Curate whether an INDRA statement is supported by its evidence."""


ERROR_CATEGORIES = (
    "Entity Boundaries",
    "Grounding",
    "No Relation",
    "Wrong Relation",
    "Activity vs. Amount",
    "Polarity",
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

_CATEGORY_LIST = "\n".join(f"- {category}" for category in ERROR_CATEGORIES)
CURATION_INSTRUCTIONS = f"""You are an expert biological curator. Decide whether an
INDRA statement is supported by its evidence sentence.

RULES
- Before deciding, identify the statement's relation type, subject, and object,
  then compare the evidence with that exact meaning.
- Judge whether the evidence discusses the stated relation; positive experimental
  confirmation is not required. A relation may be tested, proposed, reduced,
  blocked, or unchanged, but its entities, direction, and relation type must match.
- Supporting context is only for resolving ambiguity, not for inferring a relation.
- Use UniProt gene/protein synonyms to match evidence names; synonyms identify
  entities but do not establish relations.
- For directed binary statements, Relation(A, B) means A affects or modifies B;
  evidence that B affects A does not match. Complex is symmetric and has no
  causal direction.
- Match relation type exactly. Promoter activity, transcription, expression,
  production, abundance, stability, or degradation supports
  IncreaseAmount/DecreaseAmount. Activation/Inhibition requires a change in the
  object's functional activity; expression, secretion, or localization alone is
  not activity.
- A statement may summarize an aggregate causal mechanism; direct biochemical
  contact is not required. In Phosphorylation(A, B), A may be an upstream cause
  and need not be the kinase if the evidence says A induces B phosphorylation.
  Compose polarity carefully: if inhibiting A blocks activation of B, A activates
  B; if A inhibits B phosphorylation, this can support Dephosphorylation(A, B).
- In modification statements, None means the enzyme or regulator is unspecified,
  not missing. Phosphorylation(None, X) is supported when X phosphorylation is
  discussed.
- Check mutation, modification, activity, location, and site conditions only when
  they are encoded in the statement. A general statement can be supported by
  evidence about a more specific form of the same entity.
- Use accepted when the extraction is supported, rejected when it is not, and
  uncertain only when the evidence is genuinely insufficient.
- When rejected, error_category must be one of:
{_CATEGORY_LIST}

Return only this JSON shape:
{{
  "decision": "accepted | rejected | uncertain",
  "reasoning": "short explanation",
  "error_category": "category or null"
}}"""


def statement_specific_instruction(statement):
    statement_type = type(statement).__name__
    if isinstance(statement, str):
        statement_type = statement.split("(", 1)[0]

    agents = statement.agent_list() if hasattr(statement, "agent_list") else []
    subject = agents[0].name if len(agents) > 0 and agents[0] else "the subject"
    object_ = agents[1].name if len(agents) > 1 and agents[1] else "the object"

    meanings = {
        "Activation": (
            f"{subject} -> {object_} functional activity (positive regulation). "
            "Expression or abundance alone does not match."
        ),
        "Inhibition": (
            f"{subject} -> {object_} functional activity (negative regulation). "
            "Reduced expression or abundance alone does not match."
        ),
        "IncreaseAmount": (
            f"{subject} -> {object_} amount, expression, production, or stability "
            "(positive regulation). Functional activity alone does not match."
        ),
        "DecreaseAmount": (
            f"{subject} -> {object_} amount, expression, production, or stability "
            "(negative regulation). Functional inhibition alone does not match."
        ),
    }
    return meanings.get(statement_type)


def build_prompt(
    statement,
    evidence_text,
    abstract=None,
    uniprot_context=None,
    mesh_terms=None,
):
    """Build the Gemma curation prompt."""
    abstract_context = f"Abstract:\n{abstract}" if abstract else ""
    entity_context = uniprot_context or ""
    mesh_context = f"MeSH terms: {', '.join(mesh_terms)}" if mesh_terms else ""
    context_parts = [abstract_context, mesh_context, entity_context]
    context_parts = [part for part in context_parts if part]
    supporting_context = (
        "SUPPORTING CONTEXT\n" + "\n".join(context_parts)
        if context_parts
        else ""
    )
    target_meaning = statement_specific_instruction(statement)
    target_section = (
        f"TARGET MEANING\n{target_meaning}" if target_meaning else ""
    )
    return f"""{CURATION_INSTRUCTIONS}

{supporting_context}

STATEMENT
{statement}

{target_section}

EVIDENCE
{evidence_text}"""


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
