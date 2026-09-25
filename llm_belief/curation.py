"""Curate whether an INDRA statement is supported by its evidence."""


ERROR_CATEGORIES = (
    "entity_boundaries",
    "grounding",
    "no_relation",
    "wrong_relation",
    "act_vs_amt",
    "polarity",
    "agent_conditions",
    "mod_site",
    "hypothesis",
    "negative_result",
    "other",
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
  confirmation is not required.
- Use the hypothesis category only when speculative or prospective language
  applies to whether the stated relation itself occurs. Modal words alone are
  not sufficient; determine which claim they modify.
- Do not use hypothesis when the relation is asserted or presupposed and the
  uncertainty applies only to its mechanism, consequence, significance,
  context, or a downstream effect. An experimental purpose without an outcome
  is hypothesis only when the existence of the relation itself remains open.
- Merely stating that a relation was tested, analyzed, evaluated, measured, or
  assayed does not report an outcome. If no outcome is reported, use hypothesis.
- A relation can still be supported when the evidence says that the relation
  itself was reduced or blocked by another intervention. Evidence that A had no
  effect on B does not support either activation or inhibition; use the
  negative_result error category.
- Supporting context is only for resolving ambiguity, not for inferring a relation.
- Use UniProt context to validate each entity's biological sense, not only its
  name or abbreviation. If the same text refers to a different biological
  entity, reject it with the grounding error category.
- For directed binary statements, Relation(A, B) means A affects or modifies B;
  evidence that B affects A does not match. Match the relation type exactly and
  follow TARGET MEANING when it is provided.
- A directed causal or modification statement may summarize an aggregate
  mechanism; direct biochemical contact is not required. In Phosphorylation(A,
  B), A may be an upstream cause and need not be the kinase if the evidence says
  A induces B phosphorylation.
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
    members = ", ".join(agent.name for agent in agents if agent)

    meanings = {
        "Activation": (
            f"{subject} -> {object_} functional activity (positive regulation). "
            f"Classify by the property of {object_} that changes, not by whether "
            f"{subject} is active. Changes only to {object_} expression, abundance, "
            "protein or mRNA level, stability, or degradation are amount changes, "
            "not activity; reject them as act_vs_amt. Generic verbs such as "
            "'upregulates' do not establish activity by themselves."
        ),
        "Inhibition": (
            f"{subject} -> {object_} functional activity (negative regulation). "
            f"Classify by the property of {object_} that changes, not by whether "
            f"{subject} is active. Changes only to {object_} expression, abundance, "
            "protein or mRNA level, stability, or degradation are amount changes, "
            "not activity; reject them as act_vs_amt. Generic verbs such as "
            "'downregulates' or 'suppresses' do not establish activity by themselves."
        ),
        "IncreaseAmount": (
            f"{subject} -> {object_} amount, expression, production, or stability "
            f"(positive regulation). Classify by the property of {object_} that "
            f"changes. Changes only to {object_} activation state, catalytic "
            "activity, or functional output are activity changes, not amount; "
            "reject them as act_vs_amt."
        ),
        "DecreaseAmount": (
            f"{subject} -> {object_} amount, expression, production, or stability "
            f"(negative regulation). Classify by the property of {object_} that "
            f"changes. Changes only to {object_} activation state, catalytic "
            "activity, or functional output are activity changes, not amount; "
            "reject them as act_vs_amt."
        ),
        "Complex": (
            f"Physical binding or membership in one molecular complex containing "
            f"all listed members: {members or 'the listed members'}. Order does not "
            "matter. Co-expression, co-localization, functional association, or "
            "separate binding to the same third entity does not match."
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
