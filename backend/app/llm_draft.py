"""Evidence-backed reply drafting via LLM with deterministic fallback."""

from __future__ import annotations

import logging
import re
from textwrap import dedent

from .config import Settings, get_settings
from .schemas import ClassificationResult, DraftReply, SimilarCase

logger = logging.getLogger(__name__)


class ReplyDrafter:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def ready(self) -> bool:
        provider = self._resolve_provider()
        return provider != "template"

    def _resolve_provider(self) -> str:
        choice = (self.settings.llm_provider or "auto").lower()
        if choice == "template":
            return "template"
        if choice == "groq" and self.settings.groq_api_key:
            return "groq"
        if choice == "openai" and self.settings.openai_api_key:
            return "openai"
        if choice == "auto":
            if self.settings.groq_api_key:
                return "groq"
            if self.settings.openai_api_key:
                return "openai"
        return "template"

    def draft(
        self,
        narrative: str,
        classification: ClassificationResult,
        similar_cases: list[SimilarCase],
        customer_name: str | None = None,
        account_last4: str | None = None,
        channel: str = "web",
    ) -> DraftReply:
        provider = self._resolve_provider()
        if provider == "template":
            return self._template_draft(
                narrative, classification, similar_cases, customer_name, account_last4
            )
        try:
            return self._llm_draft(
                provider=provider,
                narrative=narrative,
                classification=classification,
                similar_cases=similar_cases,
                customer_name=customer_name,
                account_last4=account_last4,
                channel=channel,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM draft failed (%s). Using template fallback.", exc)
            fallback = self._template_draft(
                narrative, classification, similar_cases, customer_name, account_last4
            )
            fallback.provider = f"{provider}-fallback-template"
            fallback.is_fallback = True
            return fallback

    def _build_evidence_block(self, similar_cases: list[SimilarCase]) -> str:
        if not similar_cases:
            return "No similar historical cases retrieved."
        lines: list[str] = []
        for idx, case in enumerate(similar_cases, start=1):
            lines.append(
                dedent(
                    f"""
                    Case {idx} [{case.case_id}] (similarity={case.similarity:.2f})
                    Product: {case.product}
                    Issue: {case.issue}
                    Narrative excerpt: {case.narrative[:280]}
                    Prior resolution: {case.resolution_summary}
                    Prior company response style: {case.company_response}
                    """
                ).strip()
            )
        return "\n\n".join(lines)

    def _llm_draft(
        self,
        provider: str,
        narrative: str,
        classification: ClassificationResult,
        similar_cases: list[SimilarCase],
        customer_name: str | None,
        account_last4: str | None,
        channel: str,
    ) -> DraftReply:
        from openai import OpenAI

        if provider == "groq":
            client = OpenAI(
                api_key=self.settings.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
            )
            model = self.settings.groq_model
        else:
            client = OpenAI(api_key=self.settings.openai_api_key)
            model = self.settings.openai_model

        evidence = self._build_evidence_block(similar_cases)
        name = customer_name or "Valued Customer"
        account = account_last4 or "****"

        system_prompt = dedent(
            """
            You are a senior banking customer-care specialist drafting a first-response email.
            Rules:
            - Be empathetic, clear, and professional.
            - Ground the response in the provided similar past cases and their resolutions.
            - Do NOT invent account balances, refund amounts, legal conclusions, or policy guarantees.
            - Do NOT claim a final decision; this is a draft for human agent review.
            - Acknowledge the classified product/issue.
            - Outline next investigation steps and expected timeline (3-5 business days).
            - Keep the body under 280 words.
            - Return plain text with:
              SUBJECT: ...
              BODY:
              ...
            """
        ).strip()

        user_prompt = dedent(
            f"""
            Customer name: {name}
            Account last4: {account}
            Intake channel: {channel}
            Predicted product category: {classification.product} (confidence={classification.confidence:.2f})

            Customer complaint:
            {narrative}

            Similar past cases and resolutions:
            {evidence}
            """
        ).strip()

        completion = client.chat.completions.create(
            model=model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = (completion.choices[0].message.content or "").strip()
        subject, body = self._parse_subject_body(content, classification.product)
        return DraftReply(
            subject=subject,
            body=body,
            tone="empathetic-professional",
            grounded_case_ids=[c.case_id for c in similar_cases],
            provider=provider,
            is_fallback=False,
        )

    def _parse_subject_body(self, content: str, product: str) -> tuple[str, str]:
        subject_match = re.search(r"SUBJECT:\s*(.+)", content, flags=re.IGNORECASE)
        body_match = re.search(r"BODY:\s*(.*)", content, flags=re.IGNORECASE | re.DOTALL)
        subject = (
            subject_match.group(1).strip()
            if subject_match
            else f"Update on your {product} complaint"
        )
        body = body_match.group(1).strip() if body_match else content
        return subject, body

    def _template_draft(
        self,
        narrative: str,
        classification: ClassificationResult,
        similar_cases: list[SimilarCase],
        customer_name: str | None,
        account_last4: str | None,
    ) -> DraftReply:
        name = customer_name or "Valued Customer"
        account = account_last4 or "****"
        top_case = similar_cases[0] if similar_cases else None
        evidence_lines = []
        for case in similar_cases[:3]:
            evidence_lines.append(
                f"- Historical case {case.case_id} ({case.issue}): {case.resolution_summary}"
            )
        evidence_block = "\n".join(evidence_lines) if evidence_lines else (
            "- No closely matching historical case was retrieved; a specialist will review manually."
        )
        resolution_hint = (
            top_case.resolution_summary
            if top_case
            else "complete a full account and policy review"
        )
        body = dedent(
            f"""
            Dear {name},

            Thank you for contacting us about your recent experience related to
            {classification.product.lower()}. We understand how frustrating this is and we are
            treating your complaint as a priority.

            We have classified this matter as "{classification.product}" with model confidence
            {classification.confidence:.0%}. Our specialist team will investigate the details you
            shared about account ending in {account}, including the circumstances described in your
            message.

            Based on similar past cases, the usual next steps are to:
            1. Verify transaction/account activity tied to your report
            2. Compare your situation with precedent resolutions (for example: {resolution_hint})
            3. Confirm applicable policy and regulatory response requirements
            4. Follow up with a written update within 3-5 business days

            Precedent notes used to ground this draft:
            {evidence_block}

            This is a draft for internal review only. It is not a final decision or refund approval.

            Sincerely,
            Customer Care Team
            """
        ).strip()
        # Keep narrative referenced lightly for agent context without dumping full text
        _ = narrative
        return DraftReply(
            subject=f"We are reviewing your {classification.product} complaint",
            body=body,
            tone="empathetic-professional",
            grounded_case_ids=[c.case_id for c in similar_cases],
            provider="template",
            is_fallback=True,
        )

