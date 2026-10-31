"""Generate CFPB-style synthetic bank complaint sample data for demo/training."""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample_complaints.json"

PRODUCTS = {
    "Checking or savings account": {
        "issues": [
            "Managing an account",
            "Problem caused by your funds being low",
            "Closing an account",
            "Unauthorized transactions",
        ],
        "templates": [
            "I deposited a paycheck of ${amount} on {date} but the funds still show as pending and I was charged overdraft fees.",
            "My checking account was frozen without clear notice after I reported a disputed ATM withdrawal of ${amount}.",
            "The bank closed my savings account ending in {last4} and mailed a check weeks late, causing bill payment failures.",
            "Unauthorized ACH debits totaling ${amount} posted to my account. I disputed them but the provisional credit was reversed.",
            "I requested account statements for the last 12 months and only received incomplete PDF files missing several months.",
        ],
        "resolutions": [
            "Refunded incorrect overdraft fees and restored provisional credit while investigation completed.",
            "Unfroze account after identity verification and documented hold reason in writing.",
            "Reissued closing balance check via expedited mail and waived related service fees.",
            "Completed Reg E investigation, permanently credited validated unauthorized transactions.",
            "Provided complete statement package and confirmed delivery through secure inbox.",
        ],
    },
    "Credit card or prepaid card": {
        "issues": [
            "Problem with a purchase shown on your statement",
            "Fees or interest",
            "Trouble using your card",
            "Advertising and marketing, including promotional offers",
        ],
        "templates": [
            "A merchant charge of ${amount} appears twice on my credit card ending in {last4}. The merchant says they only billed once.",
            "I was promised 0% APR for 15 months but interest of ${amount} was assessed in month two.",
            "My card was declined repeatedly while traveling even though my balance available was above ${amount}.",
            "A promotional balance transfer fee was disclosed as 3% but I was charged 5% on a ${amount} transfer.",
            "I returned merchandise on {date} and the credit never posted, leaving me paying interest on ${amount}.",
        ],
        "resolutions": [
            "Removed duplicate merchant charge and issued billing correction letter.",
            "Reversed improperly assessed interest and corrected APR promotional coding.",
            "Updated travel notes, replaced card, and confirmed authorization controls with customer.",
            "Refunded excess transfer fee and updated offer terms in account disclosures.",
            "Posted pending merchant credit and waived related interest for the dispute period.",
        ],
    },
    "Mortgage": {
        "issues": [
            "Trouble during payment process",
            "Struggling to pay mortgage",
            "Applying for a mortgage or refinancing an existing mortgage",
            "Closing on a mortgage",
        ],
        "templates": [
            "My mortgage payment of ${amount} was drafted twice in one month and customer service cannot locate the second payment.",
            "I applied for loss mitigation on {date} and have not received a complete decision package after multiple calls.",
            "Escrow analysis increased my payment by ${amount} without explaining tax or insurance changes.",
            "Refinance closing was delayed because the lender requested the same appraisal documents three times.",
            "I submitted a payoff quote request and received conflicting principal balances differing by ${amount}.",
        ],
        "resolutions": [
            "Reversed duplicate draft, refunded fees, and confirmed correct loan ledger balance.",
            "Issued complete loss-mitigation decision with appeal rights and single point of contact.",
            "Provided escrow analysis worksheet and adjusted payment after corrected tax data.",
            "Assigned dedicated closer, validated document checklist, and rescheduled closing.",
            "Issued corrected payoff quote and formal balance reconciliation letter.",
        ],
    },
    "Debt collection": {
        "issues": [
            "Attempts to collect debt not owed",
            "False statements or representation",
            "Communication tactics",
            "Took or threatened to take negative or legal action",
        ],
        "templates": [
            "A collector is demanding ${amount} for an account I already paid in full on {date}. They will not send validation.",
            "I received calls before 8am about a debt that does not match my credit report and may belong to someone else.",
            "The agency threatened wage garnishment without providing a judgment or verification letter.",
            "Despite a written cease-communication request, I continue to get daily calls about ${amount}.",
            "They reported a collection to the bureaus while an active dispute for identity theft is open.",
        ],
        "resolutions": [
            "Closed collection as paid/verified and sent zero-balance letter to customer and bureaus.",
            "Stopped outreach pending validation and corrected identity mismatch documentation.",
            "Provided validation package and withdrew premature legal-action language.",
            "Honored cease-communication request and moved correspondence to mail only.",
            "Suppressed bureau reporting until dispute investigation completed.",
        ],
    },
    "Money transfer, virtual currency, or money service": {
        "issues": [
            "Fraud or scam",
            "Money was not available when promised",
            "Other transaction problem",
            "Wrong amount charged or received",
        ],
        "templates": [
            "I sent ${amount} via wire on {date} to help a family emergency but the recipient never received funds.",
            "A peer-to-peer transfer of ${amount} was marked complete then reversed 48 hours later without explanation.",
            "I was scammed into authorizing a ${amount} transfer. I reported it immediately and need the investigation status.",
            "Currency conversion on my international transfer used a rate that overcharged me by roughly ${amount}.",
            "Cash pickup reference for ${amount} expired after the app showed successful funding.",
        ],
        "resolutions": [
            "Traced wire, confirmed beneficiary bank delay, and provided MT103 evidence to customer.",
            "Restored transfer after ledger reconciliation and clarified reversal reason in writing.",
            "Opened fraud case, froze related rails, and issued interim investigation update.",
            "Refunded conversion discrepancy and documented rate source used at execution.",
            "Reissued pickup instructions and extended reference validity after confirming funds.",
        ],
    },
    "Student loan": {
        "issues": [
            "Dealing with your lender or servicer",
            "Struggling to repay your loan",
            "Problem with a credit report or credit score",
            "Incorrect information on your report",
        ],
        "templates": [
            "I enrolled in an income-driven repayment plan but my bill still shows the standard payment of ${amount}.",
            "Forbearance request submitted on {date} was acknowledged then payments were still auto-debited.",
            "My servicer reports my student loan as delinquent even though payments posted on time.",
            "Consolidation paperwork was lost and I was asked to re-send the same tax forms multiple times.",
            "Employer certification for public service loan forgiveness has been pending far beyond the stated timeline.",
        ],
        "resolutions": [
            "Corrected IDR billing amount and issued payment history adjustment letter.",
            "Refunded auto-debit collected during approved forbearance and updated status.",
            "Furnished corrected payment history to credit bureaus and confirmed tradeline update.",
            "Completed consolidation with dedicated processor and confirmed new loan IDs.",
            "Expedited PSLF employer certification review and provided case reference number.",
        ],
    },
    "Credit reporting, credit repair services, or other personal consumer reports": {
        "issues": [
            "Incorrect information on your report",
            "Problem with a credit reporting company's investigation",
            "Improper use of your report",
            "Unable to get your credit report or credit score",
        ],
        "templates": [
            "My credit report still shows a charged-off card that was included in bankruptcy discharged on {date}.",
            "I disputed an inquiry I did not authorize and the bureau responded without contacting the furnisher properly.",
            "Personal information on my report lists an old address and an employer I never worked for.",
            "A collection account for ${amount} appears twice under two different creditor names for the same original debt.",
            "I cannot generate my free annual report online after repeated identity verification failures.",
        ],
        "resolutions": [
            "Updated tradeline to reflect discharge and confirmed bureau suppression of balance.",
            "Reopened dispute, re-verified with furnisher, and provided method-of-verification details.",
            "Corrected consumer informational fields after documentary proof review.",
            "Merged duplicate collection tradelines and issued updated report confirmation.",
            "Completed manual identity verification and mailed secure report copy.",
        ],
    },
}

COMPANY_RESPONSES = [
    "Closed with explanation",
    "Closed with non-monetary relief",
    "Closed with monetary relief",
    "In progress",
]

FIRST_NAMES = [
    "Alex", "Jordan", "Sam", "Taylor", "Riley", "Casey", "Avery", "Quinn", "Morgan", "Jamie",
    "Chris", "Pat", "Drew", "Cameron", "Noah", "Maya", "Priya", "Omar", "Elena", "Luis",
]


def build_dataset(n_per_product: int = 36, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    counter = 1000
    for product, meta in PRODUCTS.items():
        for i in range(n_per_product):
            issue = meta["issues"][i % len(meta["issues"])]
            template = meta["templates"][i % len(meta["templates"])]
            resolution = meta["resolutions"][i % len(meta["resolutions"])]
            amount = rng.choice([45, 75, 120, 250, 480, 750, 1250, 2400, 5600])
            last4 = f"{rng.randint(0, 9999):04d}"
            month = rng.randint(1, 12)
            day = rng.randint(1, 28)
            date = f"2025-{month:02d}-{day:02d}"
            name = rng.choice(FIRST_NAMES)
            narrative = (
                f"My name is {name}. "
                + template.format(amount=amount, date=date, last4=last4)
                + f" I have contacted support multiple times about this {issue.lower()} issue and need a clear written resolution."
            )
            # Light lexical variation
            if i % 3 == 0:
                narrative += " This has caused stress and late fees on other bills."
            elif i % 3 == 1:
                narrative += " Please investigate and respond with the next concrete steps."
            else:
                narrative += " I can provide statements and reference numbers upon request."

            counter += 1
            rows.append(
                {
                    "case_id": f"CX-{counter}",
                    "product": product,
                    "issue": issue,
                    "narrative": narrative,
                    "resolution_summary": resolution,
                    "company_response": rng.choice(COMPANY_RESPONSES),
                    "state": rng.choice(["CA", "TX", "NY", "FL", "IL", "WA", "GA", "NC"]),
                    "consumer_disputed": rng.choice(["Yes", "No", "N/A"]),
                }
            )
    rng.shuffle(rows)
    return rows


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = build_dataset()
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} complaints -> {OUT}")


if __name__ == "__main__":
    main()
