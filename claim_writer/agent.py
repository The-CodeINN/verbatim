"""Entry point for ADK's dev UI: run `adk web` from the project root.

This is the same agent the app uses to draft cited statements, exposed as
`root_agent` so `adk web` can find it. It is for inspecting and debugging only
(ADK Web is documented as not for production); the real pipeline drives the
agent from verbatim/generate.py.

The agent expects JSON, not free text. Paste something like:

    {"question": "How long does the program last?",
     "evidence": [{"number": 1, "source": "brochure.pdf", "section": "",
                   "text": "Duration: 16 months"}]}

Its output is *unverified* here: in the app, every statement it proposes is then
checked against the source before it is shown (verbatim/verify.py).
"""

from dotenv import load_dotenv

from verbatim.generate import build_claim_agent

load_dotenv()

root_agent = build_claim_agent()
