"""Scenario-based evaluation of the support agent.

DeepEval sends anonymous usage telemetry by default. This project opts out
before DeepEval is imported anywhere.
"""

import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
