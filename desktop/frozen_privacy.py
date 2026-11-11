"""File: desktop/frozen_privacy.py
Purpose: Set the process-only vendor telemetry opt-out before frozen application imports.
State: no global OS/environment modification, profile access or scientific imports.
"""

import os

os.environ["ORT_DISABLE_TELEMETRY"] = "1"
