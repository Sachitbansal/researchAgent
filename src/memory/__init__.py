"""
Conversation memory v1: short-term, per thread (docs/DECISIONS.md D12).

In:  nothing at import time. Out: `threads.ThreadStore` (SQLite persistence of a chat's
     turns) and `history` (turning stored turns into prior messages, and a run's trace
     into the one-line tool-call notes those messages carry).
"""
