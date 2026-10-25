"""
Conversation memory: short-term per thread (D12) and long-term across chats (D19, D20).

In:  nothing at import time. Out: `threads.ThreadStore` (a chat's turns), `history` (stored
     turns as prior messages, plus tool-call notes), `memory_store.MemoryStore` (the
     memories table), `forgetting` (decay math) and `longterm.LongTermMemory` (read/write).
"""
