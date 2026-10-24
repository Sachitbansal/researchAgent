"""
The browser UI's server, as a package so its modules are imported as `web.<name>`.

In:  nothing at import time. Out: nothing; see web/api.py for the app and web/server.py
     for the entry point. A package rather than loose scripts because src/ already owns
     top-level names like `corpus`, and a bare `import corpus` here would shadow it.
"""
