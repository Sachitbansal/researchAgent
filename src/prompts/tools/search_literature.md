Search arXiv for papers on a topic, download them, and add them to the index. Call this
only when the user has approved a fetch; their message says so explicitly. It adds papers
to the corpus permanently, is slow (30–60s) and costs API calls. Do not use it to answer
questions about papers that are already indexed; use `retrieve_evidence` for that. Do
not call it twice for the same topic. Leave `categories` unset: the query planner already
chooses them, and yours would narrow its choice a second time.
