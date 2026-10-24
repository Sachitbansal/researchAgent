/* SAMPLE chats, shown so the UI can be seen working before anyone has asked anything.
   In:  nothing. Out: three hand-written chats, each labelled "Sample" in the sidebar.
   The questions and answers are written by hand, NOT produced by the agent. The chunk ids,
   excerpts, paper ids and topic tags are real, copied from data/index/chunks.jsonl by a one-off
   script, so the citations resolve to real papers through /api/papers. They are fixtures,
   never stored as threads and never sent to the agent; real chats are threads the server
   stores (docs/DECISIONS.md D12).
   Topic names appear here only because this fixture describes the shipped corpus; nothing
   else under frontend/src may name a topic. Delete this file and its one import in
   App.tsx to remove the samples. */

import type { Chat } from "../types";

export const SAMPLE_CHATS: Chat[] = [
  {
    "id": "sample-1",
    "title": "How does SparseMixer estimate gradients for the router when training sparse MoE models?",
    "sample": true,
    "created_at": 1761300000,
    "updated_at": 1761300000,
    "messages": [
      {
        "role": "user",
        "content": "How does SparseMixer estimate gradients for the router when training sparse MoE models?",
        "ts": 1761300000
      },
      {
        "role": "assistant",
        "status": "answered",
        "content": "SparseMixer treats expert routing as a discrete choice and estimates the router's gradient without differentiating through the sampling step itself [2310_00811v1__c0015].\n\n- **Two estimators, mixed.** It combines a second-order, mid-point estimate with a first-order one, switching between them depending on whether the selected expert is the router's arg-max choice [2310_00811v1__c0016].\n- **Cheap to run.** The estimator needs no Hessian or other second-order derivatives, so the overhead is negligible, and it drops into PyTorch alongside existing training code [2310_00811v1__c0016].\n- **Effect on training.** Switch Transformer with SparseMixer converges faster and scores better downstream than plain Switch in every setting the authors ran, and beats the dense baseline in all four, where plain Switch only did so with two experts [2310_00811v1__c0023].\n\nThe paper attributes the gain to better-trained routers rather than to more capacity [2310_00811v1__c0023].",
        "steps": [
          {
            "kind": "thinking",
            "iteration": 0,
            "t_ms": 0
          },
          {
            "kind": "tool_start",
            "iteration": 0,
            "tool_name": "retrieve_evidence",
            "args": {
              "query": "SparseMixer router gradient estimation for sparse mixture of experts",
              "k": 8,
              "topic_filter": "sparse_mixture_of_experts_routing_in_transformer_models"
            },
            "t_ms": 2100
          },
          {
            "kind": "tool_end",
            "iteration": 0,
            "tool_name": "retrieve_evidence",
            "latency_ms": 1870,
            "result": {
              "chunks": [
                {
                  "chunk_id": "2310_00811v1__c0016",
                  "paper_id": "2310_00811v1",
                  "score": 7.81,
                  "chunk_type": "text",
                  "text": "Accordingly, we propose SparseMixer to balance router training and expert training: b∇SparseMixer := (1 −δD)b∇SparseMixer-2rd + δD b∇SparseMixer-1st , where δD = (1, if D = arg max Ii πIi 0, otherwise"
                },
                {
                  "chunk_id": "2310_00811v1__c0015",
                  "paper_id": "2310_00811v1",
                  "score": 6.92,
                  "chunk_type": "text",
                  "text": "Similarly, it is easy to note: ∇0 mid−point ≈ X Ii g′(πIifIi ) · πIifIi · ∂πIi ∂Wr = ED∼π[2 · ∂g( πDfD(x) ) ∂Wr ] = E[b∇SparseMixer-2rd ]. Notably, it is feasible to employ more advanced ODE solvers l"
                },
                {
                  "chunk_id": "2310_00811v1__c0023",
                  "paper_id": "2310_00811v1",
                  "score": 5.4,
                  "chunk_type": "text",
                  "text": "Regarding both convergence speed and downstream performance, Switch+SparseMixer consistently outperforms Switch in all settings. Also, similar to the experiments on machine translation, we observe tha"
                },
                {
                  "chunk_id": "2310_00811v1__c0018",
                  "paper_id": "2310_00811v1",
                  "score": 3.12,
                  "chunk_type": "text",
                  "text": "In our experiments, we observe that such sparse distribution plays a crucial role in the success of MoE (as elaborated in Section 4.2). Applying SparseMixer to Switch Transformer. Correspondingly, we "
                }
              ],
              "sufficient_evidence": true,
              "note": "4 passage(s) cleared the relevance gate",
              "n_candidates_considered": 40
            },
            "t_ms": 3970
          },
          {
            "kind": "thinking",
            "iteration": 1,
            "t_ms": 3970
          },
          {
            "kind": "tool_start",
            "iteration": 1,
            "tool_name": "check_evidence_consistency",
            "args": {
              "mode": "groundedness",
              "chunk_ids": [
                "2310_00811v1__c0016",
                "2310_00811v1__c0015",
                "2310_00811v1__c0023"
              ],
              "answer_text": "SparseMixer treats expert routing as a discrete choice and estimates the router's gradient without differentiating through the sampling step itself [2310_00811v1__c0015].\n\n- **Two estimators, mixed.**"
            },
            "t_ms": 10370
          },
          {
            "kind": "tool_end",
            "iteration": 1,
            "tool_name": "check_evidence_consistency",
            "latency_ms": 2420,
            "result": {
              "claims": [
                {
                  "claim": "It combines a second-order, mid-point estimate with a first-order one.",
                  "label": "entailed",
                  "best_supporting_chunk": "2310_00811v1__c0016",
                  "confidence": 0.91
                },
                {
                  "claim": "The estimator needs no Hessian or other second-order derivatives.",
                  "label": "entailed",
                  "best_supporting_chunk": "2310_00811v1__c0015",
                  "confidence": 0.91
                },
                {
                  "claim": "Switch Transformer with SparseMixer converges faster than plain Switch.",
                  "label": "entailed",
                  "best_supporting_chunk": "2310_00811v1__c0023",
                  "confidence": 0.91
                },
                {
                  "claim": "The paper attributes the gain to better-trained routers.",
                  "label": "neutral",
                  "best_supporting_chunk": "2310_00811v1__c0016",
                  "confidence": 0.48
                }
              ],
              "grounded_ratio": 0.75,
              "unsupported_claims": [
                "The paper attributes the gain to better-trained routers."
              ],
              "note": "Advisory only: this is a general MNLI model and is weak on scientific text."
            },
            "t_ms": 12790
          },
          {
            "kind": "thinking",
            "iteration": 2,
            "t_ms": 12790
          },
          {
            "kind": "answering",
            "iteration": 2,
            "t_ms": 17990
          }
        ],
        "stats": {
          "iterations": 3,
          "tool_calls": 2,
          "context_tokens": 6912,
          "stopped_because": "answered",
          "run_id": null
        },
        "ts": 1761300017.99,
        "elapsed_ms": 17990
      }
    ]
  },
  {
    "id": "sample-2",
    "title": "What causes expert representation collapse in sparse MoE, and how do recent papers try to prevent it?",
    "sample": true,
    "created_at": 1761294600,
    "updated_at": 1761294600,
    "messages": [
      {
        "role": "user",
        "content": "What causes expert representation collapse in sparse MoE, and how do recent papers try to prevent it?",
        "ts": 1761294600
      },
      {
        "role": "assistant",
        "status": "answered",
        "content": "Representation collapse is when expert outputs become overly similar, or a handful of experts dominate, which excessive sparsity in routing can cause [2607_16427v1__c0003].\n\nThe papers group the fixes roughly two ways:\n\n| Approach | Examples | Mechanism |\n|---|---|---|\n| Diversify expert activation | SMOE-Dropout, X-MoE, SimSMOE | stochastic gating with frozen routers, hyperspherical projection, cosine/CKA redundancy penalties [2607_16427v1__c0006] |\n| Stabilise routing over time and context | StableMoE, HyperRouter, RMOE | freezing the router after a balanced warm-up, hypernetwork-generated routing, GRU memory across layers [2607_16427v1__c0006][2607_16427v1__c0003] |\n\nMore recent work adds context: relational load balancing from token similarity, and cross-layer coherence, which the MCF-MoE authors read as a need for routing that is both stable and semantically aligned [2607_16427v1__c0007].\n\nAll of these are described in one paper's related-work survey, so this is that paper's framing of the field rather than a comparison measured across the corpus.",
        "steps": [
          {
            "kind": "thinking",
            "iteration": 0,
            "t_ms": 0
          },
          {
            "kind": "tool_start",
            "iteration": 0,
            "tool_name": "retrieve_evidence",
            "args": {
              "query": "expert representation collapse mitigation routing",
              "k": 8,
              "topic_filter": "sparse_mixture_of_experts_routing_in_transformer_models"
            },
            "t_ms": 2100
          },
          {
            "kind": "tool_end",
            "iteration": 0,
            "tool_name": "retrieve_evidence",
            "latency_ms": 1650,
            "result": {
              "chunks": [
                {
                  "chunk_id": "2607_16427v1__c0006",
                  "paper_id": "2607_16427v1",
                  "score": 8.44,
                  "chunk_type": "text",
                  "text": "2.1 Mitigating Expert Representation Collapse To mitigate the long-standing issue of representation collapse in SMOE (Shazeer et al., 2017; Fedus et al., 2022) architectures, recent studies address it"
                },
                {
                  "chunk_id": "2607_16427v1__c0003",
                  "paper_id": "2607_16427v1",
                  "score": 7.97,
                  "chunk_type": "text",
                  "text": "However, excessive sparsity can lead to representation collapse, where expert outputs become overly similar or dominated by a few experts (Chi et al., 2022; Do et al., 2025). To address this issue, se"
                },
                {
                  "chunk_id": "2607_16427v1__c0007",
                  "paper_id": "2607_16427v1",
                  "score": 6.15,
                  "chunk_type": "text",
                  "text": "Recent work addresses these challenges by enhancing stability and context sensitivity. Omi et al. (Omi et al., 2025) leverages token similarity for relational load balancing. RMOE (Qiu et al., 2025) e"
                },
                {
                  "chunk_id": "2607_16427v1__c0002",
                  "paper_id": "2607_16427v1",
                  "score": 2.31,
                  "chunk_type": "text",
                  "text": "Scaling dense Transformer models quickly leads to prohibitive training and inference costs (Wang et al., 2024). Mixture-of-Experts (MoE) alleviates this issue by activating only a subset of experts pe"
                },
                {
                  "chunk_id": "2310_00811v1__c0018",
                  "paper_id": "2310_00811v1",
                  "score": 0.84,
                  "chunk_type": "text",
                  "text": "In our experiments, we observe that such sparse distribution plays a crucial role in the success of MoE (as elaborated in Section 4.2). Applying SparseMixer to Switch Transformer. Correspondingly, we "
                }
              ],
              "sufficient_evidence": true,
              "note": "5 passage(s) cleared the relevance gate",
              "n_candidates_considered": 40
            },
            "t_ms": 3750
          },
          {
            "kind": "thinking",
            "iteration": 1,
            "t_ms": 3750
          },
          {
            "kind": "tool_start",
            "iteration": 1,
            "tool_name": "check_evidence_consistency",
            "args": {
              "mode": "groundedness",
              "chunk_ids": [
                "2607_16427v1__c0006",
                "2607_16427v1__c0003",
                "2607_16427v1__c0007"
              ],
              "answer_text": "Representation collapse is when expert outputs become overly similar, or a handful of experts dominate, which excessive sparsity in routing can cause [2607_16427v1__c0003].\n\nThe papers group the fixes"
            },
            "t_ms": 10150
          },
          {
            "kind": "tool_end",
            "iteration": 1,
            "tool_name": "check_evidence_consistency",
            "latency_ms": 2980,
            "result": {
              "claims": [
                {
                  "claim": "Excessive sparsity can lead to representation collapse.",
                  "label": "entailed",
                  "best_supporting_chunk": "2607_16427v1__c0006",
                  "confidence": 0.91
                },
                {
                  "claim": "StableMoE freezes the router after a balanced warm-up.",
                  "label": "entailed",
                  "best_supporting_chunk": "2607_16427v1__c0003",
                  "confidence": 0.91
                },
                {
                  "claim": "Recent work adds relational load balancing from token similarity.",
                  "label": "entailed",
                  "best_supporting_chunk": "2607_16427v1__c0007",
                  "confidence": 0.91
                }
              ],
              "grounded_ratio": 1.0,
              "unsupported_claims": [],
              "note": "Advisory only: this is a general MNLI model and is weak on scientific text."
            },
            "t_ms": 13130
          },
          {
            "kind": "thinking",
            "iteration": 2,
            "t_ms": 13130
          },
          {
            "kind": "answering",
            "iteration": 2,
            "t_ms": 18330
          }
        ],
        "stats": {
          "iterations": 3,
          "tool_calls": 2,
          "context_tokens": 7480,
          "stopped_because": "answered",
          "run_id": null
        },
        "ts": 1761294618.33,
        "elapsed_ms": 18330
      }
    ]
  },
  {
    "id": "sample-3",
    "title": "Which structured pruning methods here question how importance is measured, and what do they do instead?",
    "sample": true,
    "created_at": 1761289200,
    "updated_at": 1761289200,
    "messages": [
      {
        "role": "user",
        "content": "Which structured pruning methods here question how importance is measured, and what do they do instead?",
        "ts": 1761289200
      },
      {
        "role": "assistant",
        "status": "answered",
        "content": "Three of the pruning papers explicitly target the weak point of deciding *what* to remove:\n\n1. **Calibration data.** Taylor-expansion pruning depends on gradients from calibration data, and randomly chosen samples lack representativeness and diversity, giving inaccurate importance estimates; the sample-aware method selects that data adaptively instead [2503_06184v1__c0003].\n2. **Not deleting outright.** TRSP argues that pruning straight from an importance metric causes knowledge loss and needs heavy retraining, because even unimportant layers can hold valuable knowledge; it regularises layer outputs in two stages before removing them [2505_18232v4__c0001][2505_18232v4__c0003].\n3. **Fixing the score itself.** Carrying Adaptive Feature Retention over to structured pruning runs into mismatched score distributions, lost sign information and outliers; the authors use a power transformation and sign-preserving aggregation to fix them [2607_08027v1__c0001][2607_08027v1__c0003].\n\nNone of them compares against the others directly, so the corpus does not say which works best.",
        "steps": [
          {
            "kind": "thinking",
            "iteration": 0,
            "t_ms": 0
          },
          {
            "kind": "tool_start",
            "iteration": 0,
            "tool_name": "retrieve_evidence",
            "args": {
              "query": "structured pruning importance criteria calibration data knowledge loss",
              "k": 8,
              "topic_filter": "structured_pruning_and_quantization_for_efficient_transforme"
            },
            "t_ms": 2100
          },
          {
            "kind": "tool_end",
            "iteration": 0,
            "tool_name": "retrieve_evidence",
            "latency_ms": 1990,
            "result": {
              "chunks": [
                {
                  "chunk_id": "2503_06184v1__c0003",
                  "paper_id": "2503_06184v1",
                  "score": 6.73,
                  "chunk_type": "text",
                  "text": "Hence, the Taylor expansion-based methodologies for structured pruning LLMs pose two notable challenges. Acquiring high-quality calibration data covering diverse parameters and sample spaces is crucia"
                },
                {
                  "chunk_id": "2505_18232v4__c0001",
                  "paper_id": "2505_18232v4",
                  "score": 6.4,
                  "chunk_type": "text",
                  "text": "The deployment of large language models (LLMs) is largely hindered by their large number of parameters. Structural pruning has emerged as a promising solution. Prior structured pruning methods directl"
                },
                {
                  "chunk_id": "2607_08027v1__c0001",
                  "paper_id": "2607_08027v1",
                  "score": 6.02,
                  "chunk_type": "text",
                  "text": "This paper proposes an improved structured pruning method for large language models (LLMs) that addresses key challenges in adapting Adaptive Feature Retention (AFR), an unstructured pruning technique"
                },
                {
                  "chunk_id": "2505_18232v4__c0003",
                  "paper_id": "2505_18232v4",
                  "score": 5.58,
                  "chunk_type": "text",
                  "text": "However, existing layer-wise pruning methods have a certain limitation. They consistently first compute the importance of each transformer layer using a designed criteria, prune unimportant layers, an"
                },
                {
                  "chunk_id": "2607_08027v1__c0003",
                  "paper_id": "2607_08027v1",
                  "score": 5.11,
                  "chunk_type": "text",
                  "text": "When adapting AFR to structured pruning by aggregating weight-level scores to neuron-level scores, several challenges arise: distribution mismatch between heterogeneous scores (ReFer exhibits wide val"
                },
                {
                  "chunk_id": "2503_06184v1__c0002",
                  "paper_id": "2503_06184v1",
                  "score": 3.9,
                  "chunk_type": "text",
                  "text": "Copyright © 2025, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved. Structured pruning (Xia et al. 2024) emerges as a pivotal technique for mitigating res"
                }
              ],
              "sufficient_evidence": true,
              "note": "6 passage(s) cleared the relevance gate",
              "n_candidates_considered": 40
            },
            "t_ms": 4090
          },
          {
            "kind": "thinking",
            "iteration": 1,
            "t_ms": 4090
          },
          {
            "kind": "tool_start",
            "iteration": 1,
            "tool_name": "check_evidence_consistency",
            "args": {
              "mode": "groundedness",
              "chunk_ids": [
                "2503_06184v1__c0003",
                "2505_18232v4__c0001",
                "2607_08027v1__c0001",
                "2505_18232v4__c0003",
                "2607_08027v1__c0003"
              ],
              "answer_text": "Three of the pruning papers explicitly target the weak point of deciding *what* to remove:\n\n1. **Calibration data.** Taylor-expansion pruning depends on gradients from calibration data, and randomly c"
            },
            "t_ms": 10490
          },
          {
            "kind": "tool_end",
            "iteration": 1,
            "tool_name": "check_evidence_consistency",
            "latency_ms": 3310,
            "result": {
              "claims": [
                {
                  "claim": "Randomly chosen calibration samples lack representativeness and diversity.",
                  "label": "entailed",
                  "best_supporting_chunk": "2503_06184v1__c0003",
                  "confidence": 0.91
                },
                {
                  "claim": "Even unimportant layers can hold valuable knowledge.",
                  "label": "entailed",
                  "best_supporting_chunk": "2505_18232v4__c0001",
                  "confidence": 0.91
                },
                {
                  "claim": "The authors use a power transformation and sign-preserving aggregation.",
                  "label": "neutral",
                  "best_supporting_chunk": "2607_08027v1__c0001",
                  "confidence": 0.48
                }
              ],
              "grounded_ratio": 0.667,
              "unsupported_claims": [
                "The authors use a power transformation and sign-preserving aggregation."
              ],
              "note": "Advisory only: this is a general MNLI model and is weak on scientific text."
            },
            "t_ms": 13800
          },
          {
            "kind": "thinking",
            "iteration": 2,
            "t_ms": 13800
          },
          {
            "kind": "answering",
            "iteration": 2,
            "t_ms": 19000
          }
        ],
        "stats": {
          "iterations": 3,
          "tool_calls": 2,
          "context_tokens": 8155,
          "stopped_because": "answered",
          "run_id": null
        },
        "ts": 1761289219.0,
        "elapsed_ms": 19000
      }
    ]
  }
];
