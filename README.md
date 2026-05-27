<div align=center><h1>
    🌱 VitaBench 2.0: Evaluating Personalized and Proactive Agents<br>
    in Long-Term User Interactions
</h1></div>

<p align="center">
  📃 <a href="https://arxiv.org/abs/2605.27141" target="_blank">Paper</a> • 🌐 <a href="https://vitabench.github.io/" target="_blank">Website</a> • 🏆 <a href="https://vitabench.github.io/#Leaderboard" target="_blank">Leaderboard</a> • 🔁 <a href="https://github.com/meituan-longcat/vitabench" target="_blank">VitaBench (v1)</a>
</p>

## 📖 Introduction

**VitaBench 2.0** extends [VitaBench](https://github.com/meituan-longcat/vitabench) from one-shot interactive tasks to **long-term, multi-session user interactions**, where an agent must remain *personalized* and *proactive* across conversations that span days, weeks, or months. While VitaBench (v1) measures whether an agent can complete a single complex life-serving request, VitaBench 2.0 asks a harder question: **can an agent remember the user, anticipate their evolving needs, and act on their behalf — over time?**

Each evaluation in VitaBench 2.0 simulates a continuing relationship between an agent and a user across multiple sessions in food delivery, in-store consumption, and online travel scenarios. Across these sessions, user preferences drift, prior commitments must be honored, and earlier context must be retrieved or reconstructed to act correctly in the present. To stress-test how agents handle this growing context, we evaluate every model under three memory regimes:

- **Full Context** — the entire interaction history is appended to the prompt, an upper-bound on what the model can possibly leverage.
- **Agentic Memory** — the agent autonomously decides what to write to and read from a structured memory store.
- **RAG Memory** — past interactions are chunked, embedded, and retrieved on demand.

We further measure both single-attempt success (**Avg@4**, **Pass@4**) and consistency across repeated rollouts (**Pass^4**), surfacing models that can *reliably* serve the same user — not just succeed once.

Our results show that even the strongest thinking models reach only ~50% Avg@4 under Full Context and degrade further under realistic memory settings, indicating that long-horizon personalization and proactivity remain open challenges for current LLM agents.

## 🏆 Leaderboard

Performance of non-thinking and thinking models under three memory settings. The leaderboard is sorted by **Avg@4** under **Full Context**. Best results in each column are in **bold**.

### Non-thinking Models

| Model | Full Context |  |  | Agentic Memory |  |  | RAG Memory |  |  |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | **Avg@4** | Pass@4 | Pass^4 | **Avg@4** | Pass@4 | Pass^4 | **Avg@4** | Pass@4 | Pass^4 |
| GPT-4o-mini             | 0.067 | 0.180 | 0.006 | 0.084 | 0.229 | 0.008 | 0.094 | 0.227 | 0.011 |
| GPT-3.5-Turbo           | 0.140 | 0.314 | 0.019 | 0.231 | 0.467 | 0.056 | 0.205 | 0.409 | 0.059 |
| LongCat-Flash-Chat      | 0.298 | 0.510 | 0.123 | 0.302 | 0.537 | 0.105 | 0.290 | 0.471 | 0.136 |
| GLM-4.5                 | 0.307 | 0.529 | 0.127 | 0.330 | 0.569 | 0.112 | 0.316 | 0.523 | 0.152 |
| Doubao-Seed-1.6         | 0.326 | 0.512 | 0.171 | 0.340 | 0.576 | 0.129 | 0.351 | 0.543 | 0.174 |
| GLM-4.6                 | 0.342 | 0.612 | 0.113 | 0.336 | 0.623 | 0.084 | 0.317 | 0.555 | 0.123 |
| Kimi-K2.6               | 0.378 | 0.632 | 0.147 | 0.397 | 0.674 | 0.145 | 0.383 | 0.621 | 0.163 |
| GLM-5.1                 | 0.420 | 0.654 | 0.204 | 0.423 | 0.664 | 0.182 | 0.383 | 0.585 | 0.200 |
| Doubao-Seed-2.0-pro     | 0.428 | 0.649 | 0.218 | 0.426 | 0.665 | 0.198 | 0.406 | 0.625 | 0.208 |
| **DeepSeek-V4-Pro**     | **0.456** | **0.652** | **0.267** | **0.427** | **0.658** | **0.207** | **0.424** | **0.618** | **0.247** |

### Thinking Models

| Model | Full Context |  |  | Agentic Memory |  |  | RAG Memory |  |  |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | **Avg@4** | Pass@4 | Pass^4 | **Avg@4** | Pass@4 | Pass^4 | **Avg@4** | Pass@4 | Pass^4 |
| o4-mini                 | 0.210 | 0.433 | 0.047 | 0.270 | 0.533 | 0.073 | 0.261 | 0.452 | 0.091 |
| Gemini-2.5-Flash        | 0.282 | 0.556 | 0.063 | 0.312 | 0.567 | 0.098 | 0.309 | 0.544 | 0.107 |
| Qwen3-Max               | 0.284 | 0.499 | 0.105 | 0.324 | 0.599 | 0.091 | 0.315 | 0.519 | 0.134 |
| Kimi-K2.6               | 0.293 | 0.533 | 0.099 | 0.280 | 0.508 | 0.088 | 0.303 | 0.511 | 0.118 |
| Gemini-2.5-Pro          | 0.331 | 0.605 | 0.109 | 0.378 | 0.638 | 0.138 | 0.320 | 0.579 | 0.109 |
| MiniMax-M2.7            | 0.345 | 0.584 | 0.145 | 0.351 | 0.609 | 0.124 | 0.314 | 0.518 | 0.143 |
| GLM-4.6                 | 0.359 | 0.612 | 0.116 | 0.351 | 0.625 | 0.107 | 0.336 | 0.574 | 0.135 |
| GLM-4.5                 | 0.364 | 0.623 | 0.156 | 0.311 | 0.596 | 0.106 | 0.336 | 0.555 | 0.147 |
| Doubao-Seed-1.6         | 0.373 | 0.599 | 0.176 | 0.383 | 0.646 | 0.123 | 0.375 | 0.591 | 0.179 |
| GLM-5.1                 | 0.394 | 0.587 | 0.213 | 0.352 | 0.556 | 0.150 | 0.328 | 0.485 | 0.185 |
| DeepSeek-R1-0528        | 0.396 | **0.691** | 0.131 | 0.412 | **0.712** | 0.118 | 0.390 | **0.643** | 0.153 |
| o3                      | 0.403 | 0.653 | 0.169 | 0.401 | 0.669 | 0.154 | 0.362 | 0.587 | 0.158 |
| Claude-4.5-Sonnet       | 0.417 | 0.658 | 0.197 | 0.397 | 0.642 | 0.178 | 0.374 | 0.573 | 0.186 |
| GPT-5                   | 0.441 | 0.658 | 0.226 | 0.421 | 0.647 | 0.204 | 0.410 | 0.591 | 0.236 |
| DeepSeek-V4-Pro         | 0.472 | 0.649 | 0.295 | 0.449 | 0.656 | 0.255 | **0.430** | 0.584 | 0.271 |
| Doubao-Seed-2.0-pro     | 0.474 | 0.683 | 0.270 | 0.428 | 0.650 | 0.225 | 0.339 | 0.496 | 0.205 |
| **Claude-Opus-4.6**     | **0.503** | 0.664 | **0.337** | **0.454** | 0.645 | **0.259** | **0.430** | 0.566 | **0.299** |

> **Avg@4** — mean success rate over 4 independent rollouts per task (single-attempt success).
> **Pass@4** — fraction of tasks solved in *at least one* of 4 rollouts (best-of-4).
> **Pass^4** — fraction of tasks solved in *all* 4 rollouts (consistency).

## 🔎 Citation

If you find this work useful, please cite:

```bibtex
@article{vitabench2_2026,
  title  = {VitaBench 2.0: Evaluating Personalized and Proactive Agents in Long-Term User Interactions},
  year   = {2026},
  eprint = {2605.27141},
  archivePrefix = {arXiv}
}
```

Please also consider citing the original VitaBench:

```bibtex
@article{he2025vitabench,
  title   = {VitaBench: Benchmarking LLM Agents with Versatile Interactive Tasks in Real-world Applications},
  author  = {He, Wei and Sun, Yueqing and Hao, Hongyan and Hao, Xueyuan and Xia, Zhikang and Gu, Qi and Han, Chengcheng and Zhao, Dengchang and Su, Hui and Zhang, Kefeng and Gao, Man and Su, Xi and Cai, Xiaodong and Cai, Xunliang and Yang, Yu and Zhao, Yunke},
  journal = {arXiv preprint arXiv:2509.26490},
  year    = {2025}
}
```

## 📜 License

This project is licensed under the MIT License.
