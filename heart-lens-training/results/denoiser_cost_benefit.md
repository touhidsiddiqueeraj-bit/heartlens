| Front-end | Avg Macro F1 (mixed) | vs Raw | Size | Latency | Worth it? |
|---|---|---|---|---|---|
| Raw + CNN | 0.478 | — | 0 KB | 0 ms | baseline |
| Butterworth (0.5-45Hz) + CNN | **0.670** | +0.192 | 0 KB | ~5 ms | **yes** |
| Autoencoder (19KB) + CNN | 0.591 | +0.113 | 19 KB | 590 ms | **no** (filter wins + 118× faster) |

> **Contribution:** Learned denoiser improves over raw but Butterworth outperforms it at 6/7 SNRs with 0 KB / 5 ms cost. Recommendation: remove denoiser from deployed pipeline.
