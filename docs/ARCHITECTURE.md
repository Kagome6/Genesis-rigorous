# Architecture & Mathematical Audit Notes (`genesis_rigorous`)

本ドキュメントは、統合パッケージ `genesis_rigorous/` の数理構造・設計思想・安全ガードの要点をまとめた技術ノートです。詳細な理論仕様および 5 つの厳禁事項は [EMERGENT_SPACETIME_RESEARCH_SPEC.md](../EMERGENT_SPACETIME_RESEARCH_SPEC.md) を参照してください。

## 1. モジュール責務分離

| モジュール | 役割 | 計算量 / メモリ |
| :--- | :--- | :--- |
| `genesis_rigorous/safety.py` | 3層ハードウェア安全ガード（VRAM上限・RAM監視・プリフライト遮断）、`SIGINT` 安全停止、縮退行列用 `safe_eigh`/`safe_eigvalsh` | $O(1)$ |
| `genesis_rigorous/exact_hfb.py` | 完全 HFB 4次ウィック展開・厳密 2 フレーバー BdG リンク反作用・$U(N)$ 一重項物理射影 $P_{\text{phys}}$ カシミール・Cholesky 相似変換 $S_{\text{bos}}$・厳密摂動 Provost–Vallée 計量 | $O(N^6)$（$N \le 16$ 厳密基準） |
| `genesis_rigorous/matrix_native.py` | `f_abc` 全廃の行列直接交換子エンジン＋クロネッカー積ラプラシアン＋$U(1)$ トレース分離型 1 ループ HFB 包絡線勾配（任意次元 $D$ 対応） | $O(N^3)\text{–}O(N^4)$（$N = 32$ で VRAM `8.2 MB`） |
| `genesis_rigorous/dirac_matter.py` | $4N^2 \times 4N^2$ 随伴ディラック演算子 $\mathcal{D}_{\text{adj}}(Y)$・Daleckii–Krein 有理化差分商による特異点ゼロ解析勾配・子午線固有ガウス曲率スキャナ | $O(N^6)$（$N \le 12$ 推奨） |
| `genesis_rigorous/verification.py` | 全 17 項目の自動数理監査（厳密対角化・20万標本 GPU MC・ゲージ不変性・縮退球面勾配・クリフォード代数・キラル指数） | 約 2 秒（RTX 3060） |
| `genesis_rigorous/experiments/` | Phase 1, Stage 3, Phase 3A–3D の再現実験ランナーおよび可視化ダッシュボード生成 | 各 3 秒〜数分 |

## 2. 数値監査で特定・解決した 4 つの重要数理課題

1. **HFB Hohenberg–Martin ゲージゼロモード汚染の解消**：
   1 ループ自己エネルギー $K_{\text{loop}}$ により $U(N)$ ゲージ軌道モードが微小ギャップを獲得する問題に対し、ゲージ共変なクーロン横波部分空間射影 $P_{\text{phys}}$ を導入することで、曲がった全宇宙（$M=N$）のガウス則カシミールを `74,168 → 1.47e-27` へ厳密復元。
2. **等方真空における Cholesky 相似変換 $S_{\text{bos}}$**：
   $Y=0$ の完全縮退共分散行列に対する `eigh` 微分特異点を、$L_x^T S_{p,\text{in}} L_x \sim S_{p,\text{in}} S_{x,\text{in}}$ の Cholesky 相似変換で根本解消。
3. **厳密 $SO(3)$ 縮退球面上の Daleckii–Krein 勾配**：
   $|\mathcal{D}_{\text{adj}}(Y)| = \sqrt{\mathcal{D}_{\text{adj}}^2 + \varepsilon^2}$ の固有値微分で生じる $1/(\lambda_m - \lambda_n)$ 発散を、有理化恒等式 $F^{[1]}_{mn} = (\lambda_m + \lambda_n)/(|\lambda_m| + |\lambda_n|)$ で完全に相殺。
4. **$U(1)$ トレースモード分離による $O(N^3)$ 1ループ HFB 厳密一致**：
   相互作用しないトレースモード（$k=0$）を $SU(N)$ ギャップ方程式から分離することで、$Y=0$ 真空の厳密 HFB エネルギーと `1.42e-14` で一致。
