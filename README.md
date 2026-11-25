# Project Genesis-Rigorous

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%2012.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24528178.svg)](https://doi.org/10.5281/zenodo.24528178)

**Zero-Cheat Matrix Quantum Mechanics × Quantum Entanglement Laboratory for Emergent Spacetime, Gravity, Non-Abelian Gauge Bosons, and Chiral Matter**

本リポジトリは、2024–2026年の最先端理論物理学論文（行列量子力学による時空創発・ガウス則エンタングルメントによる極小面積則・背景独立テンソルネットワーク）を単一の 100% $U(N)$ ゲージ不変フレームワークへ統合し、**個人向け GPU（NVIDIA RTX 3060 12GB / 計算時間 数秒〜数分）上で「カンニングなし・対照実験付き」で再現検証するためのオープンソース計算物理テストベッド**です。

> **本研究の誠実な位置づけ**
> 本プロジェクトは「新しい量子重力理論の提唱」ではなく、先行理論研究（Fliss–Frenkel–Hartnoll–Soni 2025, Frenkel 2024, Gautam–Hanada–Jevicki 2025, Han–Hartnoll 2020, Steinacker 2024, Grosse–Klimčík–Prešnajder 1996 等）を単一コード基盤上で厳密に統合・数値監査し、個人 PC で再現可能な形で検証した **計算理論物理ベンチマーク・再現テストベッド** です。

---

## 1. 科学的厳格性の保証（5つの厳禁ルール / Zero-Cheat Policy）

すべての実験コードは以下の 5 つの偽装（Anti-Patterns）を完全に排除しています：

1. **答えのハードコード禁止**：単一球面半径を強制するカシミール拘束項（`R0_sq = alpha^2 (N^2-1)/4`）を一切使用しない。
2. **手動の鞍点脱出キック禁止**：固有値ソートや三重対角行列の注入といった非ゲージ共変な介入を行わず、100% $U(N)\times SO(D)$ 等方ランダム GUE 初期雲のみを使用。
3. **古典近似＋事後捏造の禁止**：量子共分散 $(\Sigma_x, \Sigma_p)$ を持つ HFB 波束と厳密 BdG リンクフェルミオン反作用を常に連成。
4. **連続極限で破綻する小 $N$ 専用閾値の禁止**：$N = 8, 10, 12, 16, 20, 32$ へスケールしても一貫して成立する物理量のみを評価。
5. **反証対照実験の義務化**：必ず **Universe A（カオス崩壊相）**・**幾何固定コントロール**・**可換 $U(1)$ コントロール**・**大振幅量子ノイズ保護テスト**を同一コードで同時実行し、明確な分離を確認。

---

## 2. 主要な検証成果サマリー（RTX 3060 12GB 実測）

### (A) マスター数理・ゲージ不変性監査（`python genesis_cli.py verify`：`17/17 PASS`, `2.06秒`, VRAM `68 MB`）
- $U(N)$ ゲージ不変性（全 7 観測量）：最大誤差 **`5.68e-14`**
- 4次ウィック展開 $E_{\text{YM}}^{(4)}$：等方解析誤差 **`2.66e-15`**／一般非等方 20万標本 GPU モンテカルロ **`0.26σ`**
- $N=2$ 厳密対角化（256次元 Fock 空間）vs Peschel BdG 公式：**`dE = 1.55e-15`, `dS = 3.33e-16`**
- 曲がった全宇宙（$M=N$）での $U(N)$ 一重項物理射影 $P_{\text{phys}}$ ガウス則ゼロ：**`1.47e-27`**（HFB Hohenberg–Martin ギャップ汚染の除去）
- 厳密 $SO(3)$ 縮退球面での Daleckii–Krein 有理化差分商勾配：**`RelErr = 2.46e-09`**（特異点・`NaN` ゼロ）
- $O(N^3)$ 行列直接エンジンと厳密 HFB コアの一致（$Y=0$ 厳密一致＋$Y\neq 0$ ヘッセトレース $\mathrm{Tr}(H^{(2)}_{\text{YM}}) = (D-1)g^2\mathrm{Tr}(\Delta_Y)$）：**`MaxErr = 7.11e-15`（`TrH2 = 1.87e-16`）**

### (B) 創発時空・重力・ゲージ場・キラル素粒子の定量結果

| フェーズ | 検証した物理現象 | 主要な実測結果（対照実験との比較） | 計算時間 / VRAM |
| :--- | :--- | :--- | :---: |
| **Phase 1** | 2系統量子相転移・点プローブ幾何・ブラインド極小面積則（$N=8, 10, 12$） | Universe A: `Tr(Y^2)=0.0000` 崩壊 vs Universe B: 単一既約球面自発凝縮（`AlgErr = 2.1e-7..6.4e-7`）。Provost–Vallée 計量が理論値 $\frac{N}{4r^2}$ と一致。ブラインド最小化のみで周長 **`-68.3%〜-73.7%`** 縮小、非対称化前の生リンクエントロピー相関 **`r(S_TN) = 0.996–0.9997`**（A は `-0.22`）。 | `1.5–3.6分` / `24–93 MB` |
| **Stage 3** | 局在随伴ディラック物質による創発重力（$N=8, 10$） | 北極（物質位置）の固有ガウス曲率が **`+19.2%`** 集中（$r = \mathbf{+0.9992}$, $N=10$ で `+0.9997`）、PV 面積要素 $+10.3\%$ 集中（$r = \mathbf{+0.9995}$）。北極帽 $S_{\text{EE}}$ が `+27.8%` 増加する一方、南極対照は `< 0.25%` 不変。ブラインド極小半球が物質を避けて反極側へ偏向（$N=8$: `cosθ: -0.585 → -0.903`, $N=10$: `+0.935 → -0.591`）。 | `1.8–2.1分` / `24–43 MB` |
| **Phase 3A** | 2体間の創発重力引力 ＆ 実時間光円錐（$N=8$） | 直接結合ゼロの2体物質間で、幾何固定コントロール `1.7e-5` に対し動的時空で **`V_grav(25°) = -8.77 → 40°: -4.01 → 60°: -0.56 → 180°: 0.00`** の湯川型引力を検出。固定共分散 Verlet 450 ステップ発展（エネルギー保存 `4.45e-05`）で、有限 $N$ 非可換重なり領域（$\theta \le 80^\circ$）を除いた遠方因果区間 $[85^\circ, 160^\circ]$ において **`c_ang = 1.483 rad/t`（`r = 0.9983`、YM 理論上限 $g\alpha=2.808$ 以下の亜光速群速度）**を計測。 | `~7分` / `15 MB` |
| **Phase 3B** | $O(N^3)$ エンジン（厳密 $(D-1)/D$ 1ループ係数）による大 $N=16\dots 32$ 連続極限 | $N=16, 20$ で新規等方乱数から単一球面へ収束（`AlgErr = 6.8e-8, 4.3e-8`, `alpha = 11.2363`）。$N=32$（1,024モード）単一球面も `8.2 MB` VRAM で理論計量 $N/(4r^2)$・$N(N^2-1)$ エネルギー則と 4–5 桁一致（$N=24, 32$ の 4 シード準安定マルチ球面盆地も併記）。 | `37–229秒` / **`8.2 MB`** |
| **Phase 3C** | 重なりブレーンからの $U(2)$ ゲージボソン ＆ クーロン枝行列ヒッグス分裂（$N=12, D=4$） | $v=0$ で 12 モード厳密縮退（`3.3e-13`）。外部ブレーン分離 $v>0$ で物理モード `436→434`（ゴールドストーン2個吸収）、対角 $U(1)\times U(1)$ 光子ズレ `2.1e-13`（基底 $\omega^2=\mu^2=1$ 不変）、非対角 $W^\pm$ が厳密ヒッグス式 $\Delta\omega_W^2 = g^2 v^2$ と **`r = 1.00000000`（誤差 `2.2e-13`）** で一致。非可換 $SU(2)$ 4次自己結合も実証。 | **`3.0秒`** / `24 MB` |
| **Phase 3D** | 4次元空間 $S^2\times S^2$（$D=6, N=16$）＆ キラル素粒子 3 世代 | 6D Chaos A 崩壊 vs 6D Emergent B が交差交換子誤差 `3.35e-09` で $12\oplus 3\oplus 1$ 可約積構造へ自発凝縮し、6D 点プローブ計量が **厳密ランク 4（`[0, 0, 0.00642, 0.00642, 0.00866, 0.00866]`、理論値と5桁一致）** を出力。GKP キラル・ディラック演算子（$\{\mathcal{D},\gamma_5\}=0$ `1.8e-15`）で磁束差 $q=0,1,2,3$ に対し厳密な質量ゼロ片巻き世代が **`0, 1, 2, 3` 世代** 出現し、大振幅量子ノイズ下でも `|E_0| <= 6.6e-16` で保護。 | **`58秒`** / `9.0 MB` |

---

## 3. リポジトリ構成

```text
.
├── README.md                             # 本ファイル（概要・成果・再現手順）
├── LICENSE                               # MIT License
├── CITATION.cff                          # 学術引用情報（Zenodo DOI 対応）
├── CHANGELOG.md                          # 研究・開発履歴
├── requirements.txt                      # 依存ライブラリ一覧
├── pyproject.toml                        # パッケージ設定
├── genesis_cli.py                        # 単一統合 CLI エントリポイント
├── genesis_rigorous/                     # 研究本体パッケージ
│   ├── safety.py                         # VRAM/RAM 3層安全ガード・数値安定化
│   ├── exact_hfb.py                      # 厳密 HFB・BdG・ガウス則・点プローブ
│   ├── matrix_native.py                  # O(N^3) 行列直接エンジン（f_abc 不要）
│   ├── dirac_matter.py                   # 随伴ディラック物質・Daleckii-Krein 勾配
│   ├── verification.py                   # 全 17 項目マスター監査スイート
│   └── experiments/                      # Phase 1〜3D のモジュラー実験群
├── tests/                                # unittest / pytest 互換自動テスト
├── paper/                                # Overleaf 向け LaTeX テクニカルレポート原稿
├── docs/                                 # 研究仕様書・アーキテクチャ設計文書
└── results/                              # 代表的な検証図・JSON・EXPERIMENT_LOG.md
```

---

## 4. クイックスタート（再現手順）

### 環境要件
- Python 3.10+ / PyTorch 2.x（CUDA GPU 推奨、CPU 自動フォールバック対応）
- 検証済み環境：AMD Ryzen 7 5700X / NVIDIA RTX 3060 (12GB VRAM, CUDA 12.6) / RAM 32GB

```bash
# 1. 依存パッケージ導入（CUDA 環境では環境に応じた PyTorch CUDA wheel を導入）
pip install -r requirements.txt

# 2. 全 17 項目の数理・ゲージ不変性監査を実行（約2秒）
python genesis_cli.py verify

# または標準テストランナーで実行
python -m unittest discover -s tests -v

# 3. 統合ショーケースを一括実行（--fast は N=6 + ゲージ + 4Dキラル 約1.5分）
python genesis_cli.py suite --fast

# 4. 各実験を個別に再現
python genesis_cli.py phase1 --N 8
python genesis_cli.py gravity --state-pt results/<phase1_dir>/phase1_states.pt
python genesis_cli.py realtime --state-pt results/<phase1_dir>/phase1_states.pt
python genesis_cli.py large-n --Ns 16,20,24,32
python genesis_cli.py gauge --n-sphere 6
python genesis_cli.py chiral
```

---

## 5. 科学的誠実さの境界線（射程外・未解決領域の明示）

本テストベッドの守備範囲を明確に定義します：
- **達成したこと**：$(0+1)$ 次元行列量子力学からの 2D/4D 非可換 D ブレーン世界体積創発、龍–高柳（Ryu–Takayanagi）型極小面積則、静的・動的（2体湯川型引力・光円錐）な創発重力応答、重なりブレーンからの $U(2)$ ゲージボソン＆ヒッグス機構、トポロジカル保護された質量ゼロ・キラル素粒子世代の**統一的かつ厳格な対照実験付き数値実証**。
- **本トイモデルの射程外**：10次元超対称 IKKT 模型の符号問題（Sign Problem）の厳密解決による $SO(9)\to SO(3)$ 動的次元選択、$(3+1)$ 次元ローレンツ不変な質量ゼロ・スピン2重力子の厳密散乱振幅、および現実の標準模型の質量比・結合定数の第一原理導出。

---

## 6. 主要参考文献（Foundational References）

1. J. R. Fliss, A. Frenkel, S. A. Hartnoll, & R. M. Soni, *"Minimal Areas from Entangled Matrices"*, *SciPost Phys.* **18**, 171 (2025), [`arXiv:2408.05274`](https://arxiv.org/abs/2408.05274).
2. A. Frenkel, *"APD-Invariant Tensor Networks from Matrix Quantum Mechanics"* (2024), [`arXiv:2407.16753`](https://arxiv.org/abs/2407.16753).
3. V. Gautam, M. Hanada, & A. Jevicki, *"Operator algebra, quantum entanglement, and emergent geometry from matrix degrees of freedom"*, *JHEP* **01** (2025) 019, [`arXiv:2406.13364`](https://arxiv.org/abs/2406.13364).
4. X. Han & S. A. Hartnoll, *"Deep Quantum Geometry of Matrices"*, *Phys. Rev. X* **10**, 011069 (2020), [`arXiv:1906.08781`](https://arxiv.org/abs/1906.08781).
5. H. C. Steinacker, *"Quantum Geometry, Matrix Theory, and Gravity"*, Cambridge University Press (2024).
6. H. Grosse, C. Klimčík, & P. Prešnajder, *"Topologically nontrivial field configurations in noncommutative geometry"*, *Commun. Math. Phys.* **178**, 507 (1996).
7. A. Chatzistavrakidis, H. Steinacker, & G. Zoupanos, *"Intersecting branes and a standard model realization in matrix models"*, *JHEP* **09** (2011) 115, [`arXiv:1107.0265`](https://arxiv.org/abs/1107.0265).
