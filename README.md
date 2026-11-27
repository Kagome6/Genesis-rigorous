# Project Genesis-Rigorous v1.0.2

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%2012.x-ee4c2c.svg)](https://pytorch.org/)
[![Audit: 20/20 PASS](https://img.shields.io/badge/Audit-20%2F20%20PASS-brightgreen.svg)](tests/test_genesis_audit.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24566494.svg)](https://doi.org/10.5281/zenodo.24566494)

**A Unified Mean-Field (HFB) and Exact $U(N)$-Singlet Gauged HMC PyTorch/CUDA Testbed for Emergent Fuzzy-Sphere Geometry, Subregion Entanglement Scaling, and Noncommutative Membrane Response**

<img width="1596" height="945" alt="watch_v102_birth_5panel" src="https://github.com/user-attachments/assets/031c2975-ffdd-4e38-9a7b-f1c6d747ea37" />

  <em><strong>ライブ観測（<code>python genesis_cli.py watch --mode birth</code>）</strong>：厳密な U(4) ゲージ一重項・格子 HMC で空間誕生を可視化した数値シミュレーション動画。完全に融解した 0 次元量子霧から結合 g を 1.35→0.25 へ冷却すると、フラックス ν = 2.30 では非閉じ込め転移とともに 2 次元ファジー球面が自発的に核形成し、点プローブ量子フィッシャー情報計量がカオスから 2 次元ランク 2 へ収束する一方、同一初期霧の対照系（ν = 0.80）は 0 次元真空に留まる。1992年から2007年にかけて、「0次元の量子霧から2次元球面が自発的に生まれる」という現象そのものは既に理論的に知られている。注意点として、ハミルトニアンにマイヤーズ項を入れた時点で、「低温・弱結合で安定になる結晶の形が 2 次元ファジー球面（またはその同心多重球面）になる」こと自体は数学的に決まっている。例えるなら水分子(あるいは流体)が元々持っている性質で雪の結晶が六角形になるのと同じである。</em>

---

## 1. このプロジェクトの目的と概要（Overview & Purpose）

**「あらかじめ空間の座標や距離を用意しなくても、純粋な“行列”と“量子もつれ”だけから、空間の広がり・エンタングルメント境界則・量子融解／核形成・非可換弾性応答・ゲージ場・キラル素粒子は本当に立ち上がるのか？」**

近年の理論物理学（行列量子力学・ホログラフィー・量子情報幾何）では、時空そのものを微視的な行列自由度とエンタングルメントから再構成する研究が急速に進展しています（Fliss–Frenkel–Hartnoll–Soni 2025, Frenkel 2024, Gautam–Hanada–Jevicki 2025, Han–Hartnoll 2020, Steinacker 2024 等）。しかし、これらを実際に数値計算で確かめようとすると、以下のような壁が立ちはだかります：

- 大規模モンテカルロ法や厳密対角化は計算コストが大きく、個人環境では再現が難しい。
- 一方で簡易的なトイコードでは、「目標半径をポテンシャルに書き込む」「球面に近い初期種を与える」「反証用の対照実験や $N$ スケーリング検証を省く」といった**暗黙のカンニング（誘導バイアス）**が混入しやすい。

**Project Genesis-Rigorous v1.0.2** は、個人向け GPU（NVIDIA RTX 3060 12GB、計算時間 数秒〜数分）上で、Zero-Cheat・対照実験必須・20 項目自動監査付きでこれらを一貫検証するための計算物理テストベッドです。`v1.0.1` に対する独立査読で指摘された 3 つの本質的課題（① $k=0$ トレースモード偽陽性と $N$ スケーリング欠如、② 深い古典領域・非一重項平均場への限定、③ 汎用弾性膜コントロールの不足）を解決しています。そして、本検証論文の本丸は球面の誕生そのものではなく、**生まれた球面の上で”量子エンタングルメントがどう振る舞うか（2024〜2025年にS. Hartnollらが開拓している最新領域）”**や「少なくとも2次元球面上での物質同士が引き合う効果の正体は重力ではなく、ゴム膜のたわみ効果であること。」を検算することにある。

> **本研究の誠実な位置づけ**
> 本プロジェクトは「新しい量子重力理論の提唱」や「標準模型の第一原理導出」ではなく、1980–2025 年の先行理論研究（Madore 1992, Myers 1999, Azuma et al. 2004, Kawahara–Nishimura–Takeuchi 2007, Berenstein–Dzienkowski 2012, Ishiki 2015, Han–Hartnoll 2020, Frenkel 2024, Fliss–Frenkel–Hartnoll–Soni 2025, Steinacker 2024, Grosse–Klimčík–Prešnajder 1996, Balachandran–Immirzi 2003 等）の核心機構を単一コード基盤上で統合・数値監査し、個人 PC で再現可能な形で検証した **計算物理ベンチマーク・再現テストベッド** です。

---

## 2. 今回のテスドヘッドとして解いている 3段階モジュール数理構成（3-Module Mathematical Architecture）

本リポジトリは、背景時空や座標ラベルを一切持たない $(0+1)$ 次元エルミート行列 $\hat{X}_I$（$I=1,\dots,D$）を共通の土台とし、実験の目的に応じて以下の **3 つの独立モジュール** を組み合わせて検証するテストベッド構成をとっています：

<img width="1053" height="534" alt="解いている数理モデル2" src="https://github.com/user-attachments/assets/818b08e6-4dd9-44eb-ba3a-28a50d3faaa1" />


- **モジュール (I)（ボソン行列量子力学 $\hat{H}_{\mathrm{MQM}}$ ＆ 厳密 $U(N)$ 一重項格子 HMC 作用 $S_E[X,U]$）**：本テストベッドの中核エンジン。ヤン＝ミルズ交換子 2 乗＋質量項 $\mu$＋Myers フラックス $\nu$ からなり、古典的には $\nu_{\mathrm{spin}}=\sqrt{2}\,\mu$ で球面解が現れ、**真の一次相転移点 $\nu_c = \frac{3}{2}\mu = 1.5\mu$**（厳密 HMC では量子補正により $\nu_c^{\mathrm{quantum}} \approx 1.65\text{–}1.68$ へシフト）を超えると 2 次元ファジー球面が安定化します。`v1.0.2` では平均場 HFB（4 次ウィック展開）に加え、動的ゲージ場（Polyakov ループ $U \in U(N)$）を含む **厳密 $U(N)$ 一重項・格子 HMC（`hmc_singlet.py`、符号問題ゼロ）** を実装し、強結合での量子融解と非閉じ込め転移を直接計算します。
- **モジュール (II)（補助トレースレス $\mathfrak{su}(N)$ リンク・フェルミオン $\hat{H}_{\mathrm{link}}^{\mathfrak{su}(N)}[\tilde{Y}]$）**：Frenkel (2024) に着想を得た、下に有界かつ Peschel 公式で厳密可解な 2 フレーバー BdG リンク模型（Exp 1 用）。`v1.0.2` では幾何に依存しない $k=0$ 恒等モード $E_0 = I_N/\sqrt{N}$ を対角化前に厳密射影で除去し（$\mathrm{Tr}\,\hat{\chi}^{\alpha,s}=0$）、$U(N)$ 不変な単位半径正規化 $\tilde{Y}_I = Y_I / R_{\mathrm{rms}}$ により純粋な幾何依存エンタングルメント $S_{\mathrm{TN}}^{\mathfrak{su}(N)}$ を抽出します。
- **モジュール (III)（局在物質プローブ $E_{\mathrm{Dirac}}[Y]$ vs 汎用弾性膜対照 $E_{\mathrm{rad}}[Y]$）**：点プローブ・コヒーレント波束上の単一粒子正エネルギー期待値 $E_{\mathrm{Dirac}}[Y] = M_{\mathrm{src}}\langle c\,|\,\sqrt{\mathcal{D}_{\mathrm{adj}}(Y)^2}\,|\,c\rangle$（縮退球面での `0/0` 自動微分破綻は Daleckii–Krein 有理化差分商で完全解消）を、`v1.0.2` Exp 3 で同一ソースエネルギーの汎用局所動径ポテンシャル $E_{\mathrm{rad}}[Y] = \lambda\,\mathrm{Tr}(\hat{\rho}_c \sum_I Y_I^2)$ と直接比較します。

> **創発空間の「座標無し測定」**：得られた行列 $Y_I$ の幾何は、外部座標を仮定せず、$U(N)$ ゲージ共変な点プローブ・ディラック演算子 $D_{\mathrm{probe}}(\vec{y}) = \sum_I (Y_I - y_I I_N)\otimes\Gamma_I$ の零点殻と、その基底状態の **Provost–Vallée 計量（＝量子フィッシャー情報計量 $g^{\mathrm{QFI}} = 4 g^{\mathrm{PV}}$）** によって次元・計量・固有ガウス曲率を測定することを目指します。

---

## 3. 何を行ったのか？（`v1.0.2` 3 大核心実験 ＋ 継承 Phase 1–3 ロードマップ）

科学的な重みづけを明確にするため、全成果を **「Category I：自動数理監査」「Category II：`v1.0.2` 新規コア物理・対照実験」「Category III：継承・監査済み Phase 1–3 ベースライン」** の 3 層に明確に区分しています：

### Category II — `v1.0.2` 新規コア物理・対照実験（査読で突かれた 3 つの急所を正面解決）
1. **Exp 1（トレースレス $\mathfrak{su}(N)$ ＆ HFB エンタングルメント $N$ スケーリング, `entanglement-scaling`）**：$k=0$ トレースモードを完全除去し、固定比率 $M = N/2$ で $N = 6, 8, 10, 12, 16, 20, 24$ へスケール。AICc/BIC モデル選択により、ファジー球の $O(N)$ 行列境界則と同ノルム GUE ランダム行列の $O(N^2)$ 体積則、および可換整列塵の $O(1)$ を統計的に完全分離。
2. **Exp 2（一次相転移 ＆ 強結合量子融解：HFB vs 厳密 $U(N)$ 一重項 HMC, `phase-hmc`）**：古典相転移 $\nu_c = 1.50$ 付近の精密掃引で量子シフト $\nu_c^{\mathrm{quantum}} \approx 1.65\text{–}1.68$ と球面枝での `0.6%–0.8%` 一致（および $+18.45$ エネルギー差の厳密分解）を検証。さらに $g \in [0.25, 1.45]$ の強結合掃引で **非摂動量子融解・Polyakov ループ閉じ込め転移・平均場 HFB の `48.2%` 半径破綻** を直接観測（全点で Schwinger–Dyson 恒等式合格）。
3. **Exp 3（随伴ディラック vs 汎用 $\mathrm{Tr}(\hat{\rho}Y^2)$ 弾性膜コントロール, `membrane-control`）**：同一ソースエネルギー比較により、2 体近距離引力は弾性膜の波束重なりに普遍的な効果であることを実証しつつ、1 体局所曲率の符号（長球型赤道くびれ `+11.3%` vs 扁球型極へこみ `-9.8%`）の違いを定量分離。

### Category III — 継承・監査済み Phase 1–3 ベースライン資産
4. **Phase 1（2 系統相転移 ＆ 点プローブ幾何, $N=8, 10, 12$）**・**Stage 3 / Phase 3A（局在物質の弾性膜応答 ＆ 遠方波面伝播）**・**Phase 3B（大 $N=16\dots 32$ 行列直接エンジン：古典力 $O(N^3)$ / ラプラシアン保持 $O(N^4)$ / 密対角化 $O(N^6)$）**・**Phase 3D-1（$D=6, N=16$ からの 4 次元積空間 $S^2\times S^2$ 自発因子分解）**。
5. **解析恒等式・指数の厳密検算**：**Phase 3C**（重なりブレーン $U(2)$ ゲージボソン ＆ クーロン枝行列ヒッグス分裂 $\Delta\omega_W^2 = g^2 v^2$）・**Phase 3D-2**（ファジー Ginsparg–Wilson カイラル素粒子 $q=0,1,2,3$ 世代 ＆ 耐ノイズ保護）。

---

## 4. 科学的厳格性の保証（Zero-Cheat Policy ＆ 20 項目自動監査）

### 5 つの厳禁ルール（Zero-Cheat Policy）
1. **答えのハードコード禁止**：単一球面半径を強制するカシミール拘束項（`R0_sq = alpha^2 (N^2-1)/4`）を一切使用しない。
2. **手動の鞍点脱出キック禁止**：非ゲージ共変な固有値ソートや球面種のこっそり注入を行わず、100% 等方ランダム GUE 雲または完全に融解した 0D 量子液体から出発。
3. **非一重項平均場と厳密一重項の明確な区別**：HFB 波束（$\langle\hat{X}_I\rangle = Y_I \neq 0$、汎関数は $U(N)$ 共変）と、動的 Polyakov ループ $U \in U(N)$ で厳密にゲージ一重項を課す格子 HMC（`hmc_singlet.py`）を明確に区別。
4. **$N$ スケーリングとモデル選択の義務化**：単一 $N$ の相関値だけで面積則を主張せず、$M=N/2$ 固定での $N$ 依存性を AICc/BIC と GUE 対照で検証。
5. **非自明な反証対照実験の義務化**：臨界未満真空（$\nu=0.80$）・同ノルム GUE・可換整列塵・同一エネルギー汎用弾性ポテンシャル $\mathrm{Tr}(\hat{\rho}Y^2)$ を同一コードで同時実行。

### マスター 20 項目自動監査（`python genesis_cli.py verify`：`20/20 PASS`, `3.7秒`, Live/Peak VRAM `68/118 MiB`）
- `T01`–`T17`（継承＋監査済み）：$U(N)$ 共変性（`5.68e-14`）、4 次ウィック展開（解析 `2.66e-15` / 20 万標本 MC `0.26σ`）、$N=2$ 256 次元 Fock 厳密対角化 vs Peschel BdG（`dE = 1.55e-15, dS = 3.33e-16`）、物理射影 $P_{\mathrm{phys}}$ ガウス則ゼロ（`1.47e-27`）、Daleckii–Krein 勾配（`2.46e-09`）、`MatrixNativeQuantumCore` ヘッセトレース恒等式（`TrH2 = 1.87e-16`）、GW カイラル指数（`{D,g5}=1.8e-15`, `NoisyIdx=+3`）。
- **`T18`（新規）**：トレースレス $\mathfrak{su}(N)$ BdG 基底のゼロトレース性（`max|Tr(E)| = 1.1e-16`）＆ $U(N)$ 共変性（`CovErr = 8.9e-16`）。
- **`T19`（新規）**：厳密 $U(N)$ 一重項 HMC 解析力 vs 中心差分（`Err(F_X) = 7.05e-09`, ユニタリ `Err(F_U) = 1.57e-09`）。
- **`T20`（新規）**：自由調和振動子 HMC 厳密熱平均（`0.22σ`）＆ 離散 Schwinger–Dyson 恒等式（`0.33σ`, 採択率 `0.89`）。

---

## 5. 主要な検証結果サマリー（RTX 3060 12GB / Ryzen 7 5700X 実測）

### (A) `v1.0.2` 新規 3 核心実験
| 実験 | 検証対象 | 主要な実測結果（対照実験との定量比較） | 計算時間 |
| :--- | :--- | :--- | :---: |
| **Exp 1** | **[量子情報]** トレースレス $\mathfrak{su}(N)$ ＆ HFB エンタングルメント $N$ スケーリング（$N=6\dots 24$） | 赤道切断 $M=N/2$ でファジー球は $O(N)$ 境界則（$S_{\mathrm{TN}}^{\mathfrak{su}(N)} \propto N^{0.834\pm 0.005}$, $\Delta\mathrm{BIC}=+24.51$ over $N^2$）に従う一方、同ノルム GUE は $O(N^2)$ 体積則（$N^{1.933\pm 0.024}$, $N=24$ で **`5.00×`** 大きい）、可換整列塵は厳密 $O(1)=0.833$。ボソン HFB も $N^{1.37}$ vs $N^{2.31}$（$N=12$ で **`6.32×`** 分離）。ブラインド Grassmannian 最適化で周長 `-73.5%`（GUE `-48.2%`）。 | `45.1秒` |
| **Exp 2** | **[非摂動 HMC]** 一次相転移（$\nu_c=1.50$）＆ 強結合量子融解（$N=4, L=12, \beta=3.0$） | $g=0.30$ で量子相転移点 $\nu_c^{\mathrm{quantum}} \approx 1.65\text{–}1.68$ を検出し、球面枝で HFB–HMC の $\langle\mathrm{Tr}X^2\rangle$ が `0.6%–0.8%` 一致（エネルギー差 $+18.45$ をゲージ ZPE $+4.64$ ＋格子分散 $+15.74$ に分解）。強結合 $g \in [0.25, 1.45]$（全 7 点で Schwinger–Dyson 合格）で **量子融解**（`AlgErr: 0.157 → 10.53`）、**Polyakov ループ閉じ込め**（`|Tr U|/N: 0.983 → 0.220`）、HFB 半径破綻（$g=0.85$ で **`48.2%`** 誤差）を観測。 | `380.8秒` |
| **Exp 3** | **[膜対照実験]** 随伴 Dirac vs 汎用 $\mathrm{Tr}(\hat{\rho}Y^2)$ 弾性膜応答（$N=8$, $E_{\mathrm{src}}=1254.38$） | 2 体近距離引力（$25^\circ$ で汎用 $\mathrm{Tr}(\hat{\rho}Y^2)$ `-14.18` vs Dirac `-8.77`）は **弾性膜の波束重なりに普遍的** であることを立証。一方、1 体局所曲率の符号は随伴 Dirac（長球型赤道くびれ `dK_N = +11.33%`）と汎用動径井戸（扁球型極へこみ `dK_N = -9.76%`）で逆転。 | `211.9秒` |

<img width="2250" height="1560" alt="fig8_v102_exp1_entanglement_scaling" src="https://github.com/user-attachments/assets/dabd7e38-4aa7-430f-b099-7b8e67c53a70" />
<em><strong> N=6→24 と行列サイズを増やしたとき、ファジー球は O(N)、ランダム行列 GUE は O(N2) に真っ二つに分かれる。「量子もつれの増え方や切り口の最小化（境界則 $\sim N$ か、体積則 $\sim N^2$ か）」だけで、行列の中に本物の2次元空間（球面）が生まれている状態と単なるデタラメな混沌（ランダム行列）とを厳密に見分けられる。 </em>


### (B) 継承・監査済み Phase 1–3 ベースライン
| フェーズ | 区分 / 検証対象 | 主要な実測結果（`v1.0.2` 監査反映済み） | 計算時間 |
| :--- | :--- | :--- | :---: |
| **Phase 1** | **[創発]** 2 系統相転移・点プローブ幾何（$N=8, 10, 12$） | Universe A 崩壊 vs Universe B 単一球面凝縮（`AlgErr = 2.1e-7..6.4e-7`）、Provost–Vallée 計量が理論値 $N/(4r^2)$ と一致。 | `1.5–3.6分` |
| **Stage 3 / 3A** | **[膜力学]** 局在随伴 Dirac 弾性応答 ＆ 遠方波面伝播（$N=8, 10$） | 北極ガウス曲率集中（$r > 0.999$）、固定共分散 Verlet 450 ステップ発展の遠方区間 $[85^\circ, 160^\circ]$ で亜光速弾性波面 `c_ang = 1.483 rad/t`（`r = 0.9983`, $g\alpha = 2.808$ 以下）。 | `2–7分` |
| **Phase 3B** | **[大 $N$]** Matrix-Native 連続極限（$N=16\dots 32$） | $f^{ABC}$ を廃し $O(N^4)$ ラプラシアン（$N=32$ で `16 MiB` 保持、ステップピーク `265.6 MiB` / 実行後 `8.2 MiB`）で $N=16, 20, 32$ の理論計量・エネルギー則と 4–5 桁一致。 | `37–229秒` |
| **Phase 3C** | **[恒等式検算]** 重なりブレーン $U(2)$ ＆ クーロン枝行列ヒッグス（$N=12, D=4$） | 外部ブレーン分離 $v>0$ で物理モード `436→434`、対角光子ズレ `2.1e-13`、非対角 $W^\pm$ が厳密ヒッグス式 $\Delta\omega_W^2 = g^2 v^2$ と誤差 `2.2e-13` で一致。 | `3.0秒` |
| **Phase 3D** | **[創発＋指数]** 4 次元積空間 $S^2\times S^2$ ＆ ファジー GW キラル 3 世代 | 6 行列系から Rank-4 計量へ自発因子分解（`58秒`）。双基本 GW 演算子（`{D,g5}=1.8e-15`）で $q=0,1,2,3$ 世代が大振幅ノイズ下でも `|E_0| <= 6.6e-16` で保護。 | `58秒` |

---

## 6. クイックスタート ＆ ライブ観測ビューア（再現手順）

### 環境要件
- Python 3.10+ / PyTorch 2.x（CUDA GPU 推奨、CPU 自動フォールバック対応）
- 検証済み環境：AMD Ryzen 7 5700X / NVIDIA RTX 3060 (12GB VRAM, CUDA 12.6) / RAM 32GB / PyTorch 2.14.1+cu126

```bash
# 1. 依存パッケージ導入（CUDA 環境では環境に応じた PyTorch CUDA wheel を導入）
pip install -r requirements.txt

# 2. 全 20 項目の数理・トレースレス BdG・厳密 HMC 監査を実行（約 3.7 秒）
python genesis_cli.py verify --save-cert results/audit_certificate_v1.0.2.json
# または標準テストランナーで実行
python -m unittest discover -s tests -v

# 3. ゼロチート・5 パネル観測ビューアを起動（固定拡大 3D＋面陰影＋inferno断面＋R^2＋QFIM）
python genesis_cli.py watch --mode birth         # 厳密 U(4) HMC：0D 量子霧 → 2D 球面誕生 vs 対照
python genesis_cli.py watch --mode melting       # 厳密 U(6) HMC：2D 球面 → 強結合量子融解
python genesis_cli.py watch --mode emergence --N 8  # 勾配緩和：0D GUE 雑音 → 2D 球面創発

# 4. リアルタイム nu / g スライダー付き 3D HMC オブザーバトリ
python genesis_cli.py watch-quantum --demo

# 5. v1.0.2 の 3 つの新コア実験を再現
python genesis_cli.py entanglement-scaling
python genesis_cli.py phase-hmc --N 4 --L 12 --beta 3.0 --chains 24
python genesis_cli.py membrane-control --N 8

# 6. 継承済み Phase 1-3 ベースラインの再現
python genesis_cli.py suite --fast
python genesis_cli.py phase1 --N 8
python genesis_cli.py large-n --Ns 16,20,24,32
python genesis_cli.py gauge --n-sphere 6
python genesis_cli.py chiral
```

---

## 7. リポジトリ構成

```text
.
├── README.md                              # 本ファイル（概要・数理モデル・成果・再現手順）
├── LICENSE                                # MIT License
├── CITATION.cff                           # 学術引用情報（Zenodo DOI 対応）
├── CHANGELOG.md                           # v1.0.0 -> v1.0.1 -> v1.0.1.2 -> v1.0.2 履歴
├── requirements.txt                       # 依存ライブラリ一覧
├── pyproject.toml                         # パッケージ設定
├── genesis_cli.py                         # 全 13 コマンド統合 CLI エントリポイント
├── genesis_rigorous/                      # 研究本体パッケージ
│   ├── safety.py                          # VRAM/RAM 3層安全ガード・ピーク VRAM 計測
│   ├── exact_hfb.py                       # 厳密 HFB・トレースレス su(N) BdG・点プローブ・QFIM
│   ├── hmc_singlet.py                     # 厳密 U(N) 一重項ゲージ格子 HMC（符号問題ゼロ）
│   ├── matrix_native.py                   # 大 N 行列直接エンジン（O(N^3) 力 / O(N^4) ラプラシアン）
│   ├── dirac_matter.py                    # 随伴ディラック物質・Daleckii-Krein 勾配
│   ├── verification.py                    # 全 20 項目マスター監査スイート（T01-T20）
│   └── experiments/                       # v1.0.2 Exp 1-3, watch, watch-quantum, Phase 1-3D
├── tests/                                 # unittest / pytest 互換自動テスト（20/20）
├── paper/                                 # LaTeX 原稿・コンパイル済 PDF・全 10 図・30 文献
├── docs/                                  # ARCHITECTURE.md（モジュール構造・計算量仕様）
└── results/                               # Exp 1-3 JSON/図・20/20 監査証明書・GIF 4 本
```

---

## 8. 科学的誠実さの境界線（射程内・射程外の明示）

本テストベッドの守備範囲を明確に定義します：
- **達成したこと**：$(0+1)$ 次元行列量子力学におけるトレースレス $\mathfrak{su}(N)$ エンタングルメント境界スケーリング則（$O(N)$ vs GUE $O(N^2)$）の統計分離、厳密 $U(N)$ 一重項格子 HMC による一次相転移・非閉じ込め核形成・強結合量子融解の直接観測、局在随伴 Dirac vs 汎用 $\mathrm{Tr}(\hat{\rho}Y^2)$ による非可換弾性膜応答の切り分け、および $U(2)$ クーロン枝ヒッグス分裂・4D 積空間・ファジー GW キラル指数の統合監査。
- **本トイモデルの射程外**：2 次元ファジー球面上ではアインシュタイン・テンソル $G_{\mu\nu} \equiv 0$ であるため、$(3+1)$ 次元ローレンツ不変な質量ゼロ・スピン 2 重力子（アインシュタイン重力）、10 次元超対称 IKKT/BFSS 模型の符号問題（Sign Problem）解決、完全なフェルミオン行列式による真空偏極、および現実の標準模型の自発創発・質量階層は本トイモデルの射程外です。これらの最前線の研究は、日本や米国の物理学研究組織や大学などで、2026年現在でもスーパーコンピューターを用いた実験が行われています。


###  1.今回のモデルが重力創発を起こせていない3つの数学的理由

今回の論文の実験3で起きていたのは本物の重力ではなく、**「ゴム膜の凹み」そのもの**です。ここは、計算を大規模化しても変わりません。

* **① 次元の壁（2次元球面の上には、重力波が住む場所すらない）**
論文の6.3節でも認めている通り、空間が2次元の曲面（球面など）では、アインシュタイン方程式の主役である「アインシュタインテンソル $G_{\mu\nu}$」が数学的に恒等的にゼロ（$G_{\mu\nu} \equiv 0$）になります。
時間1次元を足した $(2+1)$ 次元時空で考えても、一般相対性理論における「空間を伝わって重力を運ぶ波（重力子）」の自由度の数は、時空の次元 $D$ を使って $\frac{D(D-3)}{2}$ 個と決まっています。私たちが住む4次元時空（$D=4$）なら $\frac{4 \times 1}{2} = 2$ 個（縦横2種類の重力波）が存在できますが、2次元球面＋時間の3次元時空（$D=3$）を代入すると $\frac{3 \times 0}{2} = 0$ 個になってしまいます。つまり、「2次元球面」という舞台を選んだ時点で、遠くの星同士を引き寄せるアインシュタイン重力が働く余地は最初から数学的にゼロなのです。


* **② 仕組みの壁（「空間の距離の変化」ではなく「外側の空間へのへこみ」）**
本物の重力は、外部の空間へへこむのではなく、時空そのものの「ものさし（計量）」が内側から伸び縮みする現象です。対して今回の3行列模型は、あらかじめ用意された外部の3次元空間 $\mathbb{R}^3$ の中に「シャボン玉（あるいはゴム風船）のような膜」が浮かんでいて、その膜がバネのような力（質量項 $\mu$ や交換子の張力）で形を保っている状態です。
そこに物質をポンと置くと、膜が外側の3次元空間に向かって少しへこんだり、赤道が引き締まったりします。行列サイズ $N$ を大きくしても、**「ゴム風船の網目がどんどん細かくなって滑らかなゴム風船になる」だけ**であり、ゴム風船の弾性が勝手にアインシュタイン重力へ変身することはありません。


* **③ 距離の壁（遠くへ届かず、近所の重なりでしか効かない）**
本物の万有引力は、どれだけ遠く離れていても距離の2乗に反比例（$1/r^2$）して宇宙の果てまで届きます。しかし実験3で見つかった引力は、2つの物質パケットの「ぼやけた広がり（非可換幅 $\Delta\theta_{\text{NC}} \approx 40^\circ$）」が重なり合う超至近距離でしか強く働かず、少し離れると急激に湯川遮蔽されてしまいます。
むしろ、計算を大規模化して $N$ を大きくすると、球面の解像度が上がってパケットのぼやけ幅がどんどんシャープ（小さく）になるため、**重なりによる引力の射程距離は広がるどころかさらに短く縮んでしまいます**。
データ不足による暫定結果ではなく、この3行列ボソン模型の本質を射抜いた「覆らない確定事実」です。


### 2. では、「行列模型から本物の重力が生まれる」こと自体が間違いなのか？

ここで早合点してはいけないのが、「じゃあ行列模型で重力を作る研究は全部ムダなのか」と切り捨ててしまうことです。実は、**今回のモデルの「足りないピース」を2つ補った別の行列模型であれば、ゴム膜ではない「本物の重力」が創発することがすでに理論的に知られています**。
その2つのピースとは以下の通りです。

1. **行列の数を増やして「$(3+1)$ 次元以上の時空」にする**
3枚の行列（空間2次元）ではなく、超弦理論の **BFSS模型** や **IKKT模型**のように10枚の行列を使えば、時空の次元が4次元以上になり、さきほどの公式 $\frac{D(D-3)}{2}$ で重力子（スピン2）の自由度が生き残る器が手に入るという理論があります。

2. **「超対称性（フェルミオンの量子揺らぎ）」を組み込む**
これが決定的な違いです。ボソン（膜の振動）だけの世界では、膜の弾性エネルギー（ゴムの張力やバネの力）が強すぎて、重力のような繊細な長距離力はかき消されてしまいます。
ところが、そこに**フェルミオン（物質粒子の仲間）をちょうど釣り合うように加えた「超対称な行列模型」**にすると、ボソンが作る強い反発力やゴム膜の張力と、フェルミオンの量子揺らぎが作る力が**プラスマイナス・ゼロで相殺**するとされています。そして、その相殺の奥底から「わずかな量子補正（1ループ効果）」として顔を出すのが、遠く離れた2つの物体同士を質量に比例して引き寄せる「本物のニュートン重力（高次元版の逆二乗則）」だとされています。

---

## 9. 主要参考文献（Foundational References）

1. J. R. Fliss, A. Frenkel, S. A. Hartnoll, & R. M. Soni, *"Minimal Areas from Entangled Matrices"*, *SciPost Phys.* **18**, 171 (2025), [`arXiv:2408.05274`](https://arxiv.org/abs/2408.05274).
2. A. Frenkel, *"APD-Invariant Tensor Networks from Matrix Quantum Mechanics"* (2024), [`arXiv:2407.16753`](https://arxiv.org/abs/2407.16753).
3. V. Gautam, M. Hanada, & A. Jevicki, *"Operator algebra, quantum entanglement, and emergent geometry from matrix degrees of freedom"*, *JHEP* **01** (2025) 019, [`arXiv:2406.13364`](https://arxiv.org/abs/2406.13364).
4. X. Han & S. A. Hartnoll, *"Deep Quantum Geometry of Matrices"*, *Phys. Rev. X* **10**, 011069 (2020), [`arXiv:1906.08781`](https://arxiv.org/abs/1906.08781).
5. N. Kawahara, J. Nishimura, & S. Takeuchi, *"Phase structure of matrix quantum mechanics at finite temperature"*, *JHEP* **10** (2007) 097, [`arXiv:0706.3517`](https://arxiv.org/abs/0706.3517).
6. T. Azuma, S. Bal, K. Nagao, & J. Nishimura, *"Nonperturbative studies of fuzzy spheres in a matrix model with the Chern–Simons term"*, *JHEP* **05** (2004) 005, [`arXiv:hep-th/0401038`](https://arxiv.org/abs/hep-th/0401038).
7. H. C. Steinacker, *"Quantum Geometry, Matrix Theory, and Gravity"*, Cambridge University Press (2024).
8. H. Grosse, C. Klimčík, & P. Prešnajder, *"Topologically nontrivial field configurations in noncommutative geometry"*, *Commun. Math. Phys.* **178**, 507 (1996).
9. A. P. Balachandran & G. Immirzi, *"The fuzzy Ginsparg–Wilson algebra: A solution of the fermion doubling problem"*, *Phys. Rev. D* **68**, 065023 (2003), [`arXiv:hep-th/0301242`](https://arxiv.org/abs/hep-th/0301242).
10. S. Duane, A. D. Kennedy, B. J. Pendleton, & D. Roweth, *"Hybrid Monte Carlo"*, *Phys. Lett. B* **195**, 216 (1987).

---

## 10. AI Collaboration & Transparency Statement

This repository and its accompanying technical report were developed through an interactive human–AI research collaboration between the human author (**Alpha496**) and **Gemini by Google DeepMind** (using the Antigravity agentic coding environment), with targeted methodological upgrades in `v1.0.2` motivated by independent adversarial peer review of `v1.0.1`.

- **Role of AI**: Assisted with mathematical cross-checking, PyTorch/CUDA numerical implementation (including traceless $\mathfrak{su}(N)$ BdG projection, exact $U(N)$-singlet HMC force derivations, gauge-orbit zero-mode projection, and Daleckii–Krein derivative stabilization), automated verification-suite construction, and drafting of English/Japanese documentation.
- **Role & Responsibility of the Human Author**: Conceived the project scope, directed all research decisions and audit-driven revisions, executed and inspected the numerical benchmarks on local hardware, and assumes full responsibility for the scientific accuracy, honest scope boundaries, and published artifacts.
- **Reproducibility First**: No numerical claim in this project relies on AI assertions alone; every result is independently verifiable from source via the deterministic 20-test mathematical audit suite (`python genesis_cli.py verify`) and the reproducible CLI experiments documented above.
