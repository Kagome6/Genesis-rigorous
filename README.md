# Project Genesis-Rigorous

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%2012.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24537568.svg)](https://doi.org/10.5281/zenodo.24537568)

**Zero-Cheat Matrix Quantum Mechanics × Quantum Entanglement Laboratory for Emergent Spacetime, Gravity, Non-Abelian Gauge Bosons, and Chiral Matter**

<p align="center">
  <img width="1627" height="965" alt="watch_emergence_N8" src="https://github.com/user-attachments/assets/af2db573-2fd6-4038-ab45-2b923bdd8b9e" />
</p>
<p align="center">
  <em><strong>ゼロチート・ライブ観測（<code>python genesis_cli.py watch --mode emergence</code>）</strong>：座標も球面メッシュも一切使わず、純粋な U(8) 等方ランダム行列から球面時空（宇宙 B）が自発的に結晶化し、量子フィッシャー情報計量（QFIM）が 3D カオスから厳密ランク 2 の 2 次元多様体へ収束する一方、同一初期乱数の対照宇宙 A（&nu; = 0.80）が 0 次元へ崩壊する実測動画。</em>
</p>

---

## 1. このプロジェクトの目的と概要（Overview & Purpose）

**「あらかじめ空間の座標や距離を用意しなくても、純粋な“行列”と“量子もつれ”だけから、空間の広がり・面積則・重力応答・因果律（光円錐）・ゲージ場・キラル素粒子は本当に立ち上がるのか？」**

近年の理論物理学（行列量子力学・ホログラフィー・量子情報幾何）では、時空そのものを微視的な行列自由度とエンタングルメントから再構成する研究が急速に進展しています（Fliss–Frenkel–Hartnoll–Soni 2025, Frenkel 2024, Gautam–Hanada–Jevicki 2025, Han–Hartnoll 2020, Steinacker 2024 等）。しかし、これらを実際に数値計算で確かめようとすると、以下のような壁が立ちはだかります：

- 厳密対角化や大規模モンテカルロ法は計算コストが莫大で、個人環境では再現が難しい。
- 一方で簡易的なトイコードでは、「目標半径をポテンシャルに書き込む」「最初から球面に近い初期値を与える」「反証用の対照実験を省く」といった**暗黙のカンニング（誘導バイアス）**が混入しやすい。

**Project Genesis-Rigorous** はこのギャップを埋めるために開発した、**100% $U(N)$ ゲージ不変・カンニング排除（Zero-Cheat）・対照実験必須の計算理論物理テストベッド**です。個人向け GPU（NVIDIA RTX 3060 12GB、計算時間 数秒〜数分）上で、数理監査から時空創発・重力応答・大 $N$ 連続極限・ゲージ場・キラル素粒子までを一貫した単一コード基盤で再現検証・可視化できることを目的としています。

> **本研究の誠実な位置づけ**
> 本プロジェクトは「新しい量子重力理論の提唱」ではなく、1992–2025 年の先行理論研究（Fliss–Frenkel–Hartnoll–Soni 2025, Frenkel 2024, Gautam–Hanada–Jevicki 2025, Han–Hartnoll 2020, Steinacker 2024, Grosse–Klimčík–Prešnajder 1996, Balachandran–Immirzi 2003 等）の核心機構を単一コード基盤上で統合・数値監査し、再現可能な形で検証した **計算物理ベンチマーク・再現テストベッド** です（リンク・フェルミオン部は Frenkel 2024 に着想を得た、下に有界かつ Peschel 公式で厳密可解な 2 フレーバー BdG 代理模型を採用しています）。

---

## 2. 解いている統合数理モデル（Core Unified Hamiltonian）

本リポジトリの全実験は、背景時空や座標ラベルを一切持たない $(0+1)$ 次元エルミート行列 $\hat{X}_I$（$I=1,\dots,D$）と量子リンク・物質自由度を連成させた、以下の単一 $U(N)$ ゲージ不変ハミルトニアン $\hat{H}_{\mathrm{univ}} = \hat{H}_{\mathrm{MQM}} + \hat{H}_{\mathrm{TN}} + \hat{H}_{\mathrm{matter}}$ に基づいています：

$$
\begin{aligned}
\hat{H}_{\mathrm{univ}} \;=\;
&\;\mathrm{Tr}\!\left[
  \frac{1}{2m}\sum_{I=1}^{D}\hat{\Pi}_I^2
  - \frac{g^2}{4}\sum_{I\neq J}[\hat{X}_I, \hat{X}_J]^2
  + \frac{\mu^2}{2}\sum_{I=1}^{D}\hat{X}_I^2
  + i\frac{\nu g}{3}\sum_{I,J,K}\epsilon_{IJK}\,\hat{X}_I[\hat{X}_J,\hat{X}_K]
\right] \\[6pt]
&+ \;\kappa\sum_{\alpha=1}^{N_f}\mathrm{Tr}\!\left[
  \sum_{s=\pm}\hat{\chi}^{\alpha,s\dagger}\,\mathcal{F}_\Lambda(\Delta_{\hat{X}})\,\hat{\chi}^{\alpha,s}
  + i\bigl(\hat{\chi}^{\alpha,+\dagger}\hat{\chi}^{\alpha,-\dagger} - \hat{\chi}^{\alpha,-}\hat{\chi}^{\alpha,+}\bigr)
\right] \\[6pt]
&+ \;\mathrm{Tr}\!\left[
  \hat{\Psi}^\dagger\!\left(\sum_{I=1}^{3}\alpha^I[\hat{X}_I, \cdot\,] + m_\psi\beta\right)\!\hat{\Psi}
\right],
\qquad
\Delta_{\hat{X}}(\cdot) \equiv \sum_{I=1}^{D}[\hat{X}_I,[\hat{X}_I,\cdot\,]]
\end{aligned}
$$

- **1行目（ボソン行列量子力学 $\hat{H}_{\mathrm{MQM}}$）**：ヤン＝ミルズ交換子 2 乗＋ Myers フラックス。フラックス $\nu$ が臨界値 $\nu_c = \sqrt{2}\,\mu$ を超えると、無構造な乱数行列から 2 次元／4 次元の非可換ブレーン空間が自発凝縮し、その上の揺らぎから $U(2)$ ゲージ場や実時間光円錐が現れます（自己無撞着 HFB ガウス量子波束で 4 次ウィック展開まで厳密評価）。
- **2行目（2 フレーバー BdG リンク代理模型 $\hat{H}_{\mathrm{TN}}$）**：非可換ラプラシアン $\Delta_{\hat{X}}$ を介して空間的に近い行列モード間だけを強くもつれさせ、Peschel 公式により厳密な部分領域エンタングルメント $S_{\mathrm{TN}}$ とブラインド極小面積則を駆動します。
- **3行目（随伴ディラック物質 $\hat{H}_{\mathrm{matter}}$）**：点プローブ・コヒーレント波束上の単一粒子正エネルギー期待値 $E_{\mathrm{matter}} = M_{\mathrm{src}}\langle\sqrt{\mathcal{D}_{\mathrm{adj}}^2}\rangle$ として結合し（縮退球面での `0/0` 自動微分破綻は Daleckii–Krein 有理化差分商で完全解消）、局所曲率集中や 2 体間の曲率媒介引力をテストします。

> **創発空間の「座標無し測定」**：得られた行列 $Y_I$ の幾何は、外部座標を仮定せず、$U(N)$ ゲージ不変な点プローブ・ディラック演算子 $D_{\mathrm{probe}}(\vec{y}) = \sum_I (Y_I - y_I I_N)\otimes\Gamma_I$ の零点殻と、その基底状態の **Provost–Vallée 計量（＝量子フィッシャー情報計量 $g^{\mathrm{QFI}} = 4 g^{\mathrm{PV}}$）** によって次元・計量・固有ガウス曲率を直接測定します。

---

## 3. 何を行ったのか？（段階的検証ロードマップ）

本プロジェクトでは、いきなり複雑な現象を主張するのではなく、以下の 6 段階を順に積み上げて検証しました。また、科学的な重みづけを明確にするため、**「① 非自明な創発・力学ベンチマーク」**と**「② 解析恒等式・トポロジカル指数の厳密検算」**を明確に区別しています：

### ① 非自明な創発・力学ベンチマーク（最適化・実時間発展の結果として自発出現）
1. **Phase 1（2系統相転移 ＆ ブラインド極小面積則, $N=8,10,12$）**：全く同じ等方ランダム行列から開始し、$\nu=0.80$（宇宙 A）は 0 次元へ完全崩壊する一方、$\nu=3.00$（宇宙 B）のみが単一ファジー球面（および小スケール初期値では準安定な同心二重球面 $3\oplus 5$）へ自発凝縮することを確認。さらに座標ヒントゼロで Grassmannian 上の量子エンタングルメントを最小化すると、極小面積の球帽（龍＝高柳型面積則）が自動発見されることを検証。
2. **Stage 3（局在物質による創発曲率・計量・極小曲面偏向, $N=8,10$）**：北極に随伴ディラック物質波束を置き、HFB＋BdG＋幾何の全系を再平衡化。北極でのガウス曲率集中・計量密度増加・エンタングルメント増加（南極対照群は不変）と、ブラインド極小半球の物質回避偏向を測定。
3. **Phase 3A（2体間の曲率媒介引力 ＆ 実時間光円錐, $N=8$）**：直接結合ゼロの 2 体の物質間で、幾何固定コントロール（相互作用ゼロ）と動的時空（湯川型の引力）を比較。さらに北極パルス後のシンプレクティック実時間発展で、有限 $N$ 非可換重なり領域を除いた遠方因果区間における亜光速の光円錐伝播を計測。
4. **Phase 3B（$O(N^3)$ 行列直接エンジンによる大 $N=16\dots 32$ 連続極限）**：$O(N^6)$ の構造定数テンソル $f^{ABC}$ を全廃し、厳密な $(D-1)/D$ 1 ループ係数を持つ $O(N^3)$ エンジンを開発。VRAM わずか `8.2 MB` で $N=32$（1,024 モード）の連続極限スケーリングと準安定マルチ球面盆地を同時検証。
5. **Phase 3D-1（$D=6, N=16$ からの 4 次元積空間 $S^2\times S^2$ 自発因子分解）**：目標半径を含まない純粋 6 行列系から等方乱数で緩和させ、交差交換子が自発的にゼロへ落ちて 6D 点プローブ計量が厳密ランク 4（4 次元創発空間）になることを確認。

### ② 解析恒等式・トポロジカル指数の厳密検算（実装健全性・耐ノイズ性の機械精度検証）
6. **Phase 3C（重なりブレーン $U(2)$ ゲージボソン ＆ クーロン枝行列ヒッグス分裂, $N=12, D=4$）**：2 枚の球面を外部パラメータ $v$ で分離したときの 4D クーロン横射影ヘッセ行列を対角化し、ゴールドストーン 2 モード吸収・対角光子の不変性・非対角 $W^\pm$ の $\Delta\omega_W^2 = g^2 v^2$ および $SU(2)$ 4 次自己結合を $10^{-13}$ 精度で検算。
7. **Phase 3D-2（ファジー Ginsparg–Wilson キラル素粒子 3 世代 ＆ 耐ノイズ保護）**：Balachandran–Immirzi (2003) 系の双基本キラル・ディラック演算子 $\mathcal{D}_{\mathrm{GW}}$ に対し、磁束差 $q=0,1,2,3$ の厳密指数 $\mathrm{Ind}(\mathcal{D}_{\mathrm{GW}})=q$ と大振幅ランダム行列ノイズ下での質量ゼロ保護を $10^{-16}$ 精度で検算。

---

## 4. 科学的厳格性の保証（5つの厳禁ルール ＆ 17項目自動監査）

### 5つの厳禁ルール（Zero-Cheat Policy）
1. **答えのハードコード禁止**：単一球面半径を強制するカシミール拘束項（`R0_sq = alpha^2 (N^2-1)/4`）を一切使用しない。
2. **手動の鞍点脱出キック禁止**：固有値ソートや三重対角行列の注入といった非ゲージ共変な介入を行わず、100% $U(N)\times SO(D)$ 等方ランダム GUE 初期雲のみを使用。
3. **古典近似＋事後捏造の禁止**：量子共分散 $(\Sigma_x, \Sigma_p)$ を持つ HFB 波束と厳密 BdG リンクフェルミオン反作用を常に連成。
4. **小 $N$ 専用チューニングの禁止**：$N = 8, 10, 12, 16, 20, 32$ へスケールしても一貫して成立する物理量のみを評価。
5. **反証対照実験の義務化**：**Universe A（カオス崩壊相）**・**幾何固定コントロール**・**南極対照点**・**可換 $U(1)$ コントロール**・**大振幅量子ノイズ**を同一コードで同時実行。

### マスター数理・ゲージ不変性監査（`python genesis_cli.py verify`：`17/17 PASS`, `2.06秒`, VRAM `68 MB`）
- $U(N)$ ゲージ不変性（全 7 観測量）：最大誤差 **`5.68e-14`**
- 4 次ウィック展開 $E_{\mathrm{YM}}^{(4)}$：等方解析誤差 **`2.66e-15`**／一般非等方 20 万標本 GPU モンテカルロ **`0.26σ`**
- $N=2$ 厳密対角化（256 次元 Fock 空間）vs Peschel BdG 公式：**`dE = 1.55e-15`, `dS = 3.33e-16`**
- 曲がった全宇宙（$M=N$）での $U(N)$ 一重項物理射影 $P_{\mathrm{phys}}$ ガウス則ゼロ：**`1.47e-27`**（HFB Hohenberg–Martin ギャップ汚染の除去）
- 厳密 $SO(3)$ 縮退球面での Daleckii–Krein 有理化差分商勾配：**`RelErr = 2.46e-09`**（特異点・`NaN` ゼロ）
- $O(N^3)$ 行列直接エンジンと厳密 HFB コアの一致（$Y=0$ 厳密一致＋$Y\neq 0$ ヘッセトレース $\mathrm{Tr}(H^{(2)}_{\mathrm{YM}}) = (D-1)g^2\mathrm{Tr}(\Delta_Y)$）：**`MaxErr = 7.11e-15`（`TrH2 = 1.87e-16`）**

---

## 5. 主要な検証結果サマリー（RTX 3060 12GB 実測）

| フェーズ | 区分 / 検証対象 | 主要な実測結果（対照実験との比較） | 計算時間 / VRAM |
| :--- | :--- | :--- | :---: |
| **Phase 1** | **[創発]** 2系統量子相転移・点プローブ幾何・ブラインド極小面積則（$N=8, 10, 12$） | Universe A: `Tr(Y^2)=0.0000` 崩壊 vs Universe B: 単一既約球面自発凝縮（`AlgErr = 2.1e-7..6.4e-7`）。Provost–Vallée 計量が理論値 $\frac{N}{4r^2}$ と一致。ブラインド最小化のみで周長 **`-68.3%〜-73.7%`** 縮小、非対称化前の生リンクエントロピー相関 **`r(S_TN) = 0.996–0.9997`**（A は `-0.22`）。 | `1.5–3.6分` / `24–93 MB` |
| **Stage 3** | **[創発]** 局在随伴ディラック物質（単一粒子正エネルギー源）による弾性・曲率応答（$N=8, 10$） | 北極（物質位置）の固有ガウス曲率が **`+19.2%`** 集中（$r = \mathbf{+0.9992}$, $N=10$ で `+0.9997`）、PV 面積要素 $+10.3\%$ 集中（$r = \mathbf{+0.9995}$）。北極帽 $S_{\mathrm{EE}}$ が `+27.8%` 増加する一方、南極対照は `< 0.25%` 不変。ブラインド極小半球が物質を避けて反極側へ偏向（$N=8$: `cosθ: -0.585 → -0.903`, $N=10$: `+0.935 → -0.591`）。 | `1.8–2.1分` / `24–43 MB` |
| **Phase 3A** | **[創発]** 2体間の曲率媒介引力 ＆ 実時間光円錐（$N=8$） | 直接結合ゼロの 2 体物質間で、幾何固定コントロール `1.7e-5` に対し動的時空で **`V_grav(25°) = -8.77 → 40°: -4.01 → 60°: -0.56 → 180°: 0.00`** の湯川型引力を検出。固定共分散 Verlet 450 ステップ発展（エネルギー保存 `4.45e-05`）で、有限 $N$ 非可換重なり領域（$\theta \le 80^\circ$）を除いた遠方因果区間 $[85^\circ, 160^\circ]$ において **`c_ang = 1.483 rad/t`（`r = 0.9983`、YM 理論上限 $g\alpha=2.808$ 以下の亜光速群速度）**を計測。 | `~7分` / `15 MB` |
| **Phase 3B** | **[創発]** $O(N^3)$ エンジン（厳密 $(D-1)/D$ 1ループ係数）による大 $N=16\dots 32$ 連続極限 | $N=16, 20$ で新規等方乱数から単一球面へ収束（`AlgErr = 6.8e-8, 4.3e-8`, `alpha = 11.2363`）。$N=32$（1,024 モード）単一球面も `8.2 MB` VRAM で理論計量 $N/(4r^2)$・$N(N^2-1)$ エネルギー則と 4–5 桁一致（$N=24, 32$ の 4 シード準安定マルチ球面盆地も併記）。 | `37–229秒` / **`8.2 MB`** |
| **Phase 3D-1** | **[創発]** $D=6, N=16$ からの 4次元積空間 $S^2\times S^2$ 自発因子分解 | 6D Chaos A 崩壊 vs 6D Emergent B が交差交換子誤差 `3.35e-09` で $12\oplus 3\oplus 1$ 可約積構造へ自発凝縮し、6D 点プローブ計量が **厳密ランク 4（`[0, 0, 0.00642, 0.00642, 0.00866, 0.00866]`、理論値と 5 桁一致）** を出力。 | **`58秒`** / `9.0 MB` |
| **Phase 3C** | **[恒等式検算]** 重なりブレーン $U(2)$ ゲージボソン ＆ クーロン枝行列ヒッグス分裂（$N=12, D=4$） | $v=0$ で 12 モード厳密縮退（`3.3e-13`）。外部ブレーン分離 $v>0$ で物理モード `436→434`（ゴールドストーン 2 個吸収）、対角 $U(1)\times U(1)$ 光子ズレ `2.1e-13`（基底 $\omega^2=\mu^2=1$ 不変）、非対角 $W^\pm$ が厳密ヒッグス式 $\Delta\omega_W^2 = g^2 v^2$ と **`r = 1.00000000`（誤差 `2.2e-13`）** で一致。非可換 $SU(2)$ 4 次自己結合も検算。 | **`3.0秒`** / `24 MB` |
| **Phase 3D-2** | **[指数検算]** ファジー Ginsparg–Wilson（Balachandran–Immirzi 2003 系）キラル素粒子 3 世代 | 双基本ファジー GW キラル・ディラック演算子（$\{\mathcal{D}_{\mathrm{GW}},\gamma_5\}=0$ `1.8e-15`）で磁束差 $q=0,1,2,3$ に対し代数指数恒等式 $\mathrm{Ind}(\mathcal{D}_{\mathrm{GW}})=q$ に従う質量ゼロ片巻き世代が **`0, 1, 2, 3` 世代** 出現し、大振幅量子ノイズ下でも `|E_0| <= 6.6e-16` で保護されることを確認。 | *(3D に含む)* |

---

## 6. クイックスタート ＆ ライブ観測ビューア（再現手順）

### 環境要件
- Python 3.10+ / PyTorch 2.x（CUDA GPU 推奨、CPU 自動フォールバック対応）
- 検証済み環境：AMD Ryzen 7 5700X / NVIDIA RTX 3060 (12GB VRAM, CUDA 12.6) / RAM 32GB

```bash
# 1. 依存パッケージ導入（CUDA 環境では環境に応じた PyTorch CUDA wheel を導入）
pip install -r requirements.txt

# 2. 全 17 項目の数理・ゲージ不変性監査を実行（約 2 秒）
python genesis_cli.py verify
# または標準テストランナーで実行
python -m unittest discover -s tests -v

# 3. ゼロチート・ライブ観測ビューアを起動（GUI ウィンドウが自動で立ち上がります）
python genesis_cli.py watch --mode emergence    # 0D 乱数 → 球面時空 vs 宇宙 A 崩壊（約 21 秒）
python genesis_cli.py watch --mode metastable   # 同心二重球面 (2⊕6) への自発分裂（約 20 秒）
python genesis_cli.py watch --mode wave         # 北極パルス → 南極フォーカス重力波（約 8 秒）

# 4. 統合ショーケースを一括実行（--fast は N=6 + ゲージ + 4D キラル 約 1.5 分）
python genesis_cli.py suite --fast

# 5. 各実験フェーズを個別に再現
python genesis_cli.py phase1 --N 8
python genesis_cli.py gravity --state-pt results/<phase1_dir>/phase1_states.pt
python genesis_cli.py realtime --state-pt results/<phase1_dir>/phase1_states.pt
python genesis_cli.py large-n --Ns 16,20,24,32
python genesis_cli.py gauge --n-sphere 6
python genesis_cli.py chiral
```

---

## 7. リポジトリ構成

```text
.
├── README.md                             # 本ファイル（概要・数理モデル・成果・再現手順）
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
│   └── experiments/                      # Phase 1〜3D 実験群 ＆ spacetime_viewer.py
├── tests/                                # unittest / pytest 互換自動テスト
├── paper/                                # Overleaf 向け LaTeX テクニカルレポート原稿
├── docs/                                 # 研究仕様書・アーキテクチャ設計文書
└── results/                              # 代表的な検証図・JSON・GIF・EXPERIMENT_LOG.md
```

---

## 8. 科学的誠実さの境界線（射程内・射程外の明示）

本テストベッドの守備範囲を明確に定義します：
- **達成したこと**：$(0+1)$ 次元行列量子力学からの 2D/4D 非可換 D ブレーン世界体積創発、龍–高柳（Ryu–Takayanagi）型極小面積則、静的・動的（2 体湯川型引力・光円錐）な非可換ブレーン弾性／曲率応答、重なりブレーン上の $U(2)$ ゲージボソン＆クーロン枝行列ヒッグス分裂、およびファジー Ginsparg–Wilson 演算子によるトポロジカル保護された質量ゼロ・キラル素粒子世代の**統一的かつ厳格な対照実験付き数値検証**。
- **本トイモデルの射程外**：10 次元超対称 IKKT 模型の符号問題（Sign Problem）の厳密解決による $SO(9)\to SO(3)$ 動的次元選択、$(3+1)$ 次元ローレンツ不変な質量ゼロ・スピン 2 重力子の厳密散乱振幅、完全なフェルミオン行列式による真空偏極、および現実の標準模型の質量比・結合定数の第一原理導出。

---

## 9. 主要参考文献（Foundational References）

1. J. R. Fliss, A. Frenkel, S. A. Hartnoll, & R. M. Soni, *"Minimal Areas from Entangled Matrices"*, *SciPost Phys.* **18**, 171 (2025), [`arXiv:2408.05274`](https://arxiv.org/abs/2408.05274).
2. A. Frenkel, *"APD-Invariant Tensor Networks from Matrix Quantum Mechanics"* (2024), [`arXiv:2407.16753`](https://arxiv.org/abs/2407.16753).
3. V. Gautam, M. Hanada, & A. Jevicki, *"Operator algebra, quantum entanglement, and emergent geometry from matrix degrees of freedom"*, *JHEP* **01** (2025) 019, [`arXiv:2406.13364`](https://arxiv.org/abs/2406.13364).
4. X. Han & S. A. Hartnoll, *"Deep Quantum Geometry of Matrices"*, *Phys. Rev. X* **10**, 011069 (2020), [`arXiv:1906.08781`](https://arxiv.org/abs/1906.08781).
5. H. C. Steinacker, *"Quantum Geometry, Matrix Theory, and Gravity"*, Cambridge University Press (2024).
6. H. Grosse, C. Klimčík, & P. Prešnajder, *"Topologically nontrivial field configurations in noncommutative geometry"*, *Commun. Math. Phys.* **178**, 507 (1996).
7. A. P. Balachandran & G. Immirzi, *"The fuzzy Ginsparg–Wilson algebra: A solution of the fermion doubling problem"*, *Phys. Rev. D* **68**, 065023 (2003), [`arXiv:hep-th/0301242`](https://arxiv.org/abs/hep-th/0301242).
8. A. Chatzistavrakidis, H. Steinacker, & G. Zoupanos, *"Intersecting branes and a standard model realization in matrix models"*, *JHEP* **09** (2011) 115, [`arXiv:1107.0265`](https://arxiv.org/abs/1107.0265).

---

## 10. AI Collaboration & Transparency Statement

This repository and its accompanying technical report were developed through an interactive human–AI research collaboration between the human author (**Alpha496**) and **Gemini by Google DeepMind** (Gemini 4 Neon, using the Antigravity agentic coding environment).

- **Role of AI**: Assisted with mathematical cross-checking, PyTorch/CUDA numerical implementation, debugging of matrix-model edge cases (e.g., gauge-orbit zero-mode projection and Daleckii–Krein derivative stabilization), automated verification-suite construction, and drafting of English/Japanese documentation.
- **Role & Responsibility of the Human Author**: Conceived the project scope, directed all research decisions and external-audit revisions, executed and inspected the numerical benchmarks on local hardware, and assumes full responsibility for the scientific accuracy, honest scope boundaries, and published artifacts.
- **Reproducibility First**: No numerical claim in this project relies on AI assertions alone; every result is independently verifiable from source via the deterministic 17-test mathematical audit suite (`python genesis_cli.py verify`) and the reproducible CLI experiments documented above.
