# 研究マスター仕様書・実装ハンドオフ文書
# 行列量子力学 × 量子エンタングルメントによる時空創発シミュレータ（Project Genesis-Rigorous）

- **文書目的**：超弦理論系統の「行列模型による時空創発」と量子情報系統の「エンタングルメント／動的テンソルネットワークによる時空創発」を、2024–2026年の最先端論文に基づき**カンニング・ハードコード・ゲージ対称性破壊を一切排除した形**で統合実装するための研究仕様書兼AI間引き継ぎ文書。この文書自体は、研究が進むにしたがって、方向性が変化した場合に、書き換え可能とする。
- **対象実行環境（ユーザーPC）**：AMD Ryzen 7 5700X / NVIDIA RTX 3060 (12GB VRAM, CUDA導入済) / RAM 32GB / Python + PyTorch (CUDA自動検出＋CPUフォールバック)
- **検証方針**：
  - **Phase 1（案①・基盤実証フェーズ）**：Stage 0（メモリ安全ガード）＋ Stage 1（カオス相 vs 創発相の真の量子相転移）＋ Stage 2（100%ゲージ不変な点プローブ幾何 ＆ ブラインド粗視化QRFによる極小面積探索）。各数理モジュールは AI 側の実行環境（NumPy/SciPy）で厳密検算しつつ PyTorch CUDA 実装へ落とす。
  - **Phase 2（案②・完全統合フェーズ）**：Phase 1 の物理的妥当性を確認後、Stage 3（随伴ディラック物質の局所励起投入による計量湾曲および極小曲面の重力偏向テスト）を統合し完全版を作る。最終的に、時空創発を実現実行できる、コードができたかどうか、検証する。このあとフェーズを追加していって、拡張作業をしてもいい。

---

## 0. エージェント実行環境・絶対遵守の安全ガードレール（MANDATORY）

### 0.1 ローカル実行環境・許可ワークスペース
- **OS・シェル**: Windows 10 / PowerShell。
- **許可された作業ディレクトリ（Strict Sandbox）**:
  - `C:\QuantumUniverse1\本格的理論コード`（または `C:\QuantumUniverse` 配下の本プロジェクトフォルダのみ）。
- **ハードウェア**: AMD Ryzen 7 5700X / NVIDIA RTX 3060 (12GB VRAM, CUDA導入済) / RAM 32GB。
- **Python 実行規約**:
  - 仮想環境（`venv`）がワークスペース内に存在する場合は、有効化状態や PowerShell 実行ポリシーに左右されないよう原則 `.\venv\Scripts\python.exe <script>.py` で実行すること。
  - ※ 現時点ではまだフォルダ内に `venv` をあえて設置していないため、既存の Python / CUDA 環境（`python`）をそのまま確認して使用すること（無断でシステムPythonの破壊的変更を行わないこと）。
- **C++ / CUDA 拡張環境**:
  - Windows 環境には **Visual Studio Community 2022（MSVC `cl.exe`）** および `g++` が導入済みである。
  - 後半工程（大 $N \ge 16$ 等）で Python 用 C++ / CUDA 拡張モジュール（`.pyd` 等）のビルドが必要になった場合は、まず現在のターミナルで `cl` / `nvcc` が使えるか確認し、使えない場合は VS Developer PowerShell の初期化手順を調査して報告すること（**コンパイラの再インストール、システム環境変数の永続変更、管理者権限の使用は絶対禁止**）。

### 0.2 絶対遵守の安全・運用ガードレール（MANDATORY）
1. **許可ディレクトリ外・Windows システムへのアクセス厳禁**:
   いかなる理由があっても、`C:\QuantumUniverse1\本格的理論コード`（および `C:\QuantumUniverse` 配下の本プロジェクト領域）以外のディレクトリや Windows システムフォルダ、レジストリ、ネットワーク外部へのアクセス・操作を試みてはならない（個人情報・認証情報等の外部送信も厳禁）。
2. **ファイル削除の原則禁止**:
   コード・仕様書・結果データ・ログを問わず、`rm` / `del` / `Remove-Item` 等によるファイル削除は行わないこと（世代管理・バックアップはユーザー側および `results/` のタイムスタンプ付き保存で管理する）。
3. **バイナリ・巨大データのテキスト読み込み厳禁（トークン爆発防止）**:
   `*.pt`, `*.pth`, `*.h5`, `*.npz`, `*.png`, 巨大な JSON や大量ログファイルをテキストとして全文読み込んではならない。結果の確認は Python の要約プリントや統計値のみで行うこと。
4. **自律修正とユーザー通知の境界**:
   実装途中のバグ・テスト失敗・型エラー・CUDA/CPU の軽微な差異等は、いちいちユーザーに確認せずエージェント側で自律的に原因特定・修正してやり切ること。ユーザーへの通知・相談は **「重要な理論／設計変更」「大規模な新規ファイル構成変更の前」「物理的・仕様上の本質的な不明点」** の段階でのみ行うこと。
5. **長時間 GPU 計算の分離**:
   エージェント自身がターミナルで実行するのは数秒〜数分で完了する単体テスト（`genesis_core.py`）や中規模検証（`--N 8`, `--N 10` 等）までとし、数十分〜数時間に及ぶ大規模スケーリング探索や並列焼き戻しは CLI ワンコマンドを提示してユーザー側に実行を委ねること。

### 0.3 理論物理コードベースに対する心得と解析規律
- **サイレント・バグへの最大限の警戒**:
  本プロジェクトは一般的なアプリではなく、**「行列量子力学（HFB 4次ウィック展開）・厳密BdGフェルミオン反作用・$U(N)$ ガウス則カシミール・粗視化QRF極小面積探索」が密結合した計算理論物理学コードベース**である。符号1つ・係数2・複素共役・交換子の順序・極限順序のわずかなズレが「エラーで止まらず物理的に偽の結果を出すサイレント・バグ」を生むため、速度より深い熟考・自己点検・数学的検算（Test 1〜7）を最優先せよ。
- **優先順位**:
  「**機能性・数理論理妥当性・ハードウェア安全性（VRAM/RAM保護）**」を最優先とし、過度なリファクタリング美や不要な抽象化で検算済みの数理ロジックを壊してはならない。
- **コンテキスト効率と外部監査連携**:
  未変更のファイルを何度も全文再読せず、節目で外部監査（別セッションの Gemini 等）にかける際は、全文渡しではなく 300〜500行程度の「省トークン監査バンドル（変更目的・物理パラメータ・検証数値テーブル・重点監査項目3点）」を生成すること。

---

## 第1部：理論的土台 —— 最先端基盤論文（2020–2026）の重要抜粋

本プロジェクトは、従来分断されていた「行列の固有値／波束による時空創発」と「量子もつれ／テンソルネットワークによる時空創発」を統一する以下の3つの最重要論文（およびその系譜）を数理的土台とする。

### 基盤論文①：ガウス則エンタングルメントと「行列からの笠–高柳（極小面積）公式」の導出
- **書誌**：J. R. Fliss, A. Frenkel, S. A. Hartnoll, & R. M. Soni, *"Minimal Areas from Entangled Matrices"*, *SciPost Phys.* 18, 171 (2025), `arXiv:2408.05274`
- **関連基盤**：A. Frenkel & S. A. Hartnoll, *"Emergent Area Laws from Entangled Matrices"*, *JHEP* 05 (2023) 084, `arXiv:2301.01325`; J. R. Fliss & A. Frenkel, *"Matrix Quantum Mechanics and Entanglement Entropy: A Review"*, *Entropy* 28(1), 58 (2026), `arXiv:2512.03163`
- **重要抜粋・数理の核心**：
  1. **拡張ヒルベルト空間と $U(N)$ ガウス則**：
     行列 $\hat{X}_I$ は古典数ではなく正準運動量 $\hat{\Pi}_I$ を持つ量子演算子であり、物理状態は拡張ヒルベルト空間 $\mathcal{H}_{\text{ext}} = \bigotimes_{I=1}^D L^2(\mathrm{Herm}_N)$ 上で $U(N)$ ガウス則 $\hat{G}_{ab}|\Psi\rangle = 0$ を満たす状態に制限される。
  2. **部分領域射影と4ブロック分解**：
     階数 $M$ の直交射影行列 $\Theta_\Sigma$（$\Theta_\Sigma^2 = \Theta_\Sigma,\; \mathrm{Tr}\,\Theta_\Sigma = M$、固定体積 $|\Sigma| \propto M/N$）により、行列は内部ブロック $X_{\Sigma\Sigma}$、境界開弦ブロック $X_{\Sigma\bar{\Sigma}}, X_{\bar{\Sigma}\Sigma}$、外部ブロック $X_{\bar{\Sigma}\bar{\Sigma}}$ に分解され、$\mathcal{H}_{\text{in}} = \mathcal{H}_{\Sigma\Sigma}$、$\mathcal{H}_{\text{out}} = \mathcal{H}_{\Sigma\bar{\Sigma}} \otimes \mathcal{H}_{\bar{\Sigma}\bar{\Sigma}}$ となる。
  3. **$U(N)$ ゲージ対称性の2つの役割**：
     - **内部部分群 $U(M)\times U(N-M)$**：境界をまたぐ開弦揺らぎのせいで、部分ブロック上のガウス則が $\hat{Q}_\Sigma + \hat{Q}_{\bar{\Sigma}} = 0$ となり、境界に **$U(M)$ 既約表現 $\mu$ の最大もつれエッジモード**を強制誘起する。その表現次元 $\log d_\mu$ は2次カシミール $\langle \mathrm{Tr}(\hat{Q}_\Sigma^2)\rangle$ を通じて**幾何学的境界面積 $|\partial\Sigma|$ に比例する（面積則）**。
     - **剰余類（Wiggle modes） $V \in \mathcal{F}_\Lambda \equiv G_\Lambda / [U(M)\times U(N-M)]$**：固定体積のもとで領域形状 $\Theta_V = V^\dagger \Theta_0 V$ を体積保存変形する粗視化量子参照系（QRF）の平均化により、縮約密度行列は形状 $V$ の直和 $\hat{\rho}_{\text{in}} \sim \bigoplus \int_{\mathcal{F}_\Lambda} dV \, \hat{\rho}_V \otimes \frac{I_{\mu_V}}{d_{\mu_V}}$ となる。
  4. **極小面積の創発と極限順序**：
     レニー・エントロピーのホログラフィック極限（**固定 $n>1$ で半古典極限 $N\to\infty,\; g_{\text{eff}}\to 0$ を先にとり、その後に $n\to 1$ へ解析接続**）をとると、鞍点評価 $\int dV\, d_{\mu_V}^{1-n} \approx \exp(-(n-1)\min_V \log d_{\mu_V})$ により **最小境界面積 $\min_{|\Sigma|} |\partial\Sigma|$（笠–高柳公式の極小化原理）** が自発的に出現する。
     - *重要条件*：$U(N)$ 全体で平均すると不連続な非幾何学的分割が多数派を占めるため、長波長平滑化スケール $\Lambda$ をもつ粗視化群 $G_\Lambda \subseteq U(N)$ への制限が不可欠である。

### 基盤論文②：行列量子力学への背景独立テンソルネットワークの直接埋め込み
- **書誌**：A. Frenkel, *"APD-Invariant Tensor Networks from Matrix Quantum Mechanics"* (2024), `arXiv:2407.16753`
- **重要抜粋・数理の核心**：
  1. 従来のテンソルネットワーク（MERA 等）は格子グラフを人間が事前固定する（背景依存である）という弱点を持つ。本論文は、ネットワークの内部リンク自由度を $U(N)$ 不変な行列量子力学のヒルベルト空間へ直接埋め込むことで**背景独立性**を実現する。
  2. **非可換ラプラシアンによる動的局所性**：
     座標行列 $\hat{X}_I$ から作る非可換ラプラシアン $\Delta_{\hat{X}}(\hat{O}) \equiv \sum_{I=1}^D [\hat{X}_I, [\hat{X}_I, \hat{O}]]$ を、リンク自由度の遠距離減衰ポテンシャル $\mathcal{F}_\Lambda(\Delta_{\hat{X}}) = g_1 \Delta_{\hat{X}} + g_2 \Delta_{\hat{X}}^2 + \cdots$ としてハミルトニアンに組み込む。
  3. **弦スケールと面積則**：
     全リンクをベル対にしようとするもつれ生成項と、距離2乗で長距離リンクを抑制する $\mathcal{F}_\Lambda(\Delta_{\hat{X}})$ の競合により、動的な**弦スケール（非局所スケール）$\tilde{\ell}_s$** が生まれる。$\tilde{\ell}_s$ より大きなスケールでは任意の曲線切断 $\Theta_\Sigma$ に対して面積則エンタングルメントが成立し、大 $N$ 極限の $U(N)$ 対称性は**面積保存微分同相写像（APD）不変性**となる。
  4. **本プロジェクトでの数理的厳密化（重要修正点）**：
     原論文の単一リンク成分ハードコア量子ビット表記は基底回転で $U(N)$ 不変性を破り、自由ボソンでは $\mathcal{F}(\Delta_{\hat{X}})<2$ でエネルギーが $-\infty$ へ発散し、1フレーバー反交換フェルミオンでは $\mathrm{Tr}(\hat{\chi}^2) \equiv 0$ となる。これを解決するため、本プロジェクトでは **2フレーバー対の反交換随伴フェルミオン $\hat{\chi}^{\alpha,\pm}$** を採用し、**厳密な $U(N)$ 不変性・非ゼロ対合・パウリ排他律による無条件の下方有界性**をすべて同時に満たす形に改良する。

### 基盤論文③：行列固有値創発と量子エンタングルメント創発の統一（変分行列波束とバルク作用素環）
- **書誌**：V. Gautam, M. Hanada（花田政範）, & A. Jevicki, *"Operator algebra, quantum entanglement, and emergent geometry from matrix degrees of freedom"*, *JHEP* 01 (2025) 019, `arXiv:2406.13364`
- **関連基盤**：V. Gautam, M. Hanada, A. Jevicki, & C. Peng, *"Matrix Entanglement"*, *JHEP* 01 (2023) 003, `arXiv:2204.06472`; X. Han & S. A. Hartnoll, *"Deep Quantum Geometry of Matrices"*, *Phys. Rev. X* 10, 011069 (2020); S. R. Das et al., *JHEP* 2021 (`arXiv:2012.15683`); A. Tsuchiya et al., *JHEP* 2024
- **重要抜粋・数理の核心**：
  1. **変分行列波束 $|Y, Q\rangle \in \mathcal{H}_{\text{ext}}$**：
     個別の固有値ではなく行列空間 $\mathbb{R}^{D N^2}$ 全体の上で、中心条件 $\langle \hat{X}_I \rangle = Y_I,\; \langle \hat{\Pi}_I \rangle = Q_I$ のもとにエネルギー期待値 $\langle \Phi | \hat{H} | \Phi \rangle$ を最小化する量子波束として幾何状態を定義する。
  2. **真空差し引き後の実効幾何サイズ**：
     量子零点揺らぎにより真空（$Y_I=0$）でも $\langle \mathrm{Tr}\,\hat{X}_I^2 \rangle = O(N^2)$ となるため、創発幾何の広がりは揺らぎを差し引いた $\mathrm{Tr}(Y_I^2) = \langle \mathrm{Tr}\,\hat{X}_I^2 \rangle - \langle \mathrm{Tr}\,\delta\hat{X}_I^2 \rangle$ で測る。
  3. **大 $N$ 超選択と開弦スクイージング**：
     ゲージ射影状態 $\hat{\mathcal{P}}_{U(N)}|Y, Q\rangle$ の重なりは $\exp(-\frac{1}{8}\sum_I \mathrm{Tr}(Y_I - U^\dagger Y_I U)^2)$ で抑制され、大 $N$ で古典幾何が超選択セクターとして確定する。ヤン・ミルズ相互作用 $-\frac{g^2}{4}\mathrm{Tr}[\hat{X}_I, \hat{X}_J]^2$ により、離れたブロック間を結ぶ非対角開弦は振動数 $\omega_{ij} \propto |\vec{y}_i - \vec{y}_j|$ でスクイーズされ、局所バルク作用素環 $\mathcal{A}(\mathcal{R})$ が創発する。
  4. **純ボソン模型における量子相転移（Han–Hartnoll 2020 / Das et al. 2021）**：
     非対角モードの量子零点揺らぎは空間を原点 $Y_I=0$ へ引き潰す閉じ込め引力として働く。フラックス $\nu$ やフェルミオン反作用と量子揺らぎの競合により、**「量子カオス液体相（$Y_I=0$）」と「非可換幾何凝縮相（$Y_I\neq 0$）」の真の量子相転移**が生じる。

---

## 第2部：完全監査済みの統合数理モデル

### 2.1 単一統合マスター母関数（ゲージ化レプリカ経路積分）
3論文の数理構造と Lorentz 型 IKKT 極限（$\partial_t \to 0$）を単一式に圧縮した生成汎関数（大 $N$ 超選択下での単一 $V$ 積分表現として漸近等号 `≃` を明示）：

$$
\begin{aligned}
\mathcal{Z}_n\!\big[M, \Lambda; J\big]
\;\simeq\;& \int_{\mathcal{F}_\Lambda} dV \int_{\mathrm{Gluing}_n(\Theta_V)} \mathcal{D}A_0\,\mathcal{D}X\,\mathcal{D}\chi\,\mathcal{D}\Psi \;\exp\!\Bigg\{\, i \int_{\mathcal{C}_n} dt \; \mathrm{Tr}\!\Bigg[\;
\frac{m}{2}\sum_{I=1}^D (D_t X_I)^2 + \frac{g^2}{4}\sum_{I \neq J}[X_I, X_J]^2 - W_{\text{inv}}(X) \\
&\quad + \sum_{\alpha=1}^{N_f}\!\left(
  \sum_{s=\pm} \chi^{\alpha,s\dagger}\!\Big(i D_t - \kappa\,\mathcal{F}_\Lambda(\Delta_X)\Big)\chi^{\alpha,s}
  - i\kappa\Big(\chi^{\alpha,+\dagger}\chi^{\alpha,-\dagger} - \chi^{\alpha,-}\chi^{\alpha,+}\Big)
\right) \\
&\quad + \Psi^\dagger\!\left(i D_t - \sum_{I=1}^D \gamma^I [X_I,\,\cdot\,] - m_\psi \gamma^0\right)\Psi
+ J \cdot \mathcal{O}(X, \chi, \Psi)
\;\Bigg] \Bigg\}
\end{aligned}
$$

- **Block 1（定義）**：$n$ 枚レプリカ Schwinger–Keldysh 経路 $\mathcal{C}_n$ 上の Lorentz 型作用（Yang–Mills 交換子2乗＋2フレーバー対リンク $\chi$＋随伴 Dirac 物質 $\Psi$＋ソース $J\cdot\mathcal{O}$）。$\partial_t \to 0,\; \kappa=0,\; D=9$ で Lorentz 型 IKKT 模型へ帰着。
- **Block 2（場とガウス則）**：共変微分 $D_t = \partial_t - i[A_0, \cdot]$ の $A_0$ 積分が全セクター共通の $U(N)$ ガウス則 $\hat{G}_{ab} = 0$ を自動強制。$\chi^{\alpha,\pm}$ は厳密 $U(N)$ 不変かつ下方有界、$\Psi$ は随伴表現として $U(1)$ 中性。
- **Block 3（幾何・TN・部分領域の出現）**：$n=1$ の変分波束中心 $Y_I = \langle X_I \rangle$ がカオス相（$Y_I=0$）から幾何相（$Y_I\neq 0$）へ相転移し、共通の $\Delta_X = \sum_I [X_I,[X_I,\cdot]]$ が開弦局所性とリンク $\chi$ の弦スケール $\tilde{\ell}_s$ を同時決定。射影 $\Theta_V = V^\dagger \Theta_0 V$（$\mathrm{Tr}\,\Theta_V = M,\; V \in \mathcal{F}_\Lambda$）が固定体積領域を切り出す。
- **Block 4（大 $N$・レプリカ極限と変分の帰結）**：ホログラフィック極限（$N\to\infty,\; g_{\text{eff}}\to 0$ 後に $n\to 1$）で合成 $U(M)$ エッジモード＋一重項 TN／バルクもつれの極値化（一般化笠–高柳公式）を与え、ソース変分 $\delta_J$ が物質反作用による計量湾曲とエンタングルメント第1法則 $\delta_J S_{\text{EE}} = \delta_J\langle \hat{K}_\Sigma \rangle$ を与える。

### 2.2 ハミルトニアン形式の厳密規約（数値実装の直接基準）
行列随伴を $(\hat{O}^\dagger)_{ab} \equiv (\hat{O}_{ba})^\dagger$ とし、単一規約（$\hbar = 1$）で定義する：

$$
\big[\hat{X}^I_{ab},\, \hat{\Pi}^J_{cd}\big] = i\,\delta^{IJ}\delta_{ad}\delta_{bc}, \qquad
\big\{\hat{\chi}^{\alpha,s}_{ab},\, (\hat{\chi}^{\beta,s'\dagger})_{cd}\big\} = \delta^{\alpha\beta}\delta^{ss'}\delta_{ad}\delta_{bc}, \qquad
\big\{\hat{\Psi}^A_{ab},\, (\hat{\Psi}^{B\dagger})_{cd}\big\} = \delta^{AB}\delta_{ad}\delta_{bc}
$$

エルミートかつ無トレース（$\mathrm{Tr}\,\hat{G} = 0$）で全セクターに同一変換 $\delta\hat{O} = +i[\varepsilon,\hat{O}]$ を生成する $U(N)$ ガウス則生成子：
$$
\hat{G}_{ab}
= i\sum_{I=1}^D \Big([\hat{X}_I, \hat{\Pi}_I]_{ab} - iN\delta_{ab}\Big)
+ \sum_{\alpha=1}^{N_f}\sum_{s=\pm} \Big(\hat{\chi}^{\alpha,s\dagger}\hat{\chi}^{\alpha,s} + \hat{\chi}^{\alpha,s}\hat{\chi}^{\alpha,s\dagger} - N I_N\Big)_{ab}
+ \sum_{A} \Big(\hat{\Psi}^{A\dagger}\hat{\Psi}^A + \hat{\Psi}^A\hat{\Psi}^{A\dagger} - N I_N\Big)_{ab}
$$

統合ハミルトニアン（本実装の $D=3$ 基準形、将来 $D>3$ 拡張可）：
$$
\hat{H}_{\text{univ}} = \hat{H}_{\text{MQM}} + \hat{H}_{\text{TN}} + \hat{H}_{\text{matter}}
$$
$$
\hat{H}_{\text{MQM}} = \mathrm{Tr}\!\left(\frac{1}{2m}\sum_{I=1}^3 \hat{\Pi}_I^2 - \frac{g^2}{4}\sum_{I\neq J}[\hat{X}_I, \hat{X}_J]^2 + \frac{\mu^2}{2}\sum_{I=1}^3 \hat{X}_I^2 + i\frac{\nu g}{3}\sum_{I,J,K=1}^3 \epsilon_{IJK}\hat{X}_I[\hat{X}_J, \hat{X}_K]\right)
$$
$$
\hat{H}_{\text{TN}} = \kappa \sum_{\alpha=1}^{N_f} \mathrm{Tr}\!\left[i\Big(\hat{\chi}^{\alpha,+\dagger}\hat{\chi}^{\alpha,-\dagger} - \hat{\chi}^{\alpha,-}\hat{\chi}^{\alpha,+}\Big) + \sum_{s=\pm} \hat{\chi}^{\alpha,s\dagger}\,\mathcal{F}_\Lambda(\Delta_{\hat{X}})\,\hat{\chi}^{\alpha,s}\right]
$$
$$
\hat{H}_{\text{matter}} = \mathrm{Tr}\!\left(\sum_{I=1}^3 \hat{\Psi}^\dagger \gamma^I [\hat{X}_I, \hat{\Psi}] + m_\psi \hat{\Psi}^\dagger \gamma^0 \hat{\Psi}\right)
$$

### 2.3 科学的な誠実さのための境界線（過大主張の禁止）
本モデルについて、以下の3点は**明確に認識・維持すること**（誇張禁止）：
1. **$\chi$ セクターの位置づけ**：$\kappa = 0$ でも $X_I$ 自身の $U(N)$ ガウス則エッジモードにより行列創発と極小面積則の本質的統合は既に成立する。$\chi^{\alpha,\pm}$ は動的テンソルネットワークのボンド構造を顕在化させ、かつ $\kappa > 0$ で時空へ反作用する結合層である。
2. **$(0+1)$次元モデルの守備範囲**：本モデルは空間座標 $\vec{x}$ を持たない $(0+1)$ 次元行列量子力学であり、創発するのは**非可換Dブレーン世界体積幾何**である。カットオフ $\Lambda$ は非可換空間上のUV粗視化スケールであり、境界CFTから $(d+1)$ 次元双曲 AdS 奥行き $z$ を生む MERA アイソメトリと同一視してはならない。
3. **非解決領域の明示**：10次元超対称 IKKT の符号問題による $SO(9)\to SO(3)$ 次元選択や、ローレンツ不変な質量ゼロ・スピン2重力子の厳密導出は本トイモデルの射程外であり、それを達成したかのように出力・描画してはならない。

---

## 第3部：厳禁事項（旧コード `genesis_pure_ode_simulator.py` の失敗を繰り返さない監査ルール）

実装を担当する AI・開発者は、以下の **5つの偽装（Anti-Patterns）を絶対にコードへ混入させてはならない**：

1. **【厳禁①】答えのハードコード**：`R0_sq = (alpha**2) * (N**2 - 1.0) / 4.0` のような単一既約表現のカシミール拘束項 $S_{\text{cas}} = \|\sum X_I^2 - R_0^2 I_N\|^2$ をポテンシャルに入れてはならない。
   - *理由*：$\hat{H}_{\text{MQM}}$ の古典＋量子ポテンシャルでは、$\nu > \nu_{\text{crit}}$ のとき負エネルギー $v(r_*) < 0$ により大カシミール表現（単一既約球面）が**自然に**最安定となるため、拘束項は一切不要。
2. **【厳禁②】手動の「神の手」鞍点脱出キック**：`eigh(X[2])` で固有値ソートして三重対角行列 `diag(1, 1) + diag(1, -1)` を `X[0]` に足すような非ゲージ共変・非回転不変な手動介入を一切禁止する。
   - *代替*：完全に $U(N)\times SO(3)$ 等方的なランジュバン雑音（熱／量子アニーリング、または並列焼き戻し）のみを使用する。
3. **【厳禁③】古典行列のみの進化と事後フェルミオン捏造**：$\hat{\Pi}_I$ を無視して古典行列 $X_I$ だけを動かし、収束後に JAD で対角成分を捨てて人工格子フェルミオン `A_off + L_hop` を被せることを厳禁とする。
   - *代替*：量子共分散 $(\Sigma_x, \Sigma_p)$ を持つ変分波束と、厳密な BdG 反作用 $E_{\text{TN}}(Y)$ を常に連成させる。
4. **【厳禁④】連続極限（$N\to\infty$）で破綻する閾値チューニング**：`0.05 < rel_var < 0.33` のように $N$ が大きく滑らかな連続時空に近づくほど `FAIL` になる小 $N$ 専用ハードコード判定を禁止する。
5. **【厳禁⑤】見かけ倒しの検証ゲート**：$\sqrt{-\log I}$ の凹変換による自明な三角不等式パスや、ランダムGUEでも通過する `n_zero == 1` を「時空創発の証拠」として扱ってはならない。必ず **Universe A（カオス相・不合格になるべき対照実験）** を同一コードで走らせ、明確に分離できることでのみ合格とする。

---

## 第4部：数値アルゴリズム詳細仕様（PyTorch CUDA ＆ NumPy 検証用）

### Stage 0：ハードウェア安全ガード（Ryzen 7 5700X / RTX 3060 12GB / RAM 32GB）
コード冒頭に以下を必ず実装する：
1. **CUDA VRAM ハードリミット**：`torch.cuda.set_per_process_memory_fraction(0.30)`（RTX 3060 12GB 中 最大 3.6 GB に強制制限）。
2. **CPU RAM 監視**：`psutil.Process().memory_info().rss`（または `tracemalloc`）が `MAX_RAM_MB = 4096`（4 GB）を超過したら例外を送出して安全終了。
3. **プリフライト見積もり**：実行前に $N, D$ から $n_B = D N^2$ と $O(n_B^2)$ 配列サイズを計算し、安全上限（標準検証 $N \in [6, 12]$、最大上限 $N \le 16$）を超過する場合は割り当て前に拒否。
4. **`SIGINT` (`Ctrl+C`) ハンドラ**：途中結果を JSON/PNG に保存し `torch.cuda.empty_cache()` を呼んで安全終了。

| 行列サイズ $N$ | $n_B = 3N^2$ | 精度行列 $\Omega$ (`float64`) | ラプラシアン $\Delta_Y$ | 推定ピーク VRAM | 推定ピーク RAM | 用途 |
| :-: | :-: | :-: | :-: | :-: | :-: | :--- |
| **$N=6$** | 108 | 0.09 MB | 0.01 MB | ~50 MB | ~200 MB | AIサンドボックス高速厳密単体テスト |
| **$N=8$** | 192 | 0.29 MB | 0.03 MB | ~80 MB | ~250 MB | AIサンドボックス検証 ＆ PC高速確認 |
| **$N=10$** | 300 | 0.72 MB | 0.08 MB | ~120 MB | ~300 MB | **Phase 1 標準実験サイズ** |
| **$N=12$** | 432 | 1.49 MB | 0.16 MB | ~200 MB | ~350 MB | Phase 1/2 高精細実験サイズ |
| **$N=16$** | 768 | 4.72 MB | 0.52 MB | ~450 MB | ~500 MB | スケーリング検証上限 |

---

### Stage 1（Phase 1 中核）：変分行列波束 HFB × 厳密 BdG 反作用 × 2系統対照実験

#### 1.1 正規直交エルミート基底と変分状態
$N\times N$ エルミート行列の正規直交基底 $\{T^A\}_{A=1}^{N^2}$（$\mathrm{Tr}(T^A T^B)=\delta^{AB}$）を構成する：
- 対角 $N$ 個：$E_{aa}$
- 実対称非対角 $\frac{N(N-1)}{2}$ 個：$\frac{1}{\sqrt{2}}(E_{ab}+E_{ba})$（$a<b$）
- 虚反対称非対角 $\frac{N(N-1)}{2}$ 個：$\frac{i}{\sqrt{2}}(E_{ab}-E_{ba})$（$a<b$）

ボソン量子状態は、中心 $y \in \mathbb{R}^{n_B}$（$Y_I = \sum_A y_{I,A} T^A$）と対称正定値精度行列 $\Omega \in \mathbb{R}^{n_B \times n_B}$ を持つガウス波束 $|\Phi_X(y,\Omega)\rangle$ とする：
$$
\Sigma_x = \frac{1}{2}\Omega^{-1} = \sum_{r=1}^{n_B} \frac{1}{2\omega_r} v^{(r)}(v^{(r)})^T, \qquad
\Sigma_p = \frac{1}{2}\Omega = \sum_{r=1}^{n_B} \frac{\omega_r}{2} v^{(r)}(v^{(r)})^T
$$
ここで固有ベクトル $v^{(r)}$ を $D$ 個のエルミート $N\times N$ 行列 $\{V^{(r)}_I\}_{I=1}^D$ に戻して用いる（$\sum_I \mathrm{Tr}(V^{(r)}_I V^{(s)}_I) = \delta_{rs}$）。

#### 1.2 厳密な量子エネルギー期待値（全項解析閉形式・近似ゼロ）
1. **量子運動＋質量＋古典エネルギー**：
   $$E_{\text{kin}} = \frac{1}{4m}\sum_{r=1}^{n_B} \omega_r, \qquad E_{\text{mass}} = \frac{\mu^2}{2}\left(\sum_I \mathrm{Tr}(Y_I^2) + \sum_{r=1}^{n_B}\frac{1}{2\omega_r}\right)$$
   $$E_{\text{cl}}^{\text{YM+flux}} = -\frac{g^2}{4}\sum_{I\neq J}\mathrm{Tr}[Y_I, Y_J]^2 + i\frac{\nu g}{3}\sum_{I,J,K}\epsilon_{IJK}\mathrm{Tr}\big(Y_I[Y_J,Y_K]\big)$$
2. **2次揺らぎ（開弦ヘッセ行列＋フラックス揺らぎ）**：
   $\sigma_r^2 \equiv \frac{1}{2\omega_r}$ として：
   $$E_{\text{fluc}}^{(2)} = \sum_{r=1}^{n_B} \sigma_r^2 \left[ -\frac{g^2}{4}\sum_{I\neq J}\mathrm{Tr}\!\Big(\big([Y_I,V^{(r)}_J] + [V^{(r)}_I,Y_J]\big)^2 + 2[Y_I,Y_J][V^{(r)}_I,V^{(r)}_J]\Big) + i\nu g \sum_{I,J,K}\epsilon_{IJK}\mathrm{Tr}\big(Y_I[V^{(r)}_J, V^{(r)}_K]\big) \right]$$
   *(※ 3次フラックス項の2次縮約は $\epsilon_{IJK}$ の完全反対称性より3つの位置が同値で係数 $3 \times \frac{1}{3} = 1$ となることを検算済み。)*
3. **4次ヤン・ミルズ厳密ウィック1ループ補正（全3縮約・$O(N^6)$ テンソル不要！）**：
   $$E_{\text{YM}}^{(4)} = -\frac{g^2}{4}\sum_{I\neq J}\sum_{r,s=1}^{n_B} \sigma_r^2 \sigma_s^2 \;\mathrm{Tr}\!\Big( [V^{(r)}_I, V^{(r)}_J][V^{(s)}_I, V^{(s)}_J] + [V^{(r)}_I, V^{(s)}_J]^2 + [V^{(r)}_I, V^{(s)}_J][V^{(s)}_I, V^{(r)}_J] \Big)$$
   - *厳密検算値*：$Y_I = 0$、等方共分散 $\Sigma_x = \sigma^2 I_{n_B}$ のとき $E_{\text{YM}}^{(4)} = \frac{1}{2}g^2 D(D-1) N(N^2-1)\sigma^4$ と解析的に厳密一致する（Unit Test 1 の検証対象）。
4. **動的TNリンクフェルミオンの厳密 BdG 基底エネルギー（論文②＋直接反作用）**：
   非可換ラプラシアン $\Delta_Y(O) = \sum_{I=1}^D [Y_I,[Y_I,O]]$ の固有値 $\lambda_k \ge 0$（$k=1,\dots,N^2$）とフィルタ $f_k = \mathcal{F}_\Lambda(\lambda_k) = g_1\lambda_k + g_2\lambda_k^2$ に対し、2フレーバー対 $\hat{\chi}^{\alpha,\pm}$ の厳密基底エネルギー：
   $$E_{\text{TN}}(Y) = \kappa N_f \sum_{k=1}^{N^2}\left(f_k - \sqrt{f_k^2 + 1}\right)$$
   *(※ $\partial E_{\text{TN}} / \partial Y_I$ は `autograd` により $Y_I$ の最適化へ直接反作用する。)*

#### 1.3 自己無撞着 HFB 最適化と等方アニーリング
- **内側ループ（$\Omega$ の自己無撞着更新）**：固定 $Y$ のもとで実効ヘッセ行列 $\mathcal{K}_{\text{eff}}(Y, \Sigma_x) = \frac{\partial^2 E_{\text{cl}}}{\partial y^2} + 2\frac{\partial E_{\text{YM}}^{(4)}}{\partial \Sigma_x}$ を計算し、ゲージゼロモード（$\delta Y_I = i[\varepsilon, Y_I]$ 方向の小さな負/零固有値）を物理的正則化 $\omega_{\min}$ で保護しつつ $\Omega \leftarrow \sqrt{m\,\mathcal{K}_{\text{eff}}}$ を不動点反復で解く。
- **外側ループ（$Y_I$ の緩和）**：$U(N)\times SO(3)$ 完全等方的なランジュバン雑音 $\sqrt{2 T(\tau) \eta}\,\xi_I$（温度 $T(\tau)$ を高温から $0$ へ滑らかに冷却、または多初期値マルチスタート）付きの勾配降下で $E_{\text{tot}}(Y, \Omega^*(Y))$ を最小化する。
- **必須の2系統同時対照実験（反証テスト）**：
  - **Universe A（強量子カオス対照相）**：$\nu < \nu_{\text{crit}}$（例：$\mu=1.0, g=0.35, \nu=0.8$ または $\nu=0.0$）。**目標結果：$Y_I \to 0$（真空差し引き後の幾何サイズ $\mathrm{Tr}\,Y_I^2 \approx 0$）となり、時空が創発しないこと。**
  - **Universe B（創発幾何相）**：$\nu > \nu_{\text{crit}}$（例：$\mu=1.0, g=0.35, \nu=2.2$）。**目標結果：ヒントゼロの等方ランダム初期値から $Y_I \neq 0$ の非可換多様体が自発的に凝縮すること。**

---

### Stage 2（Phase 1 中核）：100% ゲージ不変な幾何抽出 ＆ ブラインド QRF 極小面積探索

#### 2.1 点プローブ・ディラック幾何学（Berenstein–Dzienkowski / Steinacker）
JAD や基底固定を一切使わず、テスト座標 $\vec{y} \in \mathbb{R}^3$ 上で $2N \times 2N$ エルミート演算子
$$D_{\text{probe}}(\vec{y}) = \sum_{I=1}^3 (Y_I - y_I I_N) \otimes \sigma_I$$
を対角化する。
1. **創発多様体の抽出**：最小絶対固有値 $|\lambda_{\min}(\vec{y})|$ の等値面／極小軌跡が創発空間 $\mathcal{M} \subset \mathbb{R}^3$ を100% $U(N)$ ゲージ不変に描き出す（Universe A では原点付近のボケた塊、Universe B では鋭い球殻が浮かぶ）。
2. **リーマン量子計量テンソル**：最小固有状態 $|\psi_0(\vec{y})\rangle$ の数値微分／摂動展開から Provost–Vallée 計量
   $$g_{ab}(\vec{y}) = \mathrm{Re}\,\langle \partial_a \psi_0 | \big(I_{2N} - |\psi_0\rangle\langle\psi_0|\big) | \partial_b \psi_0 \rangle$$
   を抽出し、創発空間の局所的な距離計量と曲率を直接測定する。

#### 2.2 厳密な部分領域エンタングルメント3成分（固定階数 $M = \mathrm{Tr}\,\Theta_V$）
任意の直交射影 $\Theta_V = V^\dagger \Theta_0 V$（$\Theta_0 = \mathrm{diag}(1,\dots,1,0,\dots,0)$、階数 $M$）に対し、内部射影 $O_{\Sigma\Sigma} \equiv \Theta_V O \Theta_V$ を用いて以下を厳密計算する：
1. **厳密な $U(M)$ ガウス則内部カラー電荷カシミール（論文①・近似ゼロの閉形式！）**：
   $$\left\langle \mathrm{Tr}\!\big((\hat{Q}^X_\Sigma)^2\big) \right\rangle = \frac{1}{2}\sum_{r=1}^{n_B} \omega_r \left\| \sum_{I=1}^3 \big[Y^I_{\Sigma\Sigma},\, V^{(r)}_{I,\Sigma\Sigma}\big] \right\|_F^2 + \frac{1}{8}\sum_{r,s=1}^{n_B} \left(\sqrt{\frac{\omega_s}{\omega_r}} - \sqrt{\frac{\omega_r}{\omega_s}}\right)^2 \left\| \sum_{I=1}^3 \big[V^{(r)}_{I,\Sigma\Sigma},\, V^{(s)}_{I,\Sigma\Sigma}\big] \right\|_F^2$$
   *(※ 等方自由真空 $\omega_r = \omega_0,\; Y=0$ で恒等的に $0$ になることを検算済み。ガウス則エッジモード寄与は $\log d_{\mu_X} \propto \sqrt{\langle\mathrm{Tr}((\hat{Q}^X_\Sigma)^2)\rangle}$ で評価。)*
2. **ボソン内部ブロックの厳密シンプレクティック・エントロピー $S_{\text{bos}}(\Theta_V)$**：
   部分ブロック $\Sigma\Sigma$（自由度数 $n_{\text{in}} = 3M^2$）へ制限した位置・運動量共分散行列 $\Sigma_{x,\text{in}}, \Sigma_{p,\text{in}}$ から、積行列 $\Sigma_{x,\text{in}} \Sigma_{p,\text{in}}$ の固有値 $\nu_a^2 \ge \frac{1}{4}$ を求め、
   $$S_{\text{bos}}(\Theta_V) = \sum_{a=1}^{3M^2} \left[\left(\nu_a + \frac{1}{2}\right)\log\!\left(\nu_a + \frac{1}{2}\right) - \left(\nu_a - \frac{1}{2}\right)\log\!\left(\nu_a - \frac{1}{2}\right)\right]$$
   を厳密計算する。
3. **動的TNリンク $\hat{\chi}^{\alpha,\pm}$ の厳密 BdG 切断エントロピー $S_{\text{TN}}(\Theta_V)$（論文②・Peschel公式）**：
   $\Delta_Y$ の固有モード $E_k$ のボゴリューボフ角 $\cos 2\theta_k = \frac{f_k}{\sqrt{f_k^2+1}},\; \sin 2\theta_k = \frac{1}{\sqrt{f_k^2+1}}$ から、厳密な一般化1粒子相関行列
   $$G_{\text{BdG}} = \begin{pmatrix} \langle \hat{\chi}^{+\dagger}\hat{\chi}^+ \rangle & \langle \hat{\chi}^{+\dagger}\hat{\chi}^{-\dagger} \rangle \\ \langle \hat{\chi}^-\hat{\chi}^+ \rangle & \langle \hat{\chi}^-\hat{\chi}^{-\dagger} \rangle \end{pmatrix}$$
   を $\Sigma\Sigma$ 部分空間（次元 $2M^2 \times 2M^2$）へ射影して固有値 $\zeta_j \in (0,1)$ を求め、
   $$S_{\text{TN}}(\Theta_V) = -N_f \sum_{j=1}^{2M^2} \Big[\zeta_j \log \zeta_j + (1-\zeta_j)\log(1-\zeta_j)\Big]$$
   を厳密計算する。

#### 2.3 ブラインド粗視化 QRF 極小面積探索（Fliss–Frenkel–Hartnoll–Soni 2025 の直接実証）
- **生成子**：非可換ラプラシアン $\Delta_Y$ の低固有値モード（$\lambda_k \le \Lambda$、例：低周波 $K_\Lambda$ 個の滑らかなエルミート行列 $E_k$）により、粗視化ユニタリ群の元 $V(c) = \exp\!\big(i\sum_{k=1}^{K_\Lambda} c_k E_k\big) \in G_\Lambda$ をパラメータ化する。
- **ブラインド最適化**：ランダム初期パラメータ $c^{(0)}$ から出発し、**座標や幾何情報を一切与えずに**、エンタングルメント指標（$S_{\text{edge}}(\Theta_V) + S_{\text{TN}}(\Theta_V) + S_{\text{bos}}(\Theta_V)$）のみを勾配最適化で最小化する。
- **成功判定（非循環テスト）**：
  - **Universe B（創発相）**：最適解 $\Theta_{V^*}$ を後から独立に幾何空間へ投影（$\langle \Theta_{V^*} \vec{Y} \Theta_{V^*} \rangle$ の固有状態分布やプローブ密度）したとき、**最小周長の連結な球帽（笠–高柳極小曲面）が自動発見され**、体積 $M$ を変えると $S_{\min}(M) \propto \sqrt{M(N-M)}$（2次元球帽の周長法則 $\sin\theta_c$）に従うこと。
  - **Universe A（カオス相）**：同一の探索で明確な最小周長キャップが形成されず、体積則（Page型）的な振る舞いを示すこと。

---

### Stage 3（Phase 2 拡張）：物質投入による時空湾曲実験（創発重力テスト）
Phase 1 の全検証ゲート通過後に実装する：
1. 創発した Universe B の上の1点 $\vec{y}_0$（例：北極方向）に局在する随伴ディラック物質状態 $|\Phi_{\text{matter}}(\vec{y}_0)\rangle$（$\hat{H}_{\text{matter}} = \mathrm{Tr}(\hat{\Psi}^\dagger \gamma^I [X_I, \hat{\Psi}] + m_\psi \hat{\Psi}^\dagger \gamma^0 \hat{\Psi})$ の局所波束励起、または局所ソース $J(\vec{y}_0)$）を導入する。
2. 物質エネルギー期待値 $\langle \Phi_{\text{matter}} | \hat{H}_{\text{matter}}(Y) | \Phi_{\text{matter}} \rangle$ を $E_{\text{tot}}(Y, \Omega)$ に加えて全系を再平衡化する。
3. 投入前後（$M_{\text{src}} = 0$ vs $M_{\text{src}} > 0$）の比較により、以下を定量測定する：
   - **計量の湾曲**：点プローブ最小特異値曲面 $\lambda_{\min}(\vec{y})=0$ と Provost–Vallée 量子計量 $g_{ab}(\vec{y})$ の $\vec{y}_0$ 近傍での局所変形（曲率変化 $\delta R(\vec{y}_0)$）。
   - **極小曲面の重力偏向**：Stage 2 のブラインド QRF 最小化で得られる極小エンタングルメント面 $\Theta_{V^*}^*$ の境界形状とエントロピー変化 $\delta S_{\text{EE}}$（エンタングルメント第1法則との整合）。

---

## 第5部：必須の逐次検証テスト一覧（Verification & Falsification Checklist）

Phase 1 完了（＝Phase 2 進行許可）のため、以下の **7つの厳密テストをすべて通過**すること（最初は AI サンドボックスの NumPy/SciPy で数値証明し、その後 PyTorch CUDA 版と機械精度で一致確認する）：

- [ ] **Test 1（$U(N)$ ゲージ不変性テスト）**：任意のランダムユニタリ変換 $Y_I \to U Y_I U^\dagger$ に対して、$E_{\text{cl}}, E_{\text{fluc}}^{(2)}, E_{\text{YM}}^{(4)}, E_{\text{TN}}, \lambda_{\min}(\vec{y}), \langle\mathrm{Tr}(\hat{Q}^X_\Sigma)^2\rangle, S_{\text{bos}}, S_{\text{TN}}$ のすべてが $10^{-11}$ 以内で厳密不変であること。
- [ ] **Test 2（4次ウィック展開の解析・MC一致テスト）**：$Y=0$・等方共分散 $\sigma^2 I$ で $E_{\text{YM}}^{(4)} = \frac{1}{2}g^2 D(D-1)N(N^2-1)\sigma^4$ と一致し、さらに一般の非等方 $\Sigma_x$ に対して独立モンテカルロ標本平均と統計誤差内で一致すること。
- [ ] **Test 3（厳密ガウス則カシミールの零点テスト）**：等方自由真空（$Y=0,\; \omega_r=\omega_0$）で $\langle\mathrm{Tr}((\hat{Q}^X_\Sigma)^2)\rangle = 0$（$< 10^{-13}$）となること。
- [ ] **Test 4（厳密 BdG リンクエントロピーテスト）**：単一ペア $N=2$ において、Peschel 公式の出力 $S_{\text{TN}}$ が厳密対角化および解析式 $H_2\!\big(\frac{1}{2}(1 - \frac{f}{\sqrt{1+f^2}})\big)$ と $10^{-12}$ 以内で一致すること。
- [ ] **Test 5（反証対照実験：Universe A vs Universe B 相転移）**：同一コード・等方ランダム初期値から、Universe A（$\nu < \nu_c$）では $\mathrm{Tr}(Y_I^2) \approx 0$ のカオス相へ緩和し、Universe B（$\nu > \nu_c$）では $R_0^2$ 拘束なし・手動キックなしで非可換球殻 $Y_I \neq 0$ が自発凝縮すること。
- [ ] **Test 6（ブラインド QRF 極小面積テスト）**：Universe B において、ランダムな滑らか射影 $V \in G_\Lambda$ からエンタングルメントのみを最小化すると、最小周長の球帽（$M$ 依存性が球帽周長 $\propto \sqrt{M(N-M)}$ に相関）が自動抽出され、Universe A では抽出されないこと。
- [ ] **Test 7（ハードウェア安全テスト）**：RTX 3060 / RAM 32GB 上で $N=10, 12$ 実行時のピーク VRAM < 1 GB、ピーク RAM < 1 GB であり、意図的に巨大 $N$（例：$N=64$）を指定した際にプリフライト検査で安全停止すること。

---

## 第6部：引き継ぎプロトコル（別の Gemini インスタンスへの行動指針）

この `.md` ファイルを受け取った Gemini インスタンスは、以下の手順で研究を遂行すること：

1. **禁止事項の厳守**：第3部の5つの厳禁事項（`R0_sq` ハードコード、三重対角キック、古典近似、JAD偽ディラック、小 $N$ 専用閾値）をいかなる理由でも再導入しない。
2. **段階的検証（Phase 1 → Phase 2）**：
   - まず **Step 1-A** として、`Test 1`〜`Test 4` を通過する厳密数理コア（HFB 量子エネルギー・厳密 BdG・厳密ガウス則カシミール・点プローブ計量）を作成・検証する。
   - 次に **Step 1-B** として、Stage 1（Universe A vs Universe B の2系統相転移）と Stage 2（ブラインド QRF 極小面積探索）を実装し、`Test 5`〜`Test 7` を通過させる。
   - Step 1-A/B の実測データ（数値テーブルおよび可視化図）をユーザーと確認した上で、**Phase 2（Stage 3：随伴ディラック物質投入による時空湾曲・創発重力テスト）** を統合する。
3. **誠実な報告**：もし数値実験で相転移が起きない・極小面が収束しない・有限 $N$ 効果が強く出るなどの現象が起きた場合、**絶対にパラメータや式をごまかして「成功」と偽装せず**、物理的原因（例：$\nu_{\text{crit}}$ の量子シフト、粗視化カットオフ $\Lambda$ の選定、有限 $N$ での超選択の甘さ等）を率直に解析してユーザーと議論すること。

---

## 第7部：ローカル自律エージェント（Antigravity）実行・保存・C++最適化・Stage 3 進行プロトコル

本プロジェクトをローカル環境（Ryzen 7 5700X / RTX 3060 12GB CUDA / RAM 32GB）の自律エージェント（Antigravity）で遂行する際は、以下の厳格な手順とゲート管理に従うこと。

### 7.1 段階的データ保存ルール
- すべての実行結果（コンソールログ、PNG図、JSONレポート、パラメータ設定、再現用シード）は上書きせず、タイムスタンプおよびフェーズ名付きのディレクトリ（例：`results/phase1_N8_YYYYMMDD_HHMMSS/`）に段階保存すること。
- 各ステップ終了ごとに `results/EXPERIMENT_LOG.md` に「実行条件・実測数値・物理的考察・発見された課題（失敗も含む）」を追記すること。

### 7.2 自律実行のフェーズ進行ゲート（チェックポイント制）
1. **Checkpoint 0（環境＆数理コア確認）**：
   - `python genesis_core.py` を実行し、CUDA 認識と **Test 1（誤差 `< 1e-11`）、Test 2A（`< 1e-12`）、Test 3（`< 1e-13`）、Test 7（`N=64` ブロック）** がすべて `PASS` することを確認する。
2. **Checkpoint 1（Phase 1 実証：`N=8` → `N=10` → `N=12`）**：
   - `python genesis_run_phase1.py --N 8` を実行し、以下を定量確認する：
     - **Universe A（カオス対照相）**：$\mathrm{Tr}(Y^2) \approx 0$ に留まり、点プローブに殻構造が出ないこと。
     - **Universe B（創発相）**：$\mathrm{Tr}(Y^2) \gg 0$ へ自発凝縮し、点プローブの断面図に鮮明な極小軌跡（多様体）が出ること。
     - **ブラインド QRF 探索**：Universe B においてエンタングルメント最小化のみで独立な非可換周長 `perim_opt < perim_init` が有意に減少し、$S_{\min}(M)$ が面積則形状と整合すること。
   - **重要**：もし単一球面ではなく準安定マルチ球面（例：$j_1 \oplus j_2$）に落ちた場合、**絶対に手動キックや `R0_sq` を復活させず**、物理的に正当な方法（`--restarts` 増加、並列焼き戻し Parallel Tempering、または温度スケジュール調整）のみで改善を試み、マルチ球面が出た事実自体も物理的成果として記録すること。
   - `N=8` で正常確認できたら `--N 10`, `--N 12` へスケールアップし、$N$ 依存性（大 $N$ 連続極限の振る舞い）を `results/` に蓄積する。
3. **Checkpoint 2（Phase 2 / Stage 3：物質投入による創発重力実験の実装と実行）**：
   - Checkpoint 1 の全条件を満たした後にのみ、`genesis_stage3_gravity.py`（または統合完全版）を新規作成する。
   - Stage 3 では、創発した Universe B の背景上の1点 $\vec{y}_0$ に随伴ディラック物質 $\hat{H}_{\text{matter}} = \mathrm{Tr}(\hat{\Psi}^\dagger \gamma^I [X_I, \hat{\Psi}] + m_\psi \hat{\Psi}^\dagger \gamma^0 \hat{\Psi})$ の局所励起状態（質量源 $M_{\text{src}}$）を投入し、全系再平衡化前後での **(a) Provost–Vallée 量子計量 $g_{ab}(\vec{y})$ と局所曲率の変化**、**(b) ブラインド QRF 極小曲面 $\Theta_{V^*}^*$ の偏向と $\delta S_{\text{EE}}$** を定量検証する。

### 7.3 C++ / CUDA C++ の利用許可と数理一致条件
- `N <= 12` の検証では PyTorch CUDA（`float64`/`complex128`）を標準とする。
- 大 $N$ スケーリング（`N >= 16`）や長時間の並列焼き戻し・実時間発展で `f_abc` の $O(N^6)$ メモリや Python ループがボトルネックになった場合、**C++ / CUDA C++（PyTorch C++ Extension / `pybind11` / LibTorch 等）による高速カーネル実装（`f_abc` を保持せず直接交換子演算で $O(N^4)$ 化する実装など）を許可する**。
- ただし C++ / CUDA 実装を導入する際は、必ず既存の PyTorch 参照実装（`genesis_core.py`）との出力差が **`< 1e-11` であることを単体テストで証明**してから本番実験へ投入すること。

### 7.4 絶対遵守ルール（再掲）
- 第3部の5つの厳禁事項を破る変更を一切行わないこと。
- 実験が失敗・不収束・予想外の振る舞いを示した場合、数値や閾値を改ざんせず、物理的原因を解析して `results/EXPERIMENT_LOG.md` に正直に報告すること。