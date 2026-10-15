# sd-webui-reForge-ODE

English | [日本語](#日本語)

reForge-ODE Sampler is an independent sampler extension that makes reForge's built-in ODE sampling available with its own settings and separate solver choices for txt2img and Hires.fix. It uses [torchdiffeq](https://github.com/rtqichen/torchdiffeq) and provides 11 solvers.

## Why a separate sampler?

In my comparisons, reForge's built-in ODE produced images with a noticeably different atmosphere and drawing style from [TDE Sampler](https://github.com/seti9585/sd-webui-tde-sampler) and [RK Sampler](https://github.com/seti9585/sd-webui-rk-sampler). The difference was large enough that I considered it worth keeping as an independent sampler and extension.

The aim is to retain access to that rendering character, with independent controls for everyday use. The cause of the image differences has not been established; this release does not claim a particular numerical explanation or that one sampler produces better images than the others.

On reForge, the extension uses the native `sample_ode` function when available, reForge's `AlterSampler.get_sigmas` path, and the host `KDiffusionSampler` generation flow. A fallback based on reForge's ODE implementation is included. Results still depend on the host version and generation environment; identical output across versions or other WebUIs is not guaranteed.

## Installation and tested environments

In Extensions → Install from URL, enter:

```text
https://github.com/seti9585/sd-webui-reForge-ODE
```

Restart the WebUI after installation. `install.py` installs `torchdiffeq` if it is missing. If manual installation is needed, run this in the Python environment used by your WebUI:

```bash
pip install torchdiffeq
```

Operation has been confirmed by the author on **reForge only**, in these environments:

| WebUI | Python | PyTorch |
|---|---|---|
| reForge | 3.10.11 | 2.9.0+cu128 |
| reForge | 3.12.10 | 2.11.0+cu128 |

Both environments use:

```text
Version: f1.0.0v2-v1.10.1RC-latest-2518-g739b2e1d
Commit hash: 739b2e1d9ab63160eaff9c8f73172c8da68424e1
```

**Forge Neo has not been tested.** Other Forge variants are also unverified. The extension depends on Forge backend APIs and is not intended for A1111 (AUTOMATIC1111).

## Usage

### Choose the sampler and solver

1. Select `reForge-ODE Sampler` in the WebUI's Sampling method dropdown.
2. To use it in Hires.fix, also select it in the Hires.fix sampler dropdown, or use the WebUI option that inherits the txt2img sampler.
3. Enable the `reForge-ODE Sampler` accordion in the Script section to choose `txt2img Solver` and `hires.fix Solver`.
4. For a fixed-step solver, adjust the WebUI step count. For an adaptive solver, adjust the relative and absolute tolerances.

The sampler dropdown determines which sampler runs; the Script controls configure this sampler's ODE method. Selecting a solver in the Script controls does not switch the WebUI sampler.

With the Script accordion disabled, selecting `reForge-ODE Sampler` still runs this extension, using its Settings defaults. img2img uses the txt2img solver controls.

### Use it for one or both passes

| Combination | WebUI txt2img sampler | WebUI Hires.fix sampler |
|---|---|---|
| reForge-ODE for both | `reForge-ODE Sampler` | `reForge-ODE Sampler` or inherit txt2img |
| reForge-ODE for txt2img only | `reForge-ODE Sampler` | Another sampler |
| reForge-ODE for Hires.fix only | Another sampler | `reForge-ODE Sampler` |

TDE / RK can be selected as the other sampler if installed. Configure each extension's controls for the pass where it is used. This extension has no `→ TDE Sampler` or `→ RK Sampler` routing choices.

### What `Use same sampler` means here

| Script choice | Behavior |
|---|---|
| txt2img: `Use same sampler` | Uses this extension's Settings solver and tolerances |
| Hires.fix: `Use same sampler`, txt2img has an explicit solver | Inherits the txt2img solver, using the **Hires.fix tolerance sliders** for adaptive methods |
| Both: `Use same sampler` | Uses this extension's Settings solver and tolerances for both passes |

This choice does not delegate to another sampler. Settings → **reForge-ODE Sampler** is independent of reForge's built-in ODE Custom settings and of TDE / RK settings. To use another sampler for a pass, select it in the WebUI sampler dropdown.

## Adjusting computation and accuracy

### Fixed and adaptive steps

| Type | Main controls | Behavior |
|---|---|---|
| Fixed-step | WebUI step count and sigma schedule | Follows the supplied sigma grid |
| Adaptive-step | Relative / absolute tolerances | Uses the sigma endpoints and selects internal steps using error estimates |

Fixed-step solvers do not use rtol / atol. Max ODE Steps is an adaptive-step limit, not a replacement for the WebUI step count. Different solvers perform different numbers of model evaluations, so equal step counts do not imply equal generation times.

In the reForge sigma path used here, both passes read `p.scheduler`. A separate Hires.fix scheduler choice is not independently read by this extension's override.

### Tolerances and defaults

These controls set how much numerical error an adaptive solver allows:

- **Log Relative Tolerance**: an error allowance relative to the size of the values being calculated.
- **Log Absolute Tolerance**: an error allowance that also applies when those values are near zero.

Both sliders use the same scale:

| Slider value | Actual tolerance |
|---|---|
| −2 | 0.01 |
| −3 | 0.001 |
| −4 | 0.0001 |
| −5 | 0.00001 |

Moving from −3 to −4 makes the tolerance ten times smaller. Stricter tolerances usually take longer; they do not necessarily produce better-looking images. Start with the defaults: relative −3 (0.001), absolute −4 (0.0001).

For readers who enjoy the numerical details, the [TDE README](https://github.com/seti9585/sd-webui-tde-sampler#adjusting-computation-and-accuracy) and [RK README](https://github.com/seti9585/sd-webui-rk-sampler) explain tolerances and solver behavior in more detail. The controls and implementation details differ between extensions; use this README for reForge-ODE's settings.

The tolerance sliders become editable for explicitly selected adaptive solvers. They are greyed out for fixed-step solvers and `Use same sampler`. When Hires.fix inherits an explicit adaptive txt2img solver, its stored Hires.fix tolerance values still apply even though its sliders are greyed out; temporarily select the adaptive solver in Hires.fix to edit them.

Persistent defaults are under Settings → reForge-ODE Sampler:

| Setting | Default | Range |
|---|---|---|
| Default ODE solver | `dopri5` | The 11 solvers below |
| Default log relative tolerance | −3.0 | −7.0 to 0.0 |
| Default log absolute tolerance | −4.0 | −7.0 to 0.0 |
| Maximum ODE steps | 250 | 1 to 500 |

The enabled Script accordion's Advanced Settings → `Max ODE Steps` overrides the Settings limit for both passes. If an adaptive solver stops with `max_num_steps exceeded`, loosen the tolerances or increase this limit.

## Solver reference

Higher order describes numerical accuracy, not a guarantee of a better-looking image.

### Adaptive-step solvers

| Solver | Method |
|---|---|
| `dopri5` | Dormand–Prince, order 5 |
| `dopri8` | Dormand–Prince, order 8 |
| `bosh3` | Bogacki–Shampine, order 3 |
| `fehlberg2` | Fehlberg, order 2 |
| `adaptive_heun` | Adaptive Heun, order 2 |

### Fixed-step solvers

| Solver | Method |
|---|---|
| `euler` | Euler, order 1 |
| `midpoint` | Midpoint, order 2 |
| `rk4` | torchdiffeq RK4 (Kutta's 3/8-rule), order 4 |
| `heun3` | Heun, order 3 |
| `explicit_adams` | Explicit Adams–Bashforth multistep method |
| `implicit_adams` | Adams–Bashforth–Moulton predictor–corrector method |

`rk4` keeps reForge / torchdiffeq's naming. It corresponds to the 3/8-rule, called `kutta_38th4` in current TDE, rather than classical RK4 (`kutta4`). Matching method names alone does not guarantee matching images between extensions.

## Technical details and reproducibility

PNG infotext records the solver for each pass. Adaptive solvers also record that pass's tolerances and step cap:

```text
reForge-ODE solver
reForge-ODE log_rtol, reForge-ODE log_atol, reForge-ODE max_steps
reForge-ODE hires solver
reForge-ODE hires log_rtol, reForge-ODE hires log_atol, reForge-ODE hires max_steps
```

Fixed-step passes record only the solver among these extension-specific keys. The Script restores solver/tolerance fields from infotext; **Max ODE Steps must be matched manually**. For reproduction, also retain the WebUI commit, dependency versions, model, VAE, seed, prompts, scheduler, Hires.fix settings, and other extension/backend settings. Updating reForge can change the native function this extension calls.


---

# 日本語

[English](#sd-webui-reforge-ode)

reForge-ODE Sampler は、reForge 内蔵 ODE の生成を、独立した設定と txt2img／Hires.fix 別のソルバー選択で使うためのサンプラー拡張機能です。[torchdiffeq](https://github.com/rtqichen/torchdiffeq) を使い、11種類のソルバーを選べます。

## 独立したサンプラーにした理由

私の比較では、reForge 内蔵 ODE は [TDE Sampler](https://github.com/seti9585/sd-webui-tde-sampler) や [RK Sampler](https://github.com/seti9585/sd-webui-rk-sampler) と比べ、生成画像の雰囲気・画風に、単なる誤差と呼ぶには大きいと感じる違いがありました。その描写には、独立したサンプラー・拡張機能として残す価値があると判断しました。

この拡張機能は、その描写を使い続けられるようにし、設定を独立させて普段使いやすくすることを目的としています。画像差の原因は現段階では確定しておらず、特定の数値的要因や、他のサンプラーに対する画質の優劣を主張するものではありません。

reForge では、利用可能な場合は内蔵の `sample_ode` を呼び、sigma 生成には `AlterSampler.get_sigmas`、生成処理にはホストの `KDiffusionSampler` の経路を使います。reForge の ODE 実装に基づくフォールバックも同梱しています。出力はホストのバージョンや生成環境にも依存するため、異なるバージョンや WebUI 間での画像一致は保証しません。

## 導入と動作確認環境

Extensions → Install from URL に次のURLを入力します。

```text
https://github.com/seti9585/sd-webui-reForge-ODE
```

導入後は WebUI を再起動してください。`torchdiffeq` が未導入の場合は `install.py` がインストールします。手動導入が必要な場合は、WebUI が使う Python 環境で実行します。

```bash
pip install torchdiffeq
```

作者による動作確認は **reForge のみ**で、次の2環境で行っています。

| WebUI | Python | PyTorch |
|---|---|---|
| reForge | 3.10.11 | 2.9.0+cu128 |
| reForge | 3.12.10 | 2.11.0+cu128 |

どちらも次の reForge バージョン・コミットです。

```text
Version: f1.0.0v2-v1.10.1RC-latest-2518-g739b2e1d
Commit hash: 739b2e1d9ab63160eaff9c8f73172c8da68424e1
```

**Forge Neo は未確認です。** その他の Forge 派生版も未確認です。Forge バックエンドの API に依存しており、A1111（AUTOMATIC1111）向けではありません。

## 使い方

### サンプラーとソルバーを選ぶ

1. WebUI の Sampling method で `reForge-ODE Sampler` を選びます。
2. Hires.fix にも使う場合は、Hires.fix のサンプラーも同じものにするか、WebUI 側で txt2img のサンプラーを継承する設定にします。
3. Script 欄の `reForge-ODE Sampler` アコーディオンを有効にし、`txt2img Solver` と `hires.fix Solver` を選びます。
4. 固定ステップ法では WebUI のステップ数、可変ステップ法では相対・絶対許容誤差を調整して生成します。

WebUI のサンプラー欄は使うサンプラーを、Script 欄はこのサンプラー内の解法を選ぶものです。Script 欄でソルバーを選ぶだけでは、WebUI のサンプラー選択は切り替わりません。

Script アコーディオンを無効にしていても、サンプラー欄で `reForge-ODE Sampler` を選べば、この拡張機能の Settings の既定値で動作します。img2img は txt2img 側のソルバー設定を使います。

### txt2img と Hires.fix を使い分ける

| 組み合わせ | WebUI の txt2img サンプラー | WebUI の Hires.fix サンプラー |
|---|---|---|
| 両方 reForge-ODE | `reForge-ODE Sampler` | `reForge-ODE Sampler` または txt2img を継承 |
| txt2img だけ reForge-ODE | `reForge-ODE Sampler` | 他のサンプラー |
| Hires.fix だけ reForge-ODE | 他のサンプラー | `reForge-ODE Sampler` |

TDE／RK を導入していれば、片方のサンプラーとして選べます。使う段階に合わせて各拡張機能の設定を行ってください。この拡張機能には `→ TDE Sampler` や `→ RK Sampler` の委譲用選択肢はありません。

### この拡張機能の `Use same sampler`

| Script 欄の選択 | 動作 |
|---|---|
| txt2img が `Use same sampler` | この拡張機能の Settings のソルバーと許容誤差を使う |
| Hires.fix が `Use same sampler`、txt2img は具体的なソルバー | txt2img のソルバーを継承し、可変ステップ法の許容誤差には **Hires.fix 側のスライダー値**を使う |
| 両方 `Use same sampler` | 両段階とも、この拡張機能の Settings のソルバーと許容誤差を使う |

この選択肢は他のサンプラーへの委譲ではありません。Settings → **reForge-ODE Sampler** は、reForge 内蔵 ODE Custom や TDE／RK の設定から独立しています。他のサンプラーを使う段階は、WebUI のサンプラー欄で選んでください。

## 計算量と精度の調整

### 固定ステップ法と可変ステップ法

| 種類 | 主に調整する項目 | 動作 |
|---|---|---|
| 固定ステップ法 | WebUI のステップ数と sigma スケジュール | 渡された sigma 列に沿って計算する |
| 可変ステップ法 | 相対・絶対許容誤差 | sigma の両端を使い、推定誤差に応じて内部のステップを選ぶ |

固定ステップ法は rtol／atol を使いません。Max ODE Steps は可変ステップ法の上限で、WebUI のステップ数の代わりではありません。解法ごとにモデル評価回数が異なるため、同じステップ数でも生成時間は変わります。

本拡張機能の reForge 用 sigma 経路は、両段階で `p.scheduler` を参照します。独立した Hires.fix 用スケジューラの選択を、この拡張機能のオーバーライドが別途読む実装にはなっていません。

### 許容誤差と既定値

可変ステップ法で、計算上の誤差をどの程度まで許すかを決める設定です。

- **Log Relative Tolerance（相対許容誤差）**：計算中の値の大きさに応じて、許す誤差を決めます。
- **Log Absolute Tolerance（絶対許容誤差）**：計算中の値がゼロに近い場合にも使う、誤差の基準です。

どちらのスライダーも、表示値と実際の許容誤差は次の対応です。

| スライダーの表示値 | 実際の許容誤差 |
|---|---|
| −2 | 0.01 |
| −3 | 0.001 |
| −4 | 0.0001 |
| −5 | 0.00001 |

−3 から −4 にすると、許す誤差は10分の1になります。厳しくするほど通常は生成に時間がかかりますが、画像の見栄えが良くなるとは限りません。まずは既定値の相対 −3（0.001）、絶対 −4（0.0001）から使ってください。

数値計算の話をもっと詳しく知りたい方は、[TDE の README](https://github.com/seti9585/sd-webui-tde-sampler#計算量と精度の調整) や [RK の README](https://github.com/seti9585/sd-webui-rk-sampler#日本語) をどうぞ。許容誤差や解法について、そちらで詳しく説明しています。拡張機能ごとに設定や実装は異なるので、reForge-ODE の操作はこの README を参照してください。

許容誤差スライダーは、可変ステップ法を具体的に選んだ場合に編集できます。固定ステップ法と `Use same sampler` ではグレーアウトします。ただし Hires.fix が txt2img の具体的な可変ステップ法を継承する場合は、グレーアウト中でも Hires.fix 側に保持された値を使います。編集するときは、Hires.fix 側でも一時的にその可変ステップ法を選んでください。

Settings → reForge-ODE Sampler の既定値：

| 項目 | 既定値 | 範囲 |
|---|---|---|
| Default ODE solver | `dopri5` | 下記11種類 |
| Default log relative tolerance | −3.0 | −7.0 ～ 0.0 |
| Default log absolute tolerance | −4.0 | −7.0 ～ 0.0 |
| Maximum ODE steps | 250 | 1 ～ 500 |

Script アコーディオンを有効にすると、Advanced Settings → `Max ODE Steps` が両段階共通の上限として Settings の値より優先されます。可変ステップ法が `max_num_steps exceeded` で停止するときは、許容誤差を緩めるか上限を増やしてください。

## 解法一覧

次数は数値計算の精度に関する性質であり、高次の解法が必ず見栄えの良い画像を作るという意味ではありません。

### 可変ステップ法

| ソルバー | 解法 |
|---|---|
| `dopri5` | Dormand–Prince、5次 |
| `dopri8` | Dormand–Prince、8次 |
| `bosh3` | Bogacki–Shampine、3次 |
| `fehlberg2` | Fehlberg、2次 |
| `adaptive_heun` | Adaptive Heun、2次 |

### 固定ステップ法

| ソルバー | 解法 |
|---|---|
| `euler` | Euler、1次 |
| `midpoint` | Midpoint、2次 |
| `rk4` | torchdiffeq の RK4（Kutta の3/8則）、4次 |
| `heun3` | Heun、3次 |
| `explicit_adams` | 陽的 Adams–Bashforth 多段法 |
| `implicit_adams` | Adams–Bashforth–Moulton 予測子・修正子法 |

`rk4` は reForge／torchdiffeq の名称を維持しています。現在の TDE の `kutta_38th4` に相当する3/8則で、古典的 RK4（`kutta4`）とは異なります。解法名を合わせるだけで、拡張機能間の画像が一致するとは限りません。

## 技術的な詳細と再現性

PNG の infotext には各段階のソルバーを記録します。可変ステップ法では、その段階の許容誤差とステップ数上限も記録します。

```text
reForge-ODE solver
reForge-ODE log_rtol, reForge-ODE log_atol, reForge-ODE max_steps
reForge-ODE hires solver
reForge-ODE hires log_rtol, reForge-ODE hires log_atol, reForge-ODE hires max_steps
```

固定ステップ法で記録する拡張機能固有の項目はソルバーのみです。Script 欄は infotext からソルバー・許容誤差を復元しますが、**Max ODE Steps は手動で合わせてください**。再現には WebUI のコミット、依存ライブラリ、モデル、VAE、seed、prompt、スケジューラ、Hires.fix、他の拡張機能やバックエンドの設定も記録してください。reForge を更新すると、呼び出す内蔵関数が変わる可能性があります。

## Acknowledgements and license

Thank you to **redhottensors**, author of [ComfyUI-ODE](https://github.com/redhottensors/ComfyUI-ODE), for the original ODE implementation, and to **Panchovix**, author of [reForge](https://github.com/Panchovix/stable-diffusion-webui-reForge), for porting and integrating it into reForge. This extension builds on their work. Thanks also to the [torchdiffeq](https://github.com/rtqichen/torchdiffeq) developers.

This extension is distributed under **GNU AGPL v3**, following the reForge code included in its fallback. ComfyUI-ODE's upstream code is under **GNU GPL v3**. Both license texts are included in [LICENSE](LICENSE); see [NOTICE.md](NOTICE.md) for attribution.

## 謝辞とライセンス

ODE の元実装である [ComfyUI-ODE](https://github.com/redhottensors/ComfyUI-ODE) の作者 **redhottensors 氏**と、その実装を [reForge](https://github.com/Panchovix/stable-diffusion-webui-reForge) に移植・統合された作者 **Panchovix 氏**に感謝します。この拡張機能は、お二人の成果を基にしています。また、[torchdiffeq](https://github.com/rtqichen/torchdiffeq) の開発者の皆様にも感謝します。

同梱フォールバックの移植元である reForge に従い、本拡張機能は **GNU AGPL v3** で配布します。元となる ComfyUI-ODE のコードは **GNU GPL v3** です。両ライセンスの本文は [LICENSE](LICENSE) にまとめています。出典は [NOTICE.md](NOTICE.md) を参照してください。
