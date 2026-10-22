# 画像検査とROS 2による自律仕分け

提出用リポジトリ：[rrty-olk/P_AI_Submit](https://github.com/rrty-olk/P_AI_Submit)。
[説明資料（PDF）](output/submission/report.pdf) ／
[説明資料（Markdown・Mermaid構成図）](output/submission/report.md) ／
[実動作動画（MP4・約92秒）](output/submission/inspection_demo.mp4)。

LiquidAI/d1-omni-600Mの公式`system_one`で部品画像を判定し、ROS 2 Jazzyの
turtlesimをPASS / REJECT / RECHECKの座標へ移動させます。アームはなく、
2次元移動ロボットによる仮想仕分けです。外部LLM API・3Bモデルは使用しません。

## 構成と安全制御

画像 → CUDA上の公式モデル → 確率分布の検証 → 目標選択 → 姿勢フィードバック → 到達確認。
AIとROSはComposeの別サービス、同じUbuntu 24.04イメージを使用します。
推論時は停止。最大確率が0.60未満なら一時停止してRECHECKへ移動します。
推論30秒・移動30秒・姿勢更新0.5秒で異常停止し、その実行を中止します。
独立したwatchdogが制御ノードの指令更新停止から0.5秒で速度をゼロにします。
watchdog自体の故障・コンテナ停止時にはturtlesim側の速度指令タイムアウトも働きますが、
実機の機能安全を保証する構成ではありません。

|判定|目標座標|
|---|---|
|PASS|(8.5, 8.5)|
|REJECT|(2.0, 8.5)|
|RECHECK|(5.5, 2.0)|

到達許容誤差は0.12、速度上限1.7、角速度上限2.5です。
合成画像9枚の正解ラベルは評価専用であり、推論入力や経路選択には使いません。
モデルのsoftmax値は工場検査の校正済み欠陥確率ではありません。

## 必要環境

- Windows 11、WSL 2、NVIDIAドライバー、Docker EngineとCompose、NVIDIA Container Toolkit。
- NVIDIA GPU（今回の実測対象はRTX 3070 8GB）、十分な空きディスク（目安25GB以上）。
- WSL内で`docker run --rm --gpus all nvidia/cuda:12.0.0-runtime-ubuntu20.04 nvidia-smi`が成功すること。
  この古いCUDAイメージは既存環境の認識確認だけに使用し、本体はPyTorchのCUDA 12.8です。
- ホストの既存WSLはUbuntu 22.04。ROSと推論はUbuntu 24.04コンテナで実行します。
  Ubuntu 24.04のWSLホスト上での動作は未検証です。

## 初期セットアップ：Windows 11

以下は新規環境向けの手順です。既存環境での実測はUbuntu 22.04のWSL 2を使用しています。
Ubuntu 24.04の新規セットアップと、Ubuntu単独起動での一連の動作は未検証です。
新規導入ではUbuntu 24.04を使用し、既存のUbuntu 22.04を使う場合は配布名を読み替えます。
CPUはx86_64、GPUはNVIDIAを前提とします。

### 1．WSL 2とUbuntuを導入する（管理者PowerShell）

```powershell
wsl --list --online
wsl --install -d Ubuntu-24.04
```

Windowsの再起動を求められたら再起動します。再起動後にPowerShellで実行し、
Ubuntuの初回起動でLinuxユーザー名・パスワードを設定します。
すでに対象Ubuntuがある場合は再インストールせず、更新・確認から進めます。

```powershell
wsl --update
wsl --set-default-version 2
wsl --set-version Ubuntu-24.04 2
wsl --list --verbose
wsl -d Ubuntu-24.04
```

`wsl --list --verbose`で対象UbuntuのVERSIONが`2`になっていることを確認します。
手順の出典：[Microsoft WSL導入手順](https://learn.microsoft.com/en-us/windows/wsl/install)。

### 2．Windows用NVIDIAドライバーを導入する

PowerShellから公式ダウンロードページを開きます。

```powershell
Start-Process 'https://www.nvidia.com/en-us/drivers/'
```

GPU型番とWindows 11に対応するドライバーを選んでダウンロードし、インストーラーを実行します。
GPUごとに配布ファイルが異なるため、この工程は画面で選択します。必要に応じて再起動した後、確認します。

```powershell
nvidia-smi
wsl -d Ubuntu-24.04 -- /usr/lib/wsl/lib/nvidia-smi
```

このWSL構成では**Ubuntu内にLinux用GPUドライバーを入れません**。
WindowsドライバーがWSLへGPUを提供します。
本手順はUbuntu内のDocker Engineを使用します。Docker Desktopを使っている場合、
同じUbuntuへのWSL連携を無効にして、使用するDockerデーモンを混在させないでください。
出典：[NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。

## 初期セットアップ：Ubuntu

以降の`bash`コマンドはUbuntuのターミナルで実行します。
WSLの場合は`wsl -d Ubuntu-24.04`で起動します。

### 1．基本パッケージとsystemd

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg python3 unzip git
uname -m
ps -p 1 -o comm=
```

`uname -m`は`x86_64`、PID 1は`systemd`を想定します。
WSLでPID 1が`systemd`ではない場合だけ、次のコマンドで既存設定をバックアップして有効化します。

```bash
sudo apt-get install -y systemd systemd-sysv
sudo python3 - <<'PY'
import configparser
import shutil
import time
from pathlib import Path

path = Path('/etc/wsl.conf')
config = configparser.ConfigParser(interpolation=None)
if path.exists():
    shutil.copy2(path, path.with_name('wsl.conf.backup-' + str(time.time_ns())))
    config.read(path)
if not config.has_section('boot'):
    config.add_section('boot')
config.set('boot', 'systemd', 'true')
with path.open('w') as stream:
    config.write(stream)
PY
```

設定変更後は作業を保存し、**WindowsのPowerShell**でWSLを終了・再起動します。
`wsl --shutdown`は他のWSLディストリビューションも終了します。

```powershell
wsl --shutdown
wsl -d Ubuntu-24.04
```

Ubuntuへ戻り、`ps -p 1 -o comm=`で`systemd`を確認します。
出典：[Microsoft WSLのsystemd設定](https://learn.microsoft.com/en-us/windows/wsl/systemd)。

UbuntuをWindowsと別に単独起動している場合はWSL設定を省き、Linux用GPUドライバーを導入します。
以下は**WSLでは実行しない**コマンドです。

```bash
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers install
sudo reboot
# 再起動後
nvidia-smi
```

出典：[UbuntuのNVIDIAドライバー導入](https://ubuntu.com/server/docs/how-to/graphics/install-nvidia-drivers/)。

### 2．Docker EngineとCompose

以下はDocker未導入のUbuntu向けです。すでにDockerが正常に動作する場合は確認コマンドへ進みます。
`docker.io`や`podman-docker`など別方式のパッケージがある場合は、
[Docker公式の競合パッケージ手順](https://docs.docker.com/engine/install/ubuntu/#uninstall-old-versions)
に従って整理してから導入します。

```bash
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources > /dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo docker run --rm hello-world
sudo usermod -aG docker "$USER"
exit
```

`exit`後にUbuntuを開き直してグループ変更を反映し、sudoなしで確認します。
`docker`グループには管理者相当の権限があります。

```bash
docker version
docker compose version
docker run --rm hello-world
```

出典：[Docker Ubuntu導入](https://docs.docker.com/engine/install/ubuntu/)、
[Dockerグループ設定](https://docs.docker.com/engine/install/linux-postinstall/)。

### 3．NVIDIA Container ToolkitとGPUコンテナ確認

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
nvidia-ctk --version
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

GPU名が表示されることを確認します。Dockerの再起動は稼働中のコンテナへ影響します。
出典：[NVIDIA Container Toolkit公式導入手順](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。

### 4．配布コードの配置とアプリケーション導入

Gitが導入済みなら、ZIP展開の代わりに次のコマンドで取得できます。取得先が存在しない状態で実行します。

Windows PowerShell：

```powershell
git clone https://github.com/rrty-olk/P_AI_Submit.git C:\PhysicalAI
```

Ubuntu単独起動：

```bash
git clone https://github.com/rrty-olk/P_AI_Submit.git ~/PhysicalAI
cd ~/PhysicalAI
```

ZIPを使う場合は以下の手順です。Git cloneとZIP展開はどちらか一方を使用します。

Windowsでは配布ZIPを`C:\PhysicalAI`へ展開します。例ではZIPをダウンロードフォルダーに保存したものとします。

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\physicalai_source_and_evidence.zip" -DestinationPath 'C:\PhysicalAI'
```

WSL側では対応する場所へ移動します。

```bash
cd /mnt/c/PhysicalAI
test -f .env
docker compose config --quiet
docker compose build
docker compose run --rm --no-deps ai python -c "import torch; assert torch.cuda.is_available(); x=torch.ones((2,2),device='cuda'); y=x@x; torch.cuda.synchronize(); assert y.sum().item()==8; print(torch.cuda.get_device_name(0),torch.__version__,torch.version.cuda)"
```

Ubuntu単独起動の場合は、ZIPのあるディレクトリで次を実行し、同じビルド・GPU確認コマンドへ進みます。

```bash
mkdir -p ~/PhysicalAI
unzip physicalai_source_and_evidence.zip -d ~/PhysicalAI
cd ~/PhysicalAI
```

ROS 2 Jazzy、turtlesim、PyTorch、Transformers、OpenCV、FFmpeg、日本語PDF生成環境は
Dockerfileがコンテナ内へ導入します。ホストにROS 2やCUDA Toolkitを個別に入れる必要はありません。
固定済みの`.env`は保持します。続く再現手順で、モデル重みの取得から成果物生成まで実行します。

## 再現手順

WSL内でこのディレクトリへ移動します。初回は依存パッケージと約2.4GBの公式重みを取得するため時間がかかります。

```bash
bash scripts/inspect_environment.sh
test -f .env || python3 scripts/fetch_official.py
bash scripts/run.sh
```

Windows PowerShellからは、配布コードのディレクトリで実行します。

```powershell
Set-Location 'C:\PhysicalAI'
.\scripts\run.ps1 -Distribution Ubuntu-24.04
```

既存の検証環境では`.\scripts\run.ps1 -Distribution Ubuntu-22.04`を使用します。
引数を省略した場合の既定値も`Ubuntu-22.04`です。Ubuntu単独起動では`bash scripts/run.sh`を使用します。
`fetch_official.py`を再実行すると、その時点の最新版に固定し直されます。
同じモデルを再現する場合、既存の`.env`を保持してください。
配布コードには固定済みの`.env`と公式モデルの参照コードを含めています。
`run.sh`はビルド、入力生成、単体テスト、安全処理の統合テスト、GPU推論とROS移動、
ログ監査、日本語動画・レポート生成、ZIP作成と整合性確認まで実行します。
終了時は成功・失敗にかかわらずComposeを停止し、モデルキャッシュと出力を保持します。
画像生成時だけ`data`を書き込み可能にし、推論・移動時は読み取り専用にします。
構築ログは`output/logs/build.txt`、実行ログは`output/logs/run_console.txt`です。

個別実行:

```bash
mkdir -p output/logs
docker compose build
docker compose run --rm --no-deps tools python scripts/generate_cases.py
docker compose run --rm --no-deps robot python -m pytest tests/test_core.py -q --junitxml=output/logs/unit_tests.xml | tee output/logs/unit_tests.txt
docker compose up -d --force-recreate ai
docker compose run --rm robot bash scripts/safety.sh
docker compose up --abort-on-container-exit --exit-code-from robot robot
docker compose run --rm --no-deps robot python -m inspection.audit
docker compose run --rm --no-deps robot python -m inspection.render
docker compose run --rm --no-deps robot python -m inspection.report
docker compose run --rm --no-deps robot pip freeze --local > requirements.lock.txt
docker image inspect physicalai-inspection:local --format '{{json .}}' > output/logs/image_provenance.json
docker compose run --rm --no-deps tools python scripts/package.py
docker compose run --rm --no-deps robot python scripts/check_deliverables.py
docker compose down
```

## 成果物と証跡

- `output/submission/inspection_demo.mp4`: 実際のturtlesim画面記録、実測姿勢・軌跡、入力・判定・推論時間。
- `output/submission/report.pdf`: 実測値から生成する説明資料。
- `output/submission/report.md`: PDFと同じ実測値・本文を使うMarkdownレポート。システム構成図をMermaidコードブロックで収録。
- `docs/system_architecture.mmd`: 構成図のMermaid原稿。PDFには`docs/images/system_architecture.png`を掲載し、PNG/SVGも同梱。
- `output/runs/<run>/`: events/poses JSONL、summary、ROSグラフ、実画面の未編集記録、監査結果。
- `output/logs/`: 環境、構築、モデル情報、推論、単体・安全統合テスト。
- `docs/official_model/`: 実行した公式実装の参照コードとLICENSE。重みはDocker名前付きボリューム内。
- `output/latest_run.txt`: 直近実行のディレクトリ。失敗実行でも記録します。

動画・PDF・Markdown・Mermaid構成図の説明は日本語です。モデル名・API名・ROSトピックなどの識別子は保持しています。
`docs/fonts/`に再配布可能な日本語フォントM PLUS 1pとOFLライセンスを同梱しています。
動画とレポートは同一実行の実測ログから生成します。モデルの質問文は英語です。

動画は実測時刻に沿った1倍速（長い場合は明記した2倍速）の再生です。
記録時刻は開始時の実時刻と単調時計の経過時間から算出し、WSLの時刻補正による逆行を防ぎます。
実画面の各フレームにも同じ単調時計の時刻を記録し、画面と姿勢を同期します。
冒頭・末尾の静止説明パネルを含めて60〜180秒に調整します。
ROSデータ・公式GPU推論証跡・全ケース到達・複数経路がない場合は完成動画を生成しません。
数値制御の単体テスト内の運動計算は、ROS動作や動画の証拠としては使いません。

## 制約と失敗時の確認

- CUDA未認識: CPUには切り替えずエラーにします。`output/logs/environment.txt`とComposeログを確認。
- モデル起動失敗: `docker compose logs ai`。`trust_remote_code=True`はコミット固定の公式コードに使用。
- 画像を変えた場合はsha256も記録されます。動画の再生成時に元画像と一致する必要があります。
- 推論誤判定や全ケース同一振り分けは成功扱いせず、結果を残します。
- 固定の人工画像のみの小規模デモです。実写の検査精度、照明変化、未知の欠陥に対する有効性は未検証です。
- `requirements.lock.txt`は実行環境の記録です。Dockerfileはバージョン範囲で依存を取得するため、将来のビルドで全パッケージが同一になる保証はありません。
- `docker compose down`はモデルボリュームを残します。`down -v`は重みキャッシュも消します。

## 提出

締切: 2026年11月2日10:00 JST。課題資料の出典は`docs/sources.md`。
PDFのファイル名を`説明資料_PAI最終課題_<OmniCampusアカウント名またはID>.pdf`に変更してください。
動画タイトルは`PAI最終課題_<アカウント名またはID>`。
公開物には氏名、所属、メールを含めません。生の環境ログにはローカルユーザー名などが
含まれるため、配布ZIPから除外し、匿名化した検証結果を含めます。

元モデルはLFM Open License v1.0です。公式モデルのLICENSEを参照してください。
