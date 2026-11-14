# 第 2 章 环境准备与第一次跑通

## 本章目标

- 能按官方版本配套表选定一组驱动、固件、CANN（Compute Architecture for Neural Networks，昇腾的异构计算软件栈）、PyTorch、torch_npu（Ascend Extension for PyTorch，PyTorch 的昇腾适配插件）版本，并写出完整安装命令。
- 能运行 `check_env.sh` 与 `check_env.py`，读出设备数、卡型号、显存容量、CANN 版本、torch 与 torch_npu 版本。
- 能写出带设备节点与驱动库挂载的 `docker run` 命令，并逐项说明参数作用。

## 先说结论

- Atlas 800T A2 训练服务器、Atlas 200I DK A2 开发者套件与华为云 AI 加速型 ECS（Elastic Cloud Server，弹性云服务器）是三条取环境的路径，三者的 Host 侧 CPU 架构、可见卡数与预装软件不同。
- 安装顺序为驱动与固件 → CANN Toolkit 与算子包 → 环境变量 → Python / PyTorch / torch_npu。驱动让 `npu-smi info` 看到卡，CANN Toolkit 提供 `libascendcl.so`，torch_npu 依赖驱动与 CANN Toolkit 才能注册 `npu` 后端。
- 首次安装按「驱动 → 固件」顺序，覆盖安装按「固件 → 驱动」顺序，出自官方《安装NPU驱动和固件》。
- CANN 9.0.0 版本说明列出的配套 Ascend HDK（官方版本表中对驱动与固件包的称呼）版本为 25.7.RC1、26.0.RC1、25.5.X、25.3.X、25.2.X、25.0.X；CANN、PyTorch、torch_npu 三者的组合以昇腾社区版本配套查询助手为准。
- `docker run` 需要挂载 `/dev/davinci0`、`/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc` 四个设备节点以及宿主机驱动库目录，容器内只装 CANN 软件。

## 安装链路与版本配套

### 三条取环境的路径

CANN 必须落在真实的昇腾 AI 处理器（NPU，Neural Processing Unit，专门执行矩阵与向量计算的处理器）上才能执行计算。取环境有三条路径。

| 路径 | 具体形态 | 适用场合 | 需要自己安装的部分 |
|---|---|---|---|
| 本地 Atlas 服务器 | Atlas 800T A2 训练服务器、Atlas 800I A2 推理服务器等整机；或在通用服务器里插 Atlas 300T 训练卡、Atlas 300I 推理卡 | 有多卡、要做训练、要做性能观测与多机扩展 | 驱动、固件、CANN Toolkit、算子包、Python、PyTorch、torch_npu 全套 |
| 开发者套件 | Atlas 200I DK A2 开发者套件（Ascend 310B 芯片）、Atlas 500 A2 智能小站 | 单板入门、算子调试、边缘推理验证 | 官方《环境准备》要求先完成驱动、固件、`Ascend-cann-toolkit` 的安装，再设置 `set_env.sh` |
| 云上昇腾实例 | 华为云 AI 加速型 ECS 的 Ai1 / Ai2 规格（基于 Ascend 310 系列处理器与 Atlas 300I 板卡）；GitCode 云开发环境（在线 Ascend ARM 环境，目前仅适用于 Atlas A2 系列产品，提供 WebIDE 与 VSCode IDE 两种接入方式） | 本地无设备、按需取用、短期验证 | 公共镜像已预装开发套件包与环境变量，只需校验；私有镜像按本章顺序自行安装 |

华为云《AI-accelerated ECSs》规格页写明其公共镜像默认已包含 CANN 3.1.0 开发套件包并配好环境变量（官方公开规格：该页描述的 Ai1 / Ai2 规格，镜像版本随控制台实际镜像变化）。[GitCode 上 `cann/asc-devkit` 的快速开始文档](https://gitcode.com/cann/asc-devkit/blob/master/docs/quick_start.md)说明云开发环境默认装好商用版 NPU 驱动、固件与 CANN 包，源码下载时要与已装软件配套。

调用链位置：三条路径都提供 Host 与 Device 两侧，差别只在驱动、固件与 CANN 由谁安装。由谁决定：硬件型号决定可装的驱动与 kernels 包名。输入 → 输出：一台物理机或一个云实例 → 可被 `npu-smi info` 识别的 Device。

### 安装顺序与每层职责

下表按依赖顺序组织。表中 `<ver>`、`<chip>`、`<arch>` 是占位符，取值来自下一节的官方配套表，`<arch>` 为 `aarch64` 或 `x86_64`，`<chip>` 决定 kernels 包名中的芯片型号，例如 `910b`。

| 顺序 | 装什么 | 关键命令或路径 | 由谁决定 | 输入 → 输出 |
|---|---|---|---|---|
| 1 | 驱动与固件 | `Ascend-hdk-<chip>-npu-driver_<ver>_linux-<arch>.run --full --install-for-all`；`Ascend-hdk-<chip>-npu-firmware_<ver>.run --full` | 硬件型号 + 官方配套表 | 安装包 → 内核模块加载，`npu-smi info` 可见卡 |
| 2 | CANN Toolkit 与算子包 | `sh Ascend-cann-toolkit_<ver>_linux-<arch>.run --install`；`sh Ascend-cann-kernels-<chip>_<ver>_linux-<arch>.run --install` | CANN 版本 + 芯片型号 | 安装包 → 头文件、动态库与算子二进制落进 `/usr/local/Ascend` |
| 3 | 环境变量 | `source <install_path>/set_env.sh`，并写入 `~/.bashrc` | 安装用户与安装路径 | 无 → `PATH`、`LD_LIBRARY_PATH`、`PYTHONPATH`、`ASCEND_HOME_PATH` |
| 4 | Python / PyTorch / torch_npu | `pip3 install torch==<ver>`；`pip3 install torch-npu==<ver>` | 版本配套表 | whl 包 → `import torch_npu` 注册 `npu` 后端 |

Toolkit 与算子包的顺序由官方安装文档固定：先装 Toolkit，再装 kernels 算子包；官方说明 kernels 包提供单算子 API 执行（aclnn 类 API）所需的动态库与 kernel 二进制。驱动与固件的顺序按场景区分，官方《安装NPU驱动和固件》给出的规则是首次安装「驱动 → 固件」、覆盖安装「固件 → 驱动」，驱动与固件安装完成后用 `npu-smi info` 确认驱动加载成功。

后端对照表：

| 后端概念 | 昇腾侧对应物 | 差异 |
|---|---|---|
| 装机时先装网卡、RAID 等内核驱动 | 先装 Ascend HDK 的 NPU 驱动与固件 | 驱动包带固件，两者有各自版本，且要互相兼容 |
| 配 `JAVA_HOME` / `CLASSPATH` | `source <install_path>/set_env.sh` | 后端通常手工 export 一两个变量；CANN 侧由脚本一次注入多个路径变量 |
| 引入 SDK 与依赖包 | 安装 CANN Toolkit 与 `Ascend-cann-kernels-<chip>_<ver>_linux-<arch>.run` | kernels 包按芯片型号区分，装错型号时算子查找会失败 |
| 应用侧引入客户端库 | `pip3 install torch-npu==<ver>` 后 `import torch_npu` | 该库注册的是 PyTorch 的 `npu` 后端，业务代码仍写 `torch.*` |
| 跑通 Hello World 并看日志 | `npu-smi info` + `torch.npu.is_available()` + 一次 `torch.add` | 除了进程日志，还要看设备侧 `Health`、`HBM-Usage(MB)` 与 `AICore(%)` |
| 容器里挂载 `--device` 与宿主机目录 | `--device=/dev/davinci0` 等一组设备节点与 `/usr/local/Ascend/driver` 目录 | 容器内不安装驱动，驱动库由宿主机挂进去 |

### 版本配套关系

驱动、固件、CANN、PyTorch、torch_npu 之间存在三方配套关系：Ascend HDK（驱动与固件）与 CANN 之间一张表，CANN 与 torch_npu 之间一张表，torch_npu 与 PyTorch 之间一张表。下表只列已核实的行，未列出的组合一律以版本配套查询助手为准。

| CANN 版本 | 配套 Ascend HDK / Driving SDK | 配套 PyTorch | torch_npu 安装包 | 出处 |
|---|---|---|---|---|
| 9.0.0 | Ascend HDK 25.7.RC1、26.0.RC1、25.5.X、25.3.X、25.2.X、25.0.X | 2.7.1 | 2.7.1.post4（TorchNPU 26.0.0，对应 CANN 9.0.0） | CANN 9.0.0 版本说明、TorchNPU 26.0.0 Release Notes |
| 8.5.0 | Driving SDK 7.3.0 声明兼容 CANN 8.1.RC1、8.2.RC1、8.3.RC1、8.5.0 | 以版本配套查询助手为准 | 以版本配套查询助手为准 | TorchNPU 26.0.0 Release Notes |
| 8.2.RC1 | 以版本配套查询助手为准 | 2.1.0 | 2.1.0.post13 | Gitee `Ascend/pytorch` 昇腾辅助软件表 |
| 8.0.RC3 | 以版本配套查询助手为准 | 以版本配套查询助手为准 | 以版本配套查询助手为准 | Ascend 开源《快速安装昇腾环境》，示例为 `910b` / `aarch64` |

以上数字均为官方公开规格，出处随行给出。查询入口：

- 昇腾社区[版本配套查询助手](https://www.hiascend.com/developer/download/compatibility)：呈现昇腾全栈软件与核心三方软件的版本配套关系，可按 CANN、Ascend HDK、PyTorch 等产品筛选。
- [CANN 9.0.0 版本说明](https://www.hiascend.com/document/detail/zh/canncommercial/900/releasenote/release-notes.md)：`CANN与驱动版本配套关系` 表与 `CANN组合配套关系` 表。
- [CANN 社区版版本说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920/softwareinst/releasenote)：`CANN软件版本配套表` 与版本兼容性说明。
- Gitee `Ascend/pytorch` 的「PyTorch与Python版本配套表」与「昇腾辅助软件」表。

调用链位置：配套关系落在驱动层、CANN 层与框架适配层三者之间。由谁决定：软件发布方给出配套表，安装者按表选定一组版本。输入 → 输出：硬件型号与目标框架版本 → 一组确定的安装包版本号。

### 容器方式

昇腾镜像仓库地址为 `swr.cn-south-1.myhuaweicloud.com/ascendhub/`（出自 [GitCode `cann/asc-devkit` 快速开始文档](https://gitcode.com/cann/asc-devkit/blob/master/docs/quick_start.md)）。该文档给出的一个镜像 tag 示例是 `cann:9.0.0-beta.2-910b-ubuntu22.04-py3.11`（官方公开规格：该文档面向 Atlas A2 系列产品）；tag 随 CANN 版本、芯片型号、操作系统与 Python 版本变化，实际取值以镜像仓库中的列表为准。

容器内不安装驱动与固件，宿主机上装好即可。官方《宿主机目录挂载至容器》给出的启动命令如下，`docker_image_id` 用 `docker images` 查到的镜像 ID 替换：

```bash
docker run -it \
  --ipc=host \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/common \
  -v /usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/vnpu.cfg:/etc/vnpu.cfg \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  docker_image_id /bin/bash
```

| 参数 | 作用 |
|---|---|
| `--device=/dev/davinci0` | 把宿主机 0 号 NPU 设备节点映射进容器，`X` 为设备 ID；多卡时逐项添加 `--device=/dev/davinci1` 等 |
| `--device=/dev/davinci_manager` | 设备管理节点，供容器内进程做设备发现与资源管理 |
| `--device=/dev/devmm_svm` | 共享虚拟内存节点，支撑 Host 与 Device 之间的内存映射 |
| `--device=/dev/hisi_hdc` | Host 与 Device 之间的通信通道节点 |
| `-v /usr/local/dcmi:/usr/local/dcmi` | 挂载 DCMI（昇腾设备管理接口）目录 |
| `-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi` | 把宿主机的 `npu-smi` 工具挂进容器，容器内可执行 `npu-smi info` |
| `-v /usr/local/Ascend/driver/lib64/...` | 挂载宿主机驱动动态库，容器内 CANN 运行时链接这些库 |
| `-v /etc/ascend_install.info:/etc/ascend_install.info` | 挂载安装信息文件，供驱动探测逻辑定位驱动库 |
| `-v /usr/local/Ascend/driver/version.info:...` | 挂载驱动版本文件，供 CANN 读取驱动版本 |
| `-v /etc/vnpu.cfg:/etc/vnpu.cfg` | 算力切分持久化配置，仅在物理机特权容器的算力切分场景使用 |
| `--ipc=host` | 与宿主机共享 IPC 命名空间 |

调用链位置：设备节点挂载决定 Driver 层能否访问物理卡，驱动库挂载决定 AscendCL（Ascend Computing Language，CANN 对外提供的 C/C++ 编程接口）与 Runtime 层能否链接到 `libascendcl.so`。由谁决定：宿主机上已安装的驱动版本与要暴露的卡数。输入 → 输出：宿主机的卡与驱动库 → 容器内可见的 Device。

## 动手实操

执行前提：Linux + bash；root 权限的命令以 `sudo` 标出；下载目录假设为 `/home`，安装包文件名中的 `<ver>`、`<chip>`、`<arch>` 按上一节配套表替换。命令序列如下。

```bash
# 变量：取值来自官方版本配套表，选定一组后统一替换
CANN_VERSION=<cann_version>            # 例如 8.0.RC3
CHIP_NAME=<chip_name>                  # 例如 910b，决定 kernels 包名
ARCH=<arch>                            # aarch64 或 x86_64
DRIVER_VERSION=<driver_version>        # Ascend HDK 驱动版本
FIRMWARE_VERSION=<firmware_version>    # 固件版本
PYTORCH_VERSION=<pytorch_version>      # 与 torch_npu 配套
TORCH_NPU_VERSION=<torch_npu_version>  # 例如 2.1.0.post13

# 顺序 1：驱动与固件（物理机侧安装；首次安装为驱动 -> 固件）
sudo chmod +x Ascend-hdk-${CHIP_NAME}-npu-driver_${DRIVER_VERSION}_linux-${ARCH}.run
sudo chmod +x Ascend-hdk-${CHIP_NAME}-npu-firmware_${FIRMWARE_VERSION}.run
sudo ./Ascend-hdk-${CHIP_NAME}-npu-driver_${DRIVER_VERSION}_linux-${ARCH}.run --full --install-for-all
sudo ./Ascend-hdk-${CHIP_NAME}-npu-firmware_${FIRMWARE_VERSION}.run --full
sudo reboot
npu-smi info

# 顺序 2：CANN Toolkit 与算子包（先 Toolkit 后 Kernels）
sudo sh Ascend-cann-toolkit_${CANN_VERSION}_linux-${ARCH}.run --install
sudo sh Ascend-cann-kernels-${CHIP_NAME}_${CANN_VERSION}_linux-${ARCH}.run --install

# 顺序 3：设置环境变量
# root 默认安装路径；CANN 9.x 布局为 /usr/local/Ascend/cann/set_env.sh；
# 非 root 安装时根目录换成 ${HOME}/Ascend
echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc
source ~/.bashrc
echo "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"

# 顺序 4：Python / PyTorch / torch_npu
pip3 install torch==${PYTORCH_VERSION}
pip3 install pyyaml setuptools
pip3 install torch-npu==${TORCH_NPU_VERSION}
```

`pip3 install torch==2.1.0` 与 `pip3 install torch-npu==2.1.0.post13` 是 Gitee `Ascend/pytorch` 给出的具体安装包组合，对应 CANN 8.2.RC1；`Ascend-cann-toolkit_8.0.RC3_linux-aarch64.run` 与 `Ascend-cann-kernels-910b_8.0.RC3_linux-aarch64.run` 是 Ascend 开源快速安装文档给出的 Toolkit 与算子包示例。两组示例各自对应自己的 CANN 版本，选定一套组合后统一替换变量。

完整的环境校验脚本由两个文件组成，放在同一目录：

`check_env.sh`

```bash
#!/usr/bin/env bash
# 环境校验：设备数、卡型号、显存、驱动/固件/CANN 版本、torch 与 torch_npu 版本
# 前提：已完成本章安装顺序，且当前 Shell 已 source <install_path>/set_env.sh
# 前提：check_env.py 与本脚本放同一目录，从该目录执行 bash check_env.sh
set -u

echo "===== 1. 设备清单 ====="
npu-smi info -l

echo "===== 2. 设备实时状态 ====="
npu-smi info

echo "===== 3. 驱动版本 ====="
if [ -f /usr/local/Ascend/driver/version.info ]; then
  cat /usr/local/Ascend/driver/version.info
else
  echo "driver/version.info 不在默认路径，请按实际安装路径调整"
fi

echo "===== 4. 固件版本 ====="
if [ -f /usr/local/Ascend/firmware/version.info ]; then
  cat /usr/local/Ascend/firmware/version.info
else
  echo "firmware/version.info 不在默认路径，请按实际安装路径调整"
fi

echo "===== 5. CANN Toolkit 版本 ====="
find /usr/local/Ascend -name ascend_toolkit_install.info -print 2>/dev/null | while read -r f; do
  echo "# $f"
  cat "$f"
done

echo "===== 6. Python / PyTorch / torch_npu / 设备显存 ====="
python3 ./check_env.py
```

`check_env.py`

```python
#!/usr/bin/env python3
"""输出设备数、卡型号、显存、CANN 版本、torch 与 torch_npu 版本。

前提：已 source <install_path>/set_env.sh，且已安装 CANN、torch、torch_npu。
"""
import os
from pathlib import Path

import torch
import torch_npu


def cann_version() -> str:
    """从 ascend_toolkit_install.info 读取版本；安装布局随 CANN 版本变化。"""
    roots = []
    home = os.environ.get("ASCEND_HOME_PATH")
    if home:
        roots.append(Path(home))
    roots.append(Path("/usr/local/Ascend"))
    for root in roots:
        if not root.is_dir():
            continue
        for info in sorted(root.rglob("ascend_toolkit_install.info")):
            for line in info.read_text(errors="ignore").splitlines():
                if line.startswith("version="):
                    return f"{line.split('=', 1)[1]}  ({info})"
    return "未知（先 source set_env.sh，并核对 Toolkit 安装路径）"


def main() -> None:
    print("torch        =", torch.__version__)
    print("torch_npu    =", torch_npu.__version__)
    print("cann         =", cann_version())
    print("is_available =", torch.npu.is_available())
    count = torch.npu.device_count()
    print("device_count =", count)
    for device_id in range(count):
        name = torch.npu.get_device_name(device_id)
        free, total = torch.npu.mem_get_info(device_id)
        print(
            f"device[{device_id}]    = {name}, "
            f"HBM free/total = {free / 1024**3:.2f} / {total / 1024**3:.2f} GiB"
        )


if __name__ == "__main__":
    main()
```

执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash check_env.sh
```

预期输出的关键行（尖括号为占位，取值随本机硬件与所选版本变化；脚本输出属于实测值，环境为已按本章顺序装好的 Linux 主机，方法为在已加载环境变量的 Shell 中执行 `bash check_env.sh`）：

```
===== 1. 设备清单 =====
Card Count             : <可见卡数>
NPU ID                 : 0
Chip Count             : <每卡芯片数>
===== 2. 设备实时状态 =====
| npu-smi <工具版本> Version: <驱动版本> |
| <NPU ID>  <芯片名>  | OK | <功率> <温度> <大页占用> |
| <Chip> | <Bus-Id> | <AICore(%)> <Memory-Usage(MB)> <HBM-Usage(MB)> |
===== 3. 驱动版本 =====
Version=<驱动版本>
===== 4. 固件版本 =====
Version=<固件版本>
===== 5. CANN Toolkit 版本 =====
# <ascend_toolkit_install.info 所在路径>
version=<CANN 版本>
===== 6. Python / PyTorch / torch_npu / 设备显存 =====
torch        = <PyTorch 版本>
torch_npu    = <torch_npu 安装包版本>
cann         = <CANN 版本>  (<安装信息文件路径>)
is_available = True
device_count = <可见卡数>
device[0]    = <芯片名>, HBM free/total = <空闲 GiB> / <总量 GiB>
```

验收条件：`is_available` 为 `True`，`device_count` 大于 0，`device_count` 与 `npu-smi info -l` 的 `Card Count` 一致；`device[i]` 的 HBM 总量换算成 MB 后与 `npu-smi info` 的 `HBM-Usage(MB)` 总量一致（$1\ \mathrm{GiB} = 1024\ \mathrm{MB}$）；第 5 段读到的 CANN 版本与第 6 段 `torch_npu` 版本落在同一组官方配套关系里。

## 常见坑

| 现象 | 原因 | 处理 |
|---|---|---|
| `npu-smi info` 报 `No device found`，`/dev` 下没有 `davinci0` | 驱动与固件未装或加载失败；容器启动时漏挂设备节点 | 宿主机按官方《安装NPU驱动和固件》重装并 `reboot`；容器补齐 `--device=/dev/davinci0`、`/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc` |
| `npu-smi info` 正常，`import torch_npu` 报驱动与 CANN 版本不匹配，或算子执行返回错误码 | Ascend HDK 版本与 CANN 版本、kernels 包芯片型号没落在同一组配套关系里 | 用版本配套查询助手选定一组版本，按驱动固件 → Toolkit → kernels → torch_npu 的顺序重装；kernels 包名中的 `<chip>` 换成实际芯片型号 |
| 编译或运行时报 `libascendcl.so: cannot open shared object file` | `LD_LIBRARY_PATH` 未包含 CANN 与驱动库目录，或 kernels 包未安装 | 先 `source <install_path>/set_env.sh`，再 `echo $LD_LIBRARY_PATH` 确认非空；补装 `Ascend-cann-kernels-<chip>_<ver>_linux-<arch>.run` |
| 普通用户执行 `npu-smi info` 或跑应用提示权限不足 | 运行用户未加入驱动固件的默认运行用户组 `HwHiAiUser`，或设备节点属组不匹配 | 用 `sudo usermod -aG HwHiAiUser <username>` 把运行用户加组并重新登录；安装驱动时带 `--install-for-all` |
| 新开终端后 `ASCEND_HOME_PATH` 为空、`import torch_npu` 失败 | `set_env.sh` 只在当前 Shell 生效，环境变量未加载 | 把 `source <install_path>/set_env.sh` 写进 `~/.bashrc` 后 `source ~/.bashrc`；用 `echo $ASCEND_HOME_PATH` 验证 |
| 多个 CANN 版本共存时算子找不到、版本回显混乱 | `latest` 符号链接或 `LD_LIBRARY_PATH` 指向了另一套 CANN | 确认 `<install_path>/ascend-toolkit/latest` 指向配套版本；只 `source` 目标版本的 `set_env.sh`，清理 Shell 启动文件里多余的 `source` 行 |

## 自测题

1. 一台 Atlas 800T A2 训练服务器首次安装驱动与固件（此前未装过，也未卸载过），手上有 `Ascend-hdk-<chip>-npu-driver_<ver>_linux-aarch64.run` 与 `Ascend-hdk-<chip>-npu-firmware_<ver>.run`。写出安装顺序、两条安装命令与安装后的验证命令。
2. 某环境 `import torch_npu` 报 `ImportError: libhccl.so: cannot open shared object file`，`echo $LD_LIBRARY_PATH` 输出为空。指出原因，并写出两条处理命令（root 默认安装路径，CANN 8.x 布局）。
3. `torch.npu.is_available()` 为 `True`，`torch.npu.device_count()` 为 0，宿主机 `npu-smi info` 能看到 8 张卡且 `Health` 全为 `OK`，应用跑在容器里。列出至少三个排查方向。
4. 逐条说明 `docker run` 中 `--device=/dev/davinci0`、`-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info`、`--ipc=host` 的作用，并回答容器内是否需要安装驱动。
5. CANN 9.0.0 版本说明列出的配套 Ascend HDK 版本有哪些？只写出处与查询入口，不要凭记忆补充表中没有的版本号。

## 自测题答案

1. 首次安装按「驱动 → 固件」。命令为 `sudo ./Ascend-hdk-<chip>-npu-driver_<ver>_linux-aarch64.run --full --install-for-all` 与 `sudo ./Ascend-hdk-<chip>-npu-firmware_<ver>.run --full`，安装提示要求重启时执行 `sudo reboot`。验证命令是 `npu-smi info`，返回驱动相关信息即加载成功。覆盖安装场景顺序相反，为「固件 → 驱动」。
2. 原因是 `set_env.sh` 未在当前 Shell 执行，`LD_LIBRARY_PATH` 没有指向 CANN 与驱动库目录。处理命令：`source /usr/local/Ascend/ascend-toolkit/set_env.sh`；`echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc && source ~/.bashrc`。CANN 9.x 布局把路径换成 `/usr/local/Ascend/cann/set_env.sh`。
3. 排查方向：容器启动时是否漏挂 `/dev/davinci0` 等设备节点；容器内 `LD_LIBRARY_PATH` 是否指向宿主机驱动库；`torch_npu` 与 CANN 版本是否配套；CANN 与 Ascend HDK 版本是否配套；`set_env.sh` 是否指向了另一套 CANN；运行用户是否在 `HwHiAiUser` 组。宿主机 `npu-smi info` 正常只说明驱动层可用，Device 可见性还要由容器挂载与上层库共同决定。
4. `--device=/dev/davinci0` 把 0 号 NPU 设备节点映射进容器；`-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info` 把宿主机驱动版本文件挂进容器，供 CANN 读取驱动版本；`--ipc=host` 让容器与宿主机共享 IPC 命名空间。容器内不安装驱动与固件，只安装 CANN 软件，驱动库由宿主机挂载进去。
5. 出处是 [CANN 9.0.0 版本说明](https://www.hiascend.com/document/detail/zh/canncommercial/900/releasenote/release-notes.md) 的 `CANN与驱动版本配套关系` 表，其中列出的组合为 Ascend HDK 25.7.RC1、26.0.RC1、25.5.X、25.3.X、25.2.X、25.0.X。查询入口是昇腾社区[版本配套查询助手](https://www.hiascend.com/developer/download/compatibility)。表中没有的版本号不填，以查询助手与对应版本说明为准。

## 参考来源

- [安装NPU驱动和固件](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0005.html) — 查证于 2026-11-14；支撑首次安装与覆盖安装的顺序、驱动与固件包名参数、`--full` / `--install-for-all`
- [快速安装昇腾环境](https://ascend.readthedocs.io/zh-cn/latest/sources/ascend/quick_install.html) — 查证于 2026-11-14；支撑 Toolkit 与 kernels 包名、`--install` 参数、`set_env.sh` 写入 `~/.bashrc` 的写法
- [安装CANN](https://www.hiascend.com/doc_center/source/zh/canncommercial/800/softwareinst/instg/instg_0054.html) — 查证于 2026-11-14；支撑默认安装路径、`set_env.sh` 路径与 `ascend_toolkit_install.info` 查版本的方法
- [Ascend Extension for PyTorch README](https://gitee.com/ascend/pytorch) — 查证于 2026-11-14；支撑 `pip3 install torch==2.1.0`、`pip3 install torch-npu==2.1.0.post13` 与 CANN 8.2.RC1 配套行
- [CANN 9.0.0版本说明](https://www.hiascend.com/document/detail/zh/canncommercial/900/releasenote/release-notes.md) — 查证于 2026-11-14；支撑 CANN 与 Ascend HDK 版本配套关系表
- [版本说明（CANN 社区版 9.2.0）](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920/softwareinst/releasenote) — 查证于 2026-11-14；支撑 `CANN软件版本配套表` 与版本兼容性说明
- [TorchNPU 26.0.0 Release Notes](https://www.hiascend.com/document/detail/en/Pytorch/2600/releasenote/docs/en/release_notes/release_notes.md) — 查证于 2026-11-14；支撑 CANN 9.0.0 / PyTorch 2.7.1 / 2.7.1.post4 配套行与 Driving SDK 兼容说明
- [版本配套查询助手](https://www.hiascend.com/developer/download/compatibility) — 查证于 2026-11-14；支撑「以官方配套表为准」的查询入口，页面说明其呈现昇腾全栈软件与核心三方软件的版本配套关系
- [宿主机目录挂载至容器](https://www.hiascend.com/doc_center/source/zh/canncommercial/700/envdeployment/instg/instg_0099.html) — 查证于 2026-11-14；支撑 `docker run` 的四个 `--device` 参数、各 `-v` 挂载项与容器内不装驱动的说明
- [AI-accelerated ECSs](https://support.huaweicloud.com/intl/es-us/ae-ad-1-usermanual-ecs/en-us_topic_0147953501.html) — 查证于 2026-11-14；支撑华为云 Ai1 / Ai2 规格、Ascend 310 系列与 Atlas 300I 板卡、公共镜像预装开发套件包
