#!/usr/bin/env python3
"""把 README 与各章合并成一份可单文件阅读的完整教程。

用法（在仓库任意目录执行）：
    python3 scripts/build-fullbook.py

输出文件为生成产物，由 .gitignore 忽略，不纳入版本管理。
"""
from datetime import date
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
CHAPTERS = [
    ('01', '01-hardware-and-stack.md', '硬件与软件栈全景'),
    ('02', '02-environment-setup.md', '环境准备与第一次跑通'),
    ('03', '03-ascendcl-cpp.md', 'AscendCL C++ 入门'),
    ('04', '04-pytorch-torch-npu.md', 'PyTorch 与 torch_npu'),
    ('05', '05-model-serving.md', '模型迁移与推理服务'),
    ('06', '06-profiling-and-tuning.md', '性能观测与调优'),
    ('07', '07-distributed-training.md', '分布式训练'),
    ('08', '08-ascend-c-operator.md', 'Ascend C 自定义算子'),
    ('09', '09-troubleshooting.md', '故障排查与速查手册'),
]

readme = (ROOT / 'README.md').read_text(encoding='utf-8')
# 取 README 中「导言」到「自测与练习」之前的通用内容
start = readme.index('## 导言：手边没有昇腾设备时怎么动手')
end = readme.index('## 自测与练习')
front = readme[start:end].replace('## 章节导航', '## 分章文件')

parts = [
    '# 昇腾 AI 处理器实操教程（完整版）\n',
    '> 面向后端程序员的华为昇腾平台入门与动手指南。分章文件见 `chapters/`，本文件为合并阅读版。\n',
    f'> 生成日期：{date.today().isoformat()}\n',
    '\n---\n',
    front.strip(),
    '\n\n---\n',
]
for num, filename, title in CHAPTERS:
    path = ROOT / 'chapters' / filename
    if not path.exists():
        raise SystemExit(f'缺少章节文件: {path}')
    body = path.read_text(encoding='utf-8').strip()
    # 章节内相对链接指向 chapters/，合并版位于项目根，需要补前缀
    body = re.sub(r'\]\(\./(?!chapters/)([^)]+\.md)\)', r'](./chapters/\1)', body)
    parts.append(body)
    parts.append('\n\n---\n')

parts.append('## 自测与练习\n\n' + readme[end:].split('\n', 1)[1].strip() + '\n')
out = ROOT / '昇腾AI处理器实操教程-完整版.md'
out.write_text('\n'.join(parts), encoding='utf-8')
print(f'已生成 {out}，共 {len(out.read_text(encoding="utf-8"))} 字，{len(CHAPTERS)} 章')
