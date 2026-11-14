#!/usr/bin/env python3
"""检查昇腾教程各章节的结构、风格与来源完整性。

用法（在仓库任意目录执行）：
    python3 scripts/check-chapters.py

发现问题时以非零状态退出，可用于本地自检或 CI。
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / 'chapters'
EXPECTED = [
    '01-hardware-and-stack.md',
    '02-environment-setup.md',
    '03-ascendcl-cpp.md',
    '04-pytorch-torch-npu.md',
    '05-model-serving.md',
    '06-profiling-and-tuning.md',
    '07-distributed-training.md',
    '08-ascend-c-operator.md',
    '09-troubleshooting.md',
]
BANNED_WORDS = ['啃', '乍一看', '堆出来', '落地', '沉淀', '锁死', '定死', '钉死',
                '破局', '伏笔', '翻车', '串台', '压根', '收紧']
SECTIONS = ['本章目标', '先说结论', '动手实操', '常见坑', '自测题', '自测题答案', '参考来源']
PATTERN_NOT_BUT = re.compile(r'不是[^，。；\n]{0,50}[，,]?\s*而是')

total_words = 0
problems = []
print(f'{"文件":<38} {"字数":>6} {"来源":>4} {"结构":>4} {"代码块":>6}')
for name in EXPECTED:
    path = ROOT / name
    if not path.exists():
        problems.append(f'{name}: 文件缺失')
        print(f'{name:<38} {"缺失":>6}')
        continue
    text = path.read_text(encoding='utf-8')
    words = len(text)
    total_words += words
    links = len(re.findall(r'https?://[^\s)]+', text))
    missing_sections = [s for s in SECTIONS if s not in text]
    code_blocks = len(re.findall(r'^```', text, flags=re.M))
    print(f'{name:<38} {words:>6} {links:>4} {"齐" if not missing_sections else "缺":>4} {code_blocks:>6}')

    if words < 1500:
        problems.append(f'{name}: 篇幅偏短 ({words} 字)')
    if links < 4:
        problems.append(f'{name}: 参考链接不足 ({links} 条)')
    if missing_sections:
        problems.append(f'{name}: 缺少小节 {missing_sections}')
    if code_blocks % 2 != 0:
        problems.append(f'{name}: 代码块围栏不成对 ({code_blocks} 个)')
    for word in BANNED_WORDS:
        if word in text:
            problems.append(f'{name}: 禁用词「{word}」')
    m = PATTERN_NOT_BUT.search(text)
    if m:
        problems.append(f'{name}: 禁用句式「{m.group()[:30]}」')
    # 代码块内不应出现 LaTeX 定界符
    inside = False
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.startswith('```'):
            inside = not inside
            continue
        if inside:
            # 去掉 shell / CMake 变量与命令替换，剩下的 $ 才是 LaTeX 定界符
            stripped = re.sub(r'\$\{[^}]*\}|\$[A-Za-z_][A-Za-z0-9_]*|\$[0-9]|\$\(', '', line)
            if re.search(r'(?<!\$)\$[^$\n]+\$', stripped):
                problems.append(f'{name}:{lineno}: 代码块内出现 LaTeX 定界符')
    if not text.startswith('# '):
        problems.append(f'{name}: 缺少一级标题')

print(f'\n合计 {total_words} 字')
if problems:
    print(f'\n待处理 {len(problems)} 项:')
    for p in problems:
        print(' -', p)
    sys.exit(1)
print('\n校验通过')
