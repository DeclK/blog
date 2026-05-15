---
name: precommit
description: 提交前检查博客内容，包含列表格式检查、代码块语言标识检查、代码注释语言检查，并自动运行 helpers 脚本修复 math blocks 和重新生成导航信息。
user-invocable: true
allowed-tools: "Read Write Edit Bash Glob Grep"
---

# Precommit — 提交前博客完善

在用户准备提交新博客或修改已有博客时，执行以下检查与自动修复。

## 效率原则

**脚本扫描 → 批量修复**，避免逐行人工检查。每项检查先用几行 Python 正则扫描出所有问题位置，确认范围后再用一个脚本批量修复。

## 环境注意事项

- **Windows 编码**：`fix_math_blocks.py` 使用 `✓` 等 Unicode 字符输出，在 Windows GBK 环境下会报错。必须加 `PYTHONIOENCODING=utf-8`。
- **虚拟环境**：`generate_navs.py` 依赖 `mkdocs`，必须使用 `.venv/Scripts/python` 而非系统 Python。

## 规则

### 1. Markdown 列表格式检查

检查所有博客文章的列表（有序列表 `1.` / 无序列表 `-`）格式。**以下每个子规则都会导致列表渲染失败，必须全部修复。**

#### 1.1 列表前必须有空行（高频错误）

**这是最常见的渲染问题。** 当段落文本（非标题）后紧接列表时，中间**必须**有一个空行，否则标准 Markdown 解析器不会将其识别为列表。

```
# 错误 — 段落与列表之间没有空行，无法渲染为列表
其中：
- X 形状为 [M, K]（输入激活）
- W 形状为 [N, K]（权重）

# 正确 — 段落与列表之间有空行
其中：

- X 形状为 [M, K]（输入激活）
- W 形状为 [N, K]（权重）
```

**检测脚本**：用正则匹配「非空非标题行 + 直接跟列表项」的模式：

```python
import re

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

issues = []
for i, line in enumerate(lines):
    if i + 1 >= len(lines):
        break
    curr = line.rstrip()
    next_line = lines[i + 1]
    # 当前行是非空的普通段落（非标题、非列表、非代码块），下一行是列表
    if (curr and not curr.startswith('#') and not curr.startswith('-') 
        and not curr.startswith('1.') and not curr.startswith('```')
        and not curr.startswith('|')  # table
        and (next_line.lstrip().startswith('- ') or next_line.lstrip().startswith('1. '))):
        issues.append(i + 2)  # 1-indexed line number for next_line

if issues:
    print(f"Missing blank line before list at lines: {issues}")
```

**修复**：在问题行（列表项所在行）之前插入一个空行。

列表结束后也应有空行（除非紧接同级列表或标题）。

#### 1.2 缩进修正

列表嵌套时，缩进必须为 **4 个空格**。部分编辑器（如 Typora）会产生 3 空格缩进，导致 MathJax 在列表项内无法正确渲染公式。

**检测与修复**：逐行扫描，若前导空格数不是 4 的倍数，修正为最近的 4 的倍数级别。

```python
import re

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed = []
for line in lines:
    stripped = line.lstrip(' ')
    leading = len(line) - len(stripped)
    if leading > 0 and leading % 4 != 0:
        level = (leading + 2) // 4  # round to nearest 4-multiple level
        new_indent = ' ' * (level * 4)
        line = new_indent + stripped
    fixed.append(line)
```

```
# 错误（3 空格缩进）
1. 第一步
   1. 子步骤

# 正确（4 空格缩进）
1. 第一步
    1. 子步骤
```

#### 1.3 列表标记一致性

- 无序列表统一使用 `-`，不要混用 `-`、`*`、`+`
- 有序列表序号必须从 1 开始递增

### 2. 代码风格检查

扫描所有 Markdown 文件中的 fenced code block（`` ``` ``），确保：

#### 2.1 语言标识

每个代码块必须有正确的语言标识。常见语言标识参考：

| 代码内容 | 语言标识 |
|:---------|:---------|
| C++ | `cpp` 或 `c++` |
| Python | `python` |
| Shell / Bash | `shell` 或 `bash` |
| 纯文本 / 输出 | `txt` |
| CUDA C++ | `cpp`（内容含 CUDA 关键字的也使用 `cpp`） |

**检测脚本**：找所有 `` ``` `` 开头且后面没有语言标识的行（排除 closing fence）：

```python
import re

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Match opening fences without language identifier: ```\n (not closing fence)
# An opening fence is ``` at line start, not preceded by content on the same "block level"
# Simple heuristic: ``` followed by newline with no language tag
pattern = r'^```\s*$'
lines = content.split('\n')
issues = []
in_code_block = False
for i, line in enumerate(lines):
    if line.strip().startswith('```'):
        if not in_code_block:
            # Opening fence
            tag = line.strip()[3:].strip()
            if not tag:
                issues.append(i + 1)
            in_code_block = True
        else:
            in_code_block = False

if issues:
    print(f"Missing language identifier at lines: {issues}")
```

根据代码内容推断语言标识并补全。若无法确定，标记出来提醒用户手动补充。

#### 2.2 代码注释必须为英文

代码块内的注释必须使用**英文**。代码块外的叙述使用中文（参考 tech-note skill 规则 2）。

**批量翻译模式**：将所有翻译对收集为 dict，一次 Python 脚本全部替换，避免逐条 Edit：

```python
# 收集翻译对 → 批量替换
translations = [
    ('# 步骤 1: 获取张量形状', '# Step 1: get tensor shapes'),
    ('// 计算线程起始位置',   '// compute thread start offset'),
    # ... 更多翻译对
]

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

for old, new in translations:
    if old in content:
        content = content.replace(old, new)
    else:
        print(f'NOT FOUND: {old[:50]}...')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
```

**检测脚本**：搜索代码块内包含中文字符的注释行：

```python
import re

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Find code blocks and check for Chinese chars in comments
chinese_pattern = re.compile(r'[一-鿿]')
comment_pattern = re.compile(r'^\s*(//|#|/\*|\*)\s*(.+)', re.MULTILINE)

# Extract code blocks (text between ``` and ```)
blocks = re.findall(r'```(\w*)\n(.*?)```', content, re.DOTALL)
for lang, block in blocks:
    for match in comment_pattern.finditer(block):
        if chinese_pattern.search(match.group(2)):
            print(f"  [{lang}] {match.group(0).strip()[:80]}")
```

#### 2.3 代码块前后必须有空行（nl2br 兼容）

项目启用了 `nl2br` 扩展（段落内单换行 → `<br>`）。当文本与代码块之间没有空行分隔时，`nl2br` 会在代码块前后插入多余的 `<br>` 标签，导致代码块与周围文本之间出现异常大的间距。

```
# 错误 — 文本与代码块之间没有空行，nl2br 会插入 <br>，产生异常间距
**普通 GEMM** 计算的是：
```txt
Y = X * W^T
```
其中：

# 正确 — 代码块前后都有空行，nl2br 不会干扰
**普通 GEMM** 计算的是：

```txt
Y = X * W^T
```

其中：
```

**检测脚本**：

```python
import sys

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

issues = []
in_code = False
for i, line in enumerate(lines):
    s = line.rstrip()
    if s.startswith('```') and not in_code:
        # Opening fence: check previous line
        if i > 0:
            prev = lines[i-1].rstrip()
            if prev and not prev.startswith('#') and not prev.startswith('```'):
                issues.append((i+1, 'before'))
        in_code = True
    elif s.startswith('```') and in_code:
        # Closing fence: check next line
        if i + 1 < len(lines):
            nxt = lines[i+1].rstrip()
            if nxt and not nxt.startswith('#') and not nxt.startswith('```') and not nxt.startswith('-') and not nxt.startswith('1. '):
                issues.append((i+1, 'after'))
        in_code = False

for line_no, pos in issues:
    print(f'Line {line_no}: missing blank line {pos} code block')
```

**修复**：在代码块的开头 fence 之前和结尾 fence 之后各插入一个空行。

### 3. 运行 helpers 脚本

执行 `helpers/` 目录下的两个脚本，自动修复 math blocks 和导航信息。

#### 3.1 修复 Math Blocks

```bash
PYTHONIOENCODING=utf-8 python helpers/fix_math_blocks.py --paths <file1> <file2> ...
```

该脚本自动完成：
- 为 `$$` 公式块前后添加空行（标准 Markdown 解析器要求）
- 将 `$$ inline formula $$` 修正为 `$inline formula$`
- 修正 3 空格缩进为 4 空格缩进（与规则 1.2 协同）
- 保留列表缩进和引用标记 `>` 等上下文信息

#### 3.2 生成导航信息

```bash
.venv/Scripts/python helpers/generate_navs.py
```

该脚本自动完成：
- 标准化所有分类目录 `index.md` 的 front matter
- 检查 `mkdocs.yml` 中 blogging 插件和 document-dates 插件的目录配置是否与实际一致
- 在每个分类目录的 `index.md` 中自动生成或更新 `!!! abstract "Table of Contents"` 目录列表

## 执行流程

当用户调用 `/precommit` 时，按以下顺序执行：

1. **确认范围**：询问用户要检查的文件范围（全部 `docs/` 还是指定文件）。如用户未明确指定，默认检查 `docs/` 下所有 `.md` 文件。
2. **列表格式检查**：先用检测脚本扫描所有问题位置（1.1 缺空行、1.2 缩进），确认后批量修复。
3. **代码风格检查**：先用检测脚本扫描缺失语言标识（2.1）、中文注释（2.2）、代码块缺空行（2.3），确认后批量修复。
4. **运行 helpers 脚本**：按规则 3.1 和 3.2 顺序执行，注意 Windows 编码和 venv 路径。
5. **汇总报告**：列出所有修改的文件和修改内容摘要，提示用户 review 后再提交。
