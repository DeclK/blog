#!/usr/bin/env python3
"""
Fix formatting issues with $$ math blocks in Markdown files

Editors like Typora allow $$ blocks to have no blank lines with surrounding content,
but standard Markdown parsers require block elements to have blank lines before and after.
This script automatically adds appropriate blank lines for $$ blocks while preserving
list indentation and quotation markers.

This is not enabled.
"""

import sys
import re
from pathlib import Path
from typing import List, Tuple

def is_math_block(line: str) -> bool:
    if not line:
        return False
    
    stripped_line = line.rstrip()
    # check it only contains > space and $
    pattern = r'^(?:[>\s]*)?\$\$\s*$'
    return bool(re.match(pattern, stripped_line))

def is_inline_math(line: str):
    if "$" in line:
        if is_math_block(line): return False
        return True
    return False

def fix_inline_math(line: str) -> str:
    """ Some inline math have $$ formula $$ in it, but it should
    be $ formula $
    """
    if not is_inline_math(line): return line
    # replace $$ with $
    line = re.sub(r'\$\$', '$', line)
    # remove extra space around $
    line = re.sub(r'\s*\$\s*', '$', line)
    return line

def detect_line_context(line: str) -> Tuple[str, str]:
    """ If this is a math block, get its prefix
    e.g. input: "> $$", return "> "
    e.g. input: "> > $$", return "> > "
    e.g. input: "       $$", return "       "
    """
    if not is_math_block(line):
        return None
    
    # Match all prefixes (quotation markers and spaces) until $$ is encountered
    match = re.match(r'^([>\s]*)\$\$', line.rstrip())
    if match:
        return match.group(1)
    return ''

def is_empty_line_match(line: str, prefix: str):
    return line.strip() == prefix.strip()

def fix_indent(line: str):
    """ Use regex to make indent fixed to 4, this is a hack
    for typora editor, some indent is 3, so we need to fix it
    """
    leading_spaces = len(line) - len(line.lstrip(' '))
    if leading_spaces == 0:
        return line
    
    indent_size = 4
    indent_level = (leading_spaces + indent_size - 1) // indent_size
    new_indent = ' ' * (indent_level * indent_size)
    
    return new_indent + line.lstrip(' ')
def fix_math_blocks(content: str) -> str:
    """ Fix math blocks
    """
    lines = content.split('\n')
    result: List[str] = []
    i = 0
    n = len(lines)
    
    while i < n:
        line = lines[i]
        line = fix_inline_math(line)
        line = fix_indent(line)
        
        # if this is a math block, add empty line before and after
        if is_math_block(line):
            # use prefix as empty line
            prefix = detect_line_context(line)
            if result and not is_empty_line_match(result[-1], prefix):
                empty_line = prefix
                result.append(empty_line)

            result.append(line)
            i += 1

            # find end of math block
            while i < n:
                next_line = lines[i]
                next_line = fix_indent(next_line)
                result.append(next_line)

                if is_math_block(next_line):
                    prefix = detect_line_context(next_line)
                    if i + 1 < n and not is_empty_line_match(lines[i + 1], prefix):
                        empty_line = prefix
                        result.append(empty_line)
                    i += 1
                    break
                i += 1
            continue

        result.append(line)
        i += 1

    fixed = '\n'.join(result)
    return fixed

        
def process_single_file(filepath: Path) -> bool:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        fixed = fix_math_blocks(content)
        
        # Check if there are any changes
        if fixed == content:
            print(f"  {filepath} (no changes needed)")
            return True
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed)
        print(f"✓ {filepath}")
        
        return True
        
    except Exception as e:
        print(f"✗ {filepath}: {e}", file=sys.stderr)
        return False

def collect_files(paths: List[str]) -> List[Path]:
    files: List[Path] = []
    
    for path_str in paths:
        path = Path(path_str)
        if path.is_file():
            if path.suffix == '.md':
                files.append(path)
        elif path.is_dir():
            files.extend(list(path.rglob('*.md')))

    return sorted(set(files))

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'paths',
        nargs='+',
        help='Markdown file or directory paths'
    )

    args = parser.parse_args()
    
    files_to_process = collect_files(args.paths)
    
    if not files_to_process:
        print("No Markdown files found", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files_to_process)} files\n")

    # Process files
    success = True
    for filepath in files_to_process:
        if not process_single_file(filepath):
            success = False
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()