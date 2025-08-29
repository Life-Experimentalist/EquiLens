#!/usr/bin/env python3
"""Fix corrupted emoji in CLI file"""

import re

with open('src/equilens/cli.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Replace the corrupted character pattern
content = re.sub(r'console\.print\(".\s\[green\]Using enhanced auditor', 'console.print("🚀 [green]Using enhanced auditor', content)

with open('src/equilens/cli.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed corrupted emoji in CLI')
