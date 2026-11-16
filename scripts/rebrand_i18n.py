#!/usr/bin/env python3
"""
Rebrands translation catalogs in crates/ui-egui/src/i18n/*.tsv
Replaces "Fotodit" with "Fotodit" and "fotodit" with "fotodit" in both keys and translations.
"""

import glob
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TSV_DIR = ROOT / "crates" / "ui-egui" / "src" / "i18n"

def rebrand_tsv():
    tsv_files = list(TSV_DIR.glob("*.tsv"))
    print(f"Processing {len(tsv_files)} translation catalogs...")
    
    total_lines_changed = 0
    for path in tsv_files:
        lines = path.read_text(encoding="utf-8").splitlines()
        new_lines = []
        changed = 0
        for line in lines:
            if line.startswith("#") or not line.strip():
                # For header comments, replace Fotodit with Fotodit
                new_line = line.replace("Fotodit", "Fotodit").replace("fotodit", "fotodit")
                new_lines.append(new_line)
                if new_line != line:
                    changed += 1
                continue
            
            parts = line.split("\t")
            if len(parts) >= 2:
                # Replace in all columns
                new_parts = [p.replace("Fotodit", "Fotodit").replace("fotodit", "fotodit") for p in parts]
                new_line = "\t".join(new_parts)
                new_lines.append(new_line)
                if new_line != line:
                    changed += 1
            else:
                new_line = line.replace("Fotodit", "Fotodit").replace("fotodit", "fotodit")
                new_lines.append(new_line)
                if new_line != line:
                    changed += 1
                    
        if changed > 0:
            total_lines_changed += changed
            path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            print(f"  {path.name}: {changed} lines updated.")
            
    print(f"Completed: {total_lines_changed} lines updated across {len(tsv_files)} files.")

if __name__ == "__main__":
    rebrand_tsv()
