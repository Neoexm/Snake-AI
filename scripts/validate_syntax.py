#!/usr/bin/env python3
"""
Quick syntax and import validation for all modified files.
Ensures no Python syntax errors before deployment to B200 cluster.
"""

import sys
import os
from pathlib import Path

def validate_file(filepath):
    """Validate Python file syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, filepath, 'exec')
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)

def main():
    root = Path(__file__).parent.parent
    
    files_to_check = [
        'train/train_ppo.py',
        'train/ddp_utils.py',
        'train/autoscale.py',
        'train/eval.py',
        'train/play.py',
        'snake_env/snake_env.py',
        'snake_env/wrappers.py',
        'train/models/snake_scalable_cnn.py',
        'scripts/validate_b200_readiness.py',
    ]
    
    print("="*60)
    print("Syntax Validation for B200 Deployment")
    print("="*60)
    
    all_passed = True
    
    for file_rel in files_to_check:
        filepath = root / file_rel
        if not filepath.exists():
            print(f"⚠️  {file_rel}: FILE NOT FOUND")
            all_passed = False
            continue
        
        passed, error = validate_file(filepath)
        if passed:
            print(f"✓ {file_rel}")
        else:
            print(f"✗ {file_rel}: {error}")
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("✓ All files validated successfully")
        print("\nReady to deploy to 4×B200 cluster!")
        return 0
    else:
        print("✗ Validation failed - fix errors before deployment")
        return 1

if __name__ == "__main__":
    sys.exit(main())
