import ast
import glob
import sys


def check_syntax(filepath):
    try:
        with open(filepath, "r") as f:
            source = f.read()
        ast.parse(source)
        return True
    except SyntaxError as e:
        print(f"Syntax error in {filepath}: {e}")
        return False
    except Exception as e:
        print(f"Error checking {filepath}: {e}")
        return False


# Auto-discover all Python files under src/heylook_llm/
files_to_check = sorted(glob.glob("src/heylook_llm/**/*.py", recursive=True))

# Exclude list for files with known issues (empty for now)
exclude = set()

files_to_check = [f for f in files_to_check if f not in exclude]

if not files_to_check:
    print("No Python files found to check.")
    sys.exit(1)

success = True
for file in files_to_check:
    if not check_syntax(file):
        success = False

if success:
    print(f"All {len(files_to_check)} files passed syntax check.")
    sys.exit(0)
else:
    print("Some files failed syntax check.")
    sys.exit(1)
