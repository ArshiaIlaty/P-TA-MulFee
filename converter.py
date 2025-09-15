import os
import subprocess
import shutil
from pathlib import Path

# Define root of your project and output directory
PROJECT_ROOT = Path(__file__).parent  # Current directory is the project root
OUTPUT_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define directories to exclude
EXCLUDE_DIRS = {
    "reports",           # Output directory
    "__pycache__",       # Python cache
    "wandb",            # Weights & Biases logs
    "checkpoints",      # Model checkpoints
    "dpo_training_logs", # Training logs
    "ppo_training_logs", # Training logs
    "gpt2_dpo_hierarchical_heloc_step*",  # DPO checkpoints
    "gpt2_ppo_hierarchical_heloc_lessKL_step*",  # PPO checkpoints
    "hierarchical_discriminators_heloc_checkpoints",  # Discriminator checkpoints
    ".specstory"        # Spec story history files
}

def check_pandoc():
    """Check if pandoc is available"""
    if not shutil.which("pandoc"):
        print("❌ Pandoc is not installed or not in PATH")
        print("Please install pandoc: https://pandoc.org/installing.html")
        return False
    return True

def should_convert_file(file_path):
    """Check if file should be converted using directory exclusion approach"""
    # Don't convert files in excluded directories
    for exclude_dir in EXCLUDE_DIRS:
        if exclude_dir in file_path.parts:
            return False
    
    # Don't convert the converter script itself
    if file_path.name == "converter.py":
        return False
    
    # Don't convert README files (optional - remove if you want them)
    if file_path.name.lower().startswith("readme"):
        return False
    
    # Don't convert log files, aux files, etc.
    if file_path.suffix in ['.log', '.aux', '.out', '.toc', '.pdf']:
        return False
    
    # Only convert .tex and .md files
    if file_path.suffix.lower() not in ['.tex', '.md']:
        return False
    
    return True

def main():
    # Check if pandoc is available
    if not check_pandoc():
        return
    
    # Find all .md and .tex files in the project
    all_md_files = list(PROJECT_ROOT.rglob("*.md"))
    all_tex_files = list(PROJECT_ROOT.rglob("*.tex"))
    
    # Filter files that should be converted
    files_to_convert = [
        f for f in all_md_files + all_tex_files 
        if should_convert_file(f)
    ]
    
    if not files_to_convert:
        print("No .md or .tex files found to convert.")
        return
    
    print(f"Found {len(files_to_convert)} files to convert:")
    for f in files_to_convert:
        print(f"  - {f}")
    print()
    
    converted_count = 0
    error_count = 0
    
    for file_path in files_to_convert:
        output_file = OUTPUT_DIR / (file_path.stem + ".pdf")
        print(f"Converting {file_path.name} → {output_file.name}")
        
        try:
            # Pandoc command
            subprocess.run(
                [
                    "pandoc", str(file_path),
                    "-o", str(output_file),
                    "--pdf-engine=xelatex",  # You can also use lualatex
                    "-V", "geometry:margin=1in"
                ],
                check=True,
                capture_output=True,
                text=True
            )
            converted_count += 1
            print(f"  ✅ Success")
        except subprocess.CalledProcessError as e:
            error_count += 1
            print(f"  ❌ Error: {e}")
            if e.stdout:
                print(f"    stdout: {e.stdout}")
            if e.stderr:
                print(f"    stderr: {e.stderr}")

    print(f"\n📊 Summary:")
    print(f"  ✅ Successfully converted: {converted_count}")
    print(f"  ❌ Errors: {error_count}")
    print(f"  📁 Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()