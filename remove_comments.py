#!/usr/bin/env python3
"""
Script to remove comments from Python files and Jupyter notebooks recursively in a directory.
This script removes:
- Single-line comments (lines starting with #)
- Inline comments (# comments at the end of code lines)
- Multi-line comments (triple-quoted strings that are not docstrings)

For Jupyter notebooks, only code cells are processed - markdown cells are preserved.
By default, processes all Python files and notebooks recursively in subdirectories.

Usage: python remove_comments.py [--backup] [--exclude file1.py file2.ipynb ...] [--no-recursive] [--no-notebooks]
"""

import os
import shutil
import argparse
import ast
import tokenize
import io
import json
from typing import List, Set


def is_docstring(node: ast.AST) -> bool:
    """Check if a node is a docstring."""
    return (
        isinstance(node, ast.Expr) and
        isinstance(node.value, ast.Constant) and
        isinstance(node.value.value, str)
    )


def get_docstring_lines(filepath: str) -> Set[int]:
    """Get line numbers that contain docstrings."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        docstring_lines = set()
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                if (node.body and is_docstring(node.body[0])):
                    docstring_node = node.body[0]
                    for line_num in range(docstring_node.lineno, docstring_node.end_lineno + 1):
                        docstring_lines.add(line_num)
            elif isinstance(node, ast.Module):
                if (node.body and is_docstring(node.body[0])):
                    docstring_node = node.body[0]
                    for line_num in range(docstring_node.lineno, docstring_node.end_lineno + 1):
                        docstring_lines.add(line_num)
        
        return docstring_lines
    except Exception:
        return set()


def remove_comments_from_file(filepath: str, preserve_docstrings: bool = True) -> str:
    """Remove comments from a Python file while preserving docstrings."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get docstring line numbers if preserving docstrings
        docstring_lines = get_docstring_lines(filepath) if preserve_docstrings else set()
        
        # Use tokenize to properly handle comments
        lines = content.splitlines(True)
        result_lines = []
        
        try:
            tokens = tokenize.generate_tokens(io.StringIO(content).readline)
            comment_positions = []
            
            for token in tokens:
                if token.type == tokenize.COMMENT:
                    comment_positions.append((token.start, token.end))
        except tokenize.TokenError:
            # Fallback to simple regex if tokenize fails
            return remove_comments_simple(content, docstring_lines)
        
        # Process each line
        for line_num, line in enumerate(lines, 1):
            if line_num in docstring_lines:
                # Preserve docstring lines
                result_lines.append(line)
                continue
            
            # Check if this line has comments to remove
            line_has_comment = any(
                pos[0][0] == line_num for pos in comment_positions
            )
            
            if line_has_comment:
                # Find comment positions in this line
                line_comments = [
                    pos for pos in comment_positions 
                    if pos[0][0] == line_num
                ]
                
                if line_comments:
                    # Remove the comment part
                    comment_start_col = min(pos[0][1] for pos in line_comments)
                    cleaned_line = line[:comment_start_col].rstrip() + '\n'
                    
                    # Only keep the line if it's not empty or just whitespace
                    if cleaned_line.strip():
                        result_lines.append(cleaned_line)
                    else:
                        # Skip empty lines that were just comments
                        continue
                else:
                    result_lines.append(line)
            else:
                result_lines.append(line)
        
        return ''.join(result_lines)
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def remove_comments_simple(content: str, docstring_lines: Set[int]) -> str:
    """Simple fallback method using regex."""
    lines = content.splitlines()
    result_lines = []
    
    for line_num, line in enumerate(lines, 1):
        if line_num in docstring_lines:
            result_lines.append(line)
            continue
        
        # Remove inline comments (but be careful about # in strings)
        in_string = False
        string_char = None
        i = 0
        while i < len(line):
            char = line[i]
            
            if not in_string and char in ['"', "'"]:
                in_string = True
                string_char = char
            elif in_string and char == string_char:
                if i == 0 or line[i-1] != '\\':
                    in_string = False
                    string_char = None
            elif not in_string and char == '#':
                # Found a comment outside of a string
                line = line[:i].rstrip()
                break
            i += 1
        
        # Only keep non-empty lines
        if line.strip():
            result_lines.append(line)
    
    return '\n'.join(result_lines) + '\n' if result_lines else ''


def remove_comments_from_code_string(code: str, preserve_docstrings: bool = True) -> str:
    """Remove comments from a string of Python code."""
    if not code.strip():
        return code
    
    try:
        # Get docstring line numbers if preserving docstrings
        docstring_lines = set()
        if preserve_docstrings:
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                        if (node.body and is_docstring(node.body[0])):
                            docstring_node = node.body[0]
                            for line_num in range(docstring_node.lineno, docstring_node.end_lineno + 1):
                                docstring_lines.add(line_num)
                    elif isinstance(node, ast.Module):
                        if (node.body and is_docstring(node.body[0])):
                            docstring_node = node.body[0]
                            for line_num in range(docstring_node.lineno, docstring_node.end_lineno + 1):
                                docstring_lines.add(line_num)
            except Exception:
                pass
        
        # Use tokenize to properly handle comments
        try:
            tokens = tokenize.generate_tokens(io.StringIO(code).readline)
            comment_positions = []
            
            for token in tokens:
                if token.type == tokenize.COMMENT:
                    comment_positions.append((token.start, token.end))
        except tokenize.TokenError:
            # Fallback to simple regex if tokenize fails
            return remove_comments_simple(code, docstring_lines)
        
        # Process each line
        lines = code.splitlines(True)
        result_lines = []
        
        for line_num, line in enumerate(lines, 1):
            if line_num in docstring_lines:
                # Preserve docstring lines
                result_lines.append(line)
                continue
            
            # Check if this line has comments to remove
            line_has_comment = any(
                pos[0][0] == line_num for pos in comment_positions
            )
            
            if line_has_comment:
                # Find comment positions in this line
                line_comments = [
                    pos for pos in comment_positions 
                    if pos[0][0] == line_num
                ]
                
                if line_comments:
                    # Remove the comment part
                    comment_start_col = min(pos[0][1] for pos in line_comments)
                    cleaned_line = line[:comment_start_col].rstrip() + '\n'
                    
                    # Only keep the line if it's not empty or just whitespace
                    if cleaned_line.strip():
                        result_lines.append(cleaned_line)
                    else:
                        # Skip empty lines that were just comments
                        continue
                else:
                    result_lines.append(line)
            else:
                result_lines.append(line)
        
        return ''.join(result_lines)
    
    except Exception:
        # Fallback to simple processing
        return remove_comments_simple(code, set())


def process_notebook(filepath: str, preserve_docstrings: bool = True) -> bool:
    """Process a Jupyter notebook to remove comments from code cells."""
    try:
        # Read the notebook
        with open(filepath, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Process each cell
        modified = False
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                if isinstance(source, list):
                    # Join source lines into a single string
                    code = ''.join(source)
                else:
                    code = source
                
                if code.strip():
                    # Remove comments from the code
                    cleaned_code = remove_comments_from_code_string(code, preserve_docstrings)
                    
                    # Convert back to list format if original was a list
                    if isinstance(source, list) and cleaned_code != code:
                        cell['source'] = cleaned_code.splitlines(True)
                        modified = True
                    elif isinstance(source, str) and cleaned_code != code:
                        cell['source'] = cleaned_code
                        modified = True
        
        # Write back the modified notebook if changes were made
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=2, ensure_ascii=False)
        
        return modified
    
    except Exception as e:
        print(f"Error processing notebook {filepath}: {e}")
        return False


def get_files_to_process(directory: str, exclude_files: List[str] = None, recursive: bool = True, include_notebooks: bool = True) -> tuple[List[str], List[str]]:
    """Get all Python files and optionally Jupyter notebooks to process."""
    if exclude_files is None:
        exclude_files = []
    
    python_files = []
    notebook_files = []
    exclude_set = set(exclude_files)
    
    if recursive:
        # Walk through all subdirectories recursively
        for root, dirs, files in os.walk(directory):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for filename in files:
                if filename not in exclude_set:
                    filepath = os.path.join(root, filename)
                    if filename.endswith('.py'):
                        python_files.append(filepath)
                    elif include_notebooks and filename.endswith('.ipynb'):
                        notebook_files.append(filepath)
    else:
        # Only process files in the current directory
        for filename in os.listdir(directory):
            if filename not in exclude_set and os.path.isfile(os.path.join(directory, filename)):
                filepath = os.path.join(directory, filename)
                if filename.endswith('.py'):
                    python_files.append(filepath)
                elif include_notebooks and filename.endswith('.ipynb'):
                    notebook_files.append(filepath)
    
    return python_files, notebook_files


def main():
    parser = argparse.ArgumentParser(description='Remove comments from Python files and Jupyter notebooks')
    parser.add_argument('--backup', action='store_true', 
                       help='Create backup files with .bak extension')
    parser.add_argument('--exclude', nargs='*', default=[], 
                       help='Files to exclude from processing')
    parser.add_argument('--preserve-docstrings', action='store_true', default=True,
                       help='Preserve docstrings (default: True)')
    parser.add_argument('--directory', default='.', 
                       help='Directory to process (default: current directory)')
    parser.add_argument('--no-recursive', action='store_true',
                       help='Process only the specified directory (not subdirectories)')
    parser.add_argument('--no-notebooks', action='store_true',
                       help='Skip Jupyter notebook files (.ipynb)')
    
    args = parser.parse_args()
    
    # Add this script to exclude list to avoid modifying itself
    exclude_files = args.exclude + ['remove_comments.py']
    
    # Get all files to process
    recursive = not args.no_recursive
    include_notebooks = not args.no_notebooks
    python_files, notebook_files = get_files_to_process(args.directory, exclude_files, recursive, include_notebooks)
    
    total_files = len(python_files) + len(notebook_files)
    if total_files == 0:
        search_type = "recursively" if recursive else "in the current directory"
        print(f"No files found to process {search_type}.")
        return
    
    search_type = "recursively" if recursive else "in the current directory"
    print(f"Found {total_files} files to process {search_type}:")
    if python_files:
        print(f"  - {len(python_files)} Python files (.py)")
    if notebook_files:
        print(f"  - {len(notebook_files)} Jupyter notebooks (.ipynb)")
    
    # Group files by directory for better display
    all_files = python_files + notebook_files
    files_by_dir = {}
    for filepath in all_files:
        dir_path = os.path.dirname(filepath)
        rel_dir = os.path.relpath(dir_path, args.directory) if dir_path != args.directory else "."
        if rel_dir not in files_by_dir:
            files_by_dir[rel_dir] = []
        filename = os.path.basename(filepath)
        file_type = "📓" if filepath.endswith('.ipynb') else "🐍"
        files_by_dir[rel_dir].append(f"{file_type} {filename}")
    
    print("\nFiles to process:")
    for dir_path, files in sorted(files_by_dir.items()):
        print(f"  {dir_path}/")
        for filename in sorted(files):
            print(f"    - {filename}")
    
    # Confirm before proceeding
    response = input("\nProceed with removing comments? (y/N): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return
    
    # Process Python files
    processed_count = 0
    if python_files:
        print(f"\nProcessing {len(python_files)} Python files...")
        for filepath in python_files:
            rel_path = os.path.relpath(filepath, args.directory)
            print(f"Processing {rel_path}...")
            
            # Create backup if requested
            if args.backup:
                backup_path = filepath + '.bak'
                shutil.copy2(filepath, backup_path)
                print(f"  Created backup: {rel_path}.bak")
            
            # Remove comments
            cleaned_content = remove_comments_from_file(filepath, args.preserve_docstrings)
            
            if cleaned_content is not None:
                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(cleaned_content)
                    print(f"  ✓ Comments removed from {rel_path}")
                    processed_count += 1
                except Exception as e:
                    print(f"  ✗ Error writing to {rel_path}: {e}")
            else:
                print(f"  ✗ Failed to process {rel_path}")
    
    # Process Jupyter notebooks
    if notebook_files:
        print(f"\nProcessing {len(notebook_files)} Jupyter notebooks...")
        for filepath in notebook_files:
            rel_path = os.path.relpath(filepath, args.directory)
            print(f"Processing {rel_path}...")
            
            # Create backup if requested
            if args.backup:
                backup_path = filepath + '.bak'
                shutil.copy2(filepath, backup_path)
                print(f"  Created backup: {rel_path}.bak")
            
            # Remove comments from notebook
            success = process_notebook(filepath, args.preserve_docstrings)
            
            if success:
                print(f"  ✓ Comments removed from {rel_path}")
                processed_count += 1
            else:
                print(f"  ✗ No changes made to {rel_path}")
    
    print(f"\nProcessed {processed_count} out of {total_files} files.")
    
    if args.backup:
        print("\nBackup files created with .bak extension.")
        print("You can remove them with: find . -name '*.bak' -delete")


if __name__ == '__main__':
    main()
