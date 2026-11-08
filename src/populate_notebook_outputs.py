"""
Script to execute code cells in all 6 Jupyter notebooks and embed rich execution outputs
(stdout text, tables, and base64 matplotlib plots) directly into the notebook JSON files.

Authors: Rup Narayan Jha, Jaypal Malviya, Anuj Kumar, Ravi Kiran
Mehta Family School of Data Science and Artificial Intelligence, IIT Guwahati
"""

import json
import os
import sys
import io
import base64
import contextlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NOTEBOOKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notebooks"))
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

def execute_notebook(nb_filename):
    nb_path = os.path.join(NOTEBOOKS_DIR, nb_filename)
    print(f"\nExecuting {nb_filename}...")
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Execution namespace
    namespace = {
        '__name__': '__main__',
        '__file__': nb_path,
    }
    
    # Ensure src is on sys.path
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)

    # Change working directory to notebooks/ so relative paths work (e.g. '../data/...')
    old_cwd = os.getcwd()
    os.chdir(NOTEBOOKS_DIR)

    exec_counter = 1

    try:
        for cell_idx, cell in enumerate(nb['cells']):
            if cell['cell_type'] != 'code':
                continue

            source_code = "".join(cell['source'])
            # Remove ipython magics if any
            clean_lines = []
            for line in source_code.split("\n"):
                if line.strip().startswith("%matplotlib") or line.strip().startswith("%"):
                    continue
                clean_lines.append(line)
            executable_code = "\n".join(clean_lines)

            # Capture stdout & stderr
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()

            plt.close('all')

            outputs = []

            try:
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    # Try eval if single expression or exec
                    exec(executable_code, namespace)

                stdout_text = stdout_buf.getvalue()
                if stdout_text:
                    outputs.append({
                        "name": "stdout",
                        "output_type": "stream",
                        "text": stdout_text.splitlines(keepends=True)
                    })

                # Check if a matplotlib figure was created
                fig_nums = plt.get_fignums()
                if fig_nums:
                    for fig_num in fig_nums:
                        fig = plt.figure(fig_num)
                        img_buf = io.BytesIO()
                        fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=130)
                        img_buf.seek(0)
                        b64_data = base64.b64encode(img_buf.read()).decode('utf-8')
                        outputs.append({
                            "data": {
                                "image/png": b64_data,
                                "text/plain": ["<Figure size ...>"]
                            },
                            "metadata": {},
                            "output_type": "display_data"
                        })
                    plt.close('all')

            except Exception as e:
                print(f"  [Cell {cell_idx}] Note: {e}")
                err_text = f"Exception: {type(e).__name__}: {e}\n"
                outputs.append({
                    "name": "stderr",
                    "output_type": "stream",
                    "text": [err_text]
                })

            cell['execution_count'] = exec_counter
            cell['outputs'] = outputs
            exec_counter += 1

        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)

        print(f"Successfully populated {nb_filename} with executed outputs ({exec_counter - 1} code cells).")

    finally:
        os.chdir(old_cwd)


if __name__ == '__main__':
    notebooks = [
        "01_data_preprocessing.ipynb",
        "02_custom_mstl_implementation.ipynb",
        "03_library_and_benchmark_comparison.ipynb",
        "04_synthetic_data_experiments.ipynb",
        "05_extensions_and_ablation.ipynb",
        "06_robustness_and_scalability.ipynb"
    ]
    for nb_file in notebooks:
        execute_notebook(nb_file)
    print("\nAll 6 notebooks successfully populated with outputs and plots!")

