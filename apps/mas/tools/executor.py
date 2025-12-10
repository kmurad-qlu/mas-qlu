import subprocess
import sys
import tempfile
import os
import threading
import time
from typing import Tuple

class PythonExecutor:
    """
    Secure-ish local Python execution sandbox.
    Runs code in a temporary file via subprocess with a timeout.
    """
    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def execute(self, code: str) -> Tuple[str, str, bool]:
        """
        Execute python code.
        Returns: (stdout, stderr, success)
        """
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            # Run the process
            start_time = time.perf_counter()
            process = subprocess.Popen(
                [sys.executable, tmp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd() # Run in CWD to access local files if needed
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                return stdout, stderr, (process.returncode == 0)
            except subprocess.TimeoutExpired:
                process.kill()
                return "", f"Execution timed out after {self.timeout} seconds", False
                
        except Exception as e:
            return "", str(e), False
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    # Test
    runner = PythonExecutor()
    out, err, ok = runner.execute("print('Hello from Sandbox')")
    print(f"STDOUT: {out}")
    print(f"STDERR: {err}")

