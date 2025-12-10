"""
Safe Python REPL Executor.
Wraps execution in a separate process with time limits.
"""
import sys
import io
import contextlib
import multiprocessing
import traceback
import queue

def _execute_script(script: str, result_queue: multiprocessing.Queue):
    # Capture stdout/stderr
    capture_out = io.StringIO()
    capture_err = io.StringIO()
    
    # Safe globals
    safe_globals = {
        "print": print,
        "range": range,
        "len": len,
        "int": int,
        "float": float,
        "str": str,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "bool": bool,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
        "enumerate": enumerate,
        "zip": zip,
        "sorted": sorted,
        "reversed": sorted,
        "map": map,
        "filter": filter,
        "dir": dir,
        "help": help,
        "__name__": "__main__",
    }
    
    # Allow imports of standard math libraries
    import math
    import cmath
    import random
    import itertools
    import collections
    import numpy
    import sympy
    import networkx
    import re
    
    safe_globals.update({
        "math": math,
        "cmath": cmath,
        "random": random,
        "itertools": itertools,
        "collections": collections,
        "np": numpy,
        "numpy": numpy,
        "sympy": sympy,
        "nx": networkx,
        "re": re,
    })

    success = False
    output = ""
    error = ""
    
    try:
        with contextlib.redirect_stdout(capture_out), contextlib.redirect_stderr(capture_err):
            exec(script, safe_globals)
        success = True
        output = capture_out.getvalue()
        error = capture_err.getvalue()
    except Exception:
        success = False
        output = capture_out.getvalue()
        error = traceback.format_exc()
        
    result_queue.put({"success": success, "output": output, "error": error})

def run_python_code(code: str, timeout: float = 30.0) -> str:
    """
    Run python code in a separate process.
    Returns string output (stdout + stderr or error trace).
    """
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_execute_script, args=(code, q))
    p.start()
    p.join(timeout)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return "[Error] Execution timed out (limit: 30s)."
        
    if q.empty():
        return "[Error] Process crashed or returned no status."
        
    result = q.get()
    
    out = result["output"].strip()
    err = result["error"].strip()
    
    response = ""
    if out:
        response += f"Standard Output:\n{out}\n"
    if err:
        response += f"Error/Stderr:\n{err}\n"
        
    if not response and result["success"]:
        response = "[Success] Code executed with no output."
        
    return response.strip()

