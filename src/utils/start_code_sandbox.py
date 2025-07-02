from fastapi import FastAPI
import contextlib
import io
import signal
from typing import Dict
from pydantic import BaseModel
from argparse import ArgumentParser
import builtins
import sys
import threading
import os

app = FastAPI()

class CodeRequest(BaseModel):
    env: str = None
    call: str
    timeout: int = 5

def timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")

class RestrictedImportHook:
    def __init__(self, blacklist=None):
        self.blacklist = blacklist or [
            'os', 'subprocess', 'sys', 'shutil', 'socket', 'pathlib', 
            'tempfile', 'importlib', 'ctypes', 'multiprocessing'
        ]
    
    def find_spec(self, fullname, path, target=None):
        if fullname in self.blacklist or any(fullname.startswith(f"{module}.") for module in self.blacklist):
            raise ImportError(f"Import of '{fullname}' is not allowed for security reasons")
        return None

class ThreadWithException(threading.Thread):
    def __init__(self, target=None, args=()):
        threading.Thread.__init__(self, target=target, args=args)
        self.exception = None
        
    def run(self):
        try:
            if self._target:
                self._target(*self._args)
        except Exception as e:
            self.exception = e
            
    def get_exception(self):
        return self.exception

def execute_with_timeout(call_code, timeout_seconds=30):
    result = {"output": "", "result": None, "error": None}
    output = io.StringIO()
    
    def execute():
        try:
            with contextlib.redirect_stdout(output):
                exec_env = {"__builtins__": restricted_builtins}
                exec(compile(call_code, '<call>', 'exec'), exec_env)
        except Exception as e:
            print("error", e)
            result["error"] = str(e)
    
    # Create thread for execution
    thread = ThreadWithException(target=execute)
    thread.start()
    thread.join(timeout_seconds)
    
    # Check if thread is still alive (timeout occurred)
    if thread.is_alive():
        thread._stop()
        result["error"] = "Code execution timed out"
    elif thread.get_exception():
        result["error"] = str(thread.get_exception())
    
    result["output"] = output.getvalue()
    result["result"] = output.getvalue() if result["error"] is None else str(result['error'])
    return result

# Create restricted builtins
restricted_builtins = {
    k: v for k, v in builtins.__dict__.items() 
    if k not in ['exec', 'eval', 'compile', '__import__', 'input', 
                 'memoryview', 'staticmethod', 'classmethod', 'globals', 'locals']
}

# Add safe functions back
restricted_builtins['print'] = print
restricted_builtins['dir'] = dir
restricted_builtins['len'] = len
restricted_builtins['range'] = range
restricted_builtins['int'] = int
restricted_builtins['float'] = float
restricted_builtins['str'] = str
restricted_builtins['list'] = list
restricted_builtins['dict'] = dict
restricted_builtins['set'] = set
restricted_builtins['tuple'] = tuple
restricted_builtins['sum'] = sum
restricted_builtins['min'] = min
restricted_builtins['max'] = max
restricted_builtins['enumerate'] = enumerate
restricted_builtins['zip'] = zip
restricted_builtins['isinstance'] = isinstance
restricted_builtins['issubclass'] = issubclass
restricted_builtins['hasattr'] = hasattr
restricted_builtins['getattr'] = getattr
restricted_builtins['setattr'] = setattr
restricted_builtins['delattr'] = delattr
restricted_builtins['sorted'] = sorted
restricted_builtins['reversed'] = reversed
restricted_builtins['round'] = round
restricted_builtins['abs'] = abs
restricted_builtins['all'] = all
restricted_builtins['any'] = any
restricted_builtins['bool'] = bool
restricted_builtins['chr'] = chr
restricted_builtins['divmod'] = divmod
restricted_builtins['filter'] = filter
restricted_builtins['format'] = format
restricted_builtins['hex'] = hex
restricted_builtins['id'] = id
restricted_builtins['iter'] = iter
restricted_builtins['map'] = map
restricted_builtins['next'] = next
restricted_builtins['object'] = object
restricted_builtins['oct'] = oct
restricted_builtins['ord'] = ord
restricted_builtins['pow'] = pow
restricted_builtins['repr'] = repr
restricted_builtins['slice'] = slice
restricted_builtins['type'] = type

# Add safe types
for name in dir(builtins):
    if name.startswith('__') and name.endswith('__'):
        restricted_builtins[name] = getattr(builtins, name)

@app.post("/execute")
async def execute_code(request: CodeRequest) -> Dict:
    print("-"*30)
    print(request.call)
    print("-"*30)
    
    # Add import hook to prevent dangerous imports
    #sys.meta_path.insert(0, RestrictedImportHook())
    
    try:
        # Execute code with timeout
        result = execute_with_timeout(request.call, request.timeout)
        return result
    except Exception as e:
        return {
            "output": "",
            "result": "",
            "error": f"Execution error: {str(e)}"
        }
    finally:
        # Remove the import hook
        sys.meta_path = [hook for hook in sys.meta_path if not isinstance(hook, RestrictedImportHook)]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--port", type=int, default=1000)
    args = parser.parse_args()

    # Ensure sandbox is running with limited privileges
    if os.geteuid() == 0:  # Running as root is dangerous
        print("Warning: Running sandbox as root is not recommended")
    
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=args.port)