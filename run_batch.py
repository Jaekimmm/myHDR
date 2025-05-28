import pandas as pd
import subprocess
import sys
import os
import time
import threading
import signal

EXCEL_PATH = "batch_list.xlsx"

processes = []
summaries = []
running = True
run_queue = []
lock = threading.Lock()

def handle_sigint(signum, frame):
    global running
    print("\n[KILL] terminating all processes...")
    with lock:
        for p in processes:
            if p['process'] is not None and p['status'] == 'running':
                p['process'].kill()
                    
    running = False
signal.signal(signal.SIGINT, handle_sigint)

def allocate_gpus(num_required, running_processes):
    used = set()
    for p in running_processes:
        if p['status'] == 'running' and p.get('gpu_ids'):
            used.update(p['gpu_ids'].split(","))
    available = [g for g in AVAILABLE_GPUS if g not in used]
    return available[:num_required] if len(available) >= num_required else None

def input_listener():
    global running
    while running:
        cmd = input().strip().lower()
        if cmd == 'status':
            max_name_len = max(len(p['case_name']) for p in processes)
            max_gpu_len = max(len(str(p['gpu_ids'] or '')) for p in processes)
            max_status_len = max(len(p['status']) for p in processes)
            print(
                f"{'Run Name'.ljust(max_name_len)}  "
                f"{'GPU'.ljust(max_gpu_len)}  "
                f"{'STATUS'.ljust(max_status_len)}  "
                f"Log Tail"
            )
            print("-" * (max_name_len + max_gpu_len + max_status_len + 20))
            
            for p in processes:
                if p['status'] == 'running':
                    retcode = p['process'].poll()
                    if retcode is not None:
                        p['status'] = 'finish' if retcode == 0 else 'fail'
                        p['end_time'] = time.time()
                        elapsed_sec = int(p['end_time'] - p['start_time'])
                        elapsed = f"{elapsed_sec // 60}m {elapsed_sec % 60}s"
                        try:
                            if os.path.exists(p['log_file']):
                                with open(p['log_file'], 'rb') as lf:
                                    lf.seek(-min(1024, os.path.getsize(p['log_file'])), os.SEEK_END)
                                    p['log_tail'] = lf.read().decode(errors='ignore').splitlines()[-1]
                        except Exception:
                            p['log_tail'] = ''
                        summaries.append({
                            'Run name': p['case_name'],
                            'Status': 'Success' if retcode == 0 else 'Fail',
                            'Elapsed (sec)': elapsed,
                            'GPU': p['gpu_ids'],
                            'log tail': p['log_tail']
                        })
                        p['file_handle'].close()
                        
                p_status = p['status']
                color = {'running': '\033[33m', 'finish': '\033[32m', 'fail': '\033[31m'}.get(p_status, '\033[0m')
                reset_color = '\033[0m'
                try:
                    if os.path.exists(p['log_file']):
                        with open(p['log_file'], 'rb') as lf:
                            lf.seek(-min(1024, os.path.getsize(p['log_file'])), os.SEEK_END)
                            p['log_tail'] = lf.read().decode(errors='ignore').splitlines()[-1]
                except Exception:
                    p['log_tail'] = ''

                print(
                    f"{p['case_name'].ljust(max_name_len)}  "
                    f"{str(p['gpu_ids'] or '-').ljust(max_gpu_len)}  "
                    f"{color}{p_status.ljust(max_status_len)}{reset_color}  "
                    f"{p['log_tail'] if p_status.lower() != 'ready' else ''}"
                )
            print()
            print(">> ", end="")
            
        elif cmd == 'rerun fail':
            failed_cases = [p for p in processes if p['status'].lower() == 'fail']
            if not failed_cases:
                print("No failed processes to rerun.")
                continue
            
            print(f"[INFO] Retrying {len(failed_cases)} failed cases...")
            
            for p in failed_cases:
                f = open(p['log_file'], "w")
                p.update({
                    'process': None,
                    'status': 'ready',
                    'start_time': None,
                    'file_handle': f,
                    'gpu_ids': None,
                    'log_tail': ''
                })
                run_queue.insert(0, p)
                print(f"[RERUN] {p['case_name']} has been added to the queue.")
            print()
            print(">> ", end="")
        
        else:
            print()
            print(">> ", end="")
            
        
def parse_gpu_list(gpu_arg):
    gpu_arg = gpu_arg.strip()
    if '~' in gpu_arg:
        start, end = map(int, gpu_arg.split('~'))
        return [str(i) for i in range(start, end + 1)]
    return gpu_arg.split(',')

def get_available_gpus(max_usage_ratio=0.3):
    try:
        # GPU index, memory.total [MiB], memory.used [MiB] 가져오기
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.total,memory.used', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        available = []
        for line in result.stdout.strip().split('\n'):
            idx, total, used = map(int, line.strip().split(','))
            usage_ratio = used / total
            if usage_ratio <= max_usage_ratio:
                available.append(str(idx))

        return available

    except Exception as e:
        print(f"[ERROR] Failed to detect available GPUs: {e}")
        sys.exit(1)

def main():
    global running, SHEET_NAME, AVAILABLE_GPUS

    if len(sys.argv) < 2:
        print("Usage: python run_train_batch.py <sheet_name> [avaiable_gpus]")
        sys.exit(1)

    SHEET_NAME = sys.argv[1]
    
    if len(sys.argv) >= 3:
        AVAILABLE_GPUS = parse_gpu_list(sys.argv[2])
    else:
        AVAILABLE_GPUS = get_available_gpus(max_usage_ratio=0.3)
    print(f"[INFO] Available GPUs: {', '.join(AVAILABLE_GPUS)}")

    if not os.path.exists(EXCEL_PATH):
        print(f"Error: File '{EXCEL_PATH}' not found.")
        sys.exit(1)

    os.makedirs(os.path.dirname('./log'), exist_ok=True)

    df_full = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, header=None)
    print(f"[INFO] Loaded sheet '{SHEET_NAME}' from '{EXCEL_PATH}'")
    run_script = str(df_full.iloc[0, 0]).strip()
    enable_col = df_full.iloc[2:, 0]
    args_list = df_full.iloc[1, 1:].tolist()
    df = df_full.iloc[2:, 1:]
    print(f"[INFO] Loaded {len(df)} cases with {len(df.columns)} args")
    df.columns = args_list
    print(f"[INFO] Ready to parse runs...")

    for row_idx, (row_name, col) in enumerate(df.iterrows()):
        enable_flag = str(enable_col.iloc[row_idx]).strip().lower()
        if enable_flag not in ['o', '1']:
            continue

        cmd = f"python {run_script} "
        try:
            num_gpu = int(col.get('num_gpu', 1)) if not pd.isna(col.get('num_gpu', 1)) else 1
        except:
            num_gpu = 1
        
        for arg_name, val in col.items():
            if not isinstance(arg_name, str) or pd.isna(arg_name) or arg_name.strip() == '':
                continue
            if arg_name == 'num_gpu':
                continue
            if pd.isna(val):
                continue

            val_str = str(val).strip().lower()
            if val_str == 'none' or val_str == 'false':
                continue
            elif val_str == 'true':
                cmd += f"--{arg_name} "
            else:
                cmd += f"--{arg_name} '{str(val).strip()}' "

        train_name = col.get('train_name', row_name).replace(" ", "_")
        model_name = col.get('model', 'unknown').replace(" ", "_")
        test_name = col.get('test_name')
        case_name = f"{model_name}-{train_name}"
        
        if not pd.isna(test_name):
            case_name = f"{model_name}-{train_name}-{test_name}"
        
        log_file = f"./logs/{SHEET_NAME}-{case_name}.log"

        p = {
            'process': None,
            'case_name': case_name,
            'num_gpu': num_gpu,
            'gpu_ids': None,
            'cmd': cmd,
            'log_file': log_file,
            'start_time': None,
            'file_handle': None,
            'status': 'ready',
            'log_tail': ''
        }
        processes.append(p)
        run_queue.append(p)
    
    print(f"[INFO] All {len(processes)} batch processes has been added to the queue... \n>> ", end="")

    listener_thread = threading.Thread(target=input_listener, daemon=True)
    listener_thread.start()

    # scheduling loop
    while running and any (p['status'] in ['ready', 'running'] for p in processes):
        with lock:
            for p in run_queue[:]:
                if p['status'] != 'ready':
                    continue
                allocated = allocate_gpus(p['num_gpu'], processes)
                if allocated:
                    p['gpu_ids'] = ','.join(allocated)
                    f = open(p['log_file'], "w")
                    env_cmd = f"\nCUDA_VISIBLE_DEVICES={p['gpu_ids']} {p['cmd']}"
                    print(env_cmd)
                    proc = subprocess.Popen(env_cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
                    p.update({
                        'process': proc,
                        'status': 'running',
                        'start_time': time.time(),
                        'file_handle': f
                    })
                    run_queue.remove(p)

            for p in processes:
                if p['status'] != 'running':
                    continue
                retcode = p['process'].poll()
                if retcode is not None:
                    p['status'] = 'finish' if retcode == 0 else 'fail'
                    p['end_time'] = time.time()
                    elapsed_sec = int(p['end_time'] - p['start_time'])
                    elapsed = f"{elapsed_sec // 60}m {elapsed_sec % 60}s"
                    try:
                        if os.path.exists(p['log_file']):
                            with open(p['log_file'], 'rb') as lf:
                                lf.seek(-min(1024, os.path.getsize(p['log_file'])), os.SEEK_END)
                                p['log_tail'] = lf.read().decode(errors='ignore').splitlines()[-1]
                    except Exception as e:
                        p['log_tail'] = f"(Error reading log: {e})"
                    p['file_handle'].close()
                    summaries.append({
                        'Run name': p['case_name'],
                        'Status': 'Success' if retcode == 0 else 'Fail',
                        'Elapsed (sec)': elapsed,
                        'GPU': p['gpu_ids'],
                        'log tail': p['log_tail']
                    })
        time.sleep(1)

    running = False
    
    max_name_len = max(len(s['Run name']) for s in summaries)
    max_status_len = max(len(s['Status']) for s in summaries)

    print("\n=== Execution Summary ===")
    print(
        f"{'Run name'.ljust(max_name_len)}  "
        f"{'Status'.ljust(max_status_len)}  "
        f"{'Elapsed (sec)':<15}  "
        f"log tail")
    print("-" * (max_name_len + max_status_len + 20 + 15))
    for s in summaries:
        print(
            f"{s['Run name'].ljust(max_name_len)}  "
            f"{s['Status'].ljust(max_status_len)}  "
            f"{s['Elapsed (sec)']:<15}  "
            f"{s['log tail']}")
    print("=" * (max_name_len + max_status_len + 20 + 20))

if __name__ == "__main__":
    main()