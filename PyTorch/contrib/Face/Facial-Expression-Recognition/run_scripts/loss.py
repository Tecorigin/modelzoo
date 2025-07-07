import os, json, ast
import matplotlib.pyplot as plt

# Determine directories
this_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(this_dir)
log_dir = os.path.join(project_root, 'logs')
print("Looking for logs in:", log_dir)
if not os.path.isdir(log_dir): raise FileNotFoundError(f"Logs directory not found: {log_dir}")
files = os.listdir(log_dir)
print("Available files:", files)

# Helper: print sample lines for inspection

def print_sample(log_path, num=5):
    print(f"\nSample lines from {os.path.basename(log_path)}:")
    with open(log_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num: break
            print(line.strip())

# Paths
sdaa_log = os.path.join(log_dir, 'sdaa_loss.json')
cuda_log = os.path.join(log_dir, 'cuda_loss.json')

# Print samples
try:
    print_sample(sdaa_log)
    print_sample(cuda_log)
except Exception as e:
    print("Error reading sample:", e)

# Load losses: strip prefix, parse nested data string

def load_loss(log_path, key='train.loss'):
    losses = []
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            # strip any prefix before JSON
            try:
                json_part = line[line.index('{'):]
                entry = json.loads(json_part)
            except Exception:
                continue
            # entry.data may be a JSON string with single quotes
            raw = entry.get('data')
            if isinstance(raw, str):
                try:
                    data = ast.literal_eval(raw)
                except Exception:
                    continue
            elif isinstance(raw, dict):
                data = raw
            else:
                continue
            # extract loss
            if key in data:
                try:
                    losses.append(float(data[key]))
                except Exception:
                    continue
    return losses

# Load actual losses
sdaa_losses = load_loss(sdaa_log)
cuda_losses = load_loss(cuda_log)
print(f"Loaded {len(sdaa_losses)} SDAA loss entries, {len(cuda_losses)} CUDA loss entries.")
if not sdaa_losses or not cuda_losses:
    print("No losses loaded. Please check sample above for correct JSON key or structure.")
    exit(1)

# Plot
plt.figure(figsize=(10,6))
plt.plot(sdaa_losses, label='SDAA')
plt.plot(cuda_losses, label='CUDA')
plt.title('Loss Comparison')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_comparison.png')
plt.show()
