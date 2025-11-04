import re
import numpy as np
import matplotlib.pyplot as plt

# Read the log file
try:
    with open('train_sdaa_3rd.log', 'r') as f:
        content = f.read()
except FileNotFoundError:
    print("Error: The file 'train_sdaa_3rd.log' was not found.")
    exit(1)

# Extract step and average loss information using regex
pattern = r'Epoch: \[\d+/\d+\]\[(\d+)/\d+\].*?Loss [0-9.]+? \(([0-9.]+)\)'
matches = re.findall(pattern, content)

# Organize data by step
data = {}
for step_str, loss_str in matches:
    step = int(step_str)
    try:
        loss = float(loss_str)
        if step not in data:
            data[step] = []
        data[step].append(loss)
    except ValueError:
        print(f"Warning: Could not convert '{loss_str}' to float for step {step}")

# Calculate average loss for each step (average of the 4 values per step)
steps = []
avg_losses = []
for step, losses in sorted(data.items()):
    if len(losses) == 4:  # Each step should have 4 values
        steps.append(step)
        avg_losses.append(np.mean(losses))
    else:
        print(f"Warning: Step {step} has {len(losses)} values instead of 4")

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(steps, avg_losses, marker='o', linestyle='-',ms=1)
plt.title('Training Loss Curve')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig('train_sdaa_3rd.png')
plt.close()

print(f"Plot saved as train_sdaa_3rd.png with {len(steps)} data points")