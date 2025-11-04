import re
import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(file_path):
    """Parse the log file and extract losses grouped by training step."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            content = f.read()

    # Ensure each epoch entry starts on a new line
    content = re.sub(r'Epoch:', '\nEpoch:', content)

    # Extract all epoch, step, and loss entries
    pattern = r'Epoch: \[(\d+)\]\[(\d+)/(\d+)\].*?Loss (\d+\.\d+)'
    matches = re.findall(pattern, content)

    # Group losses by step and determine steps per epoch
    step_losses = {}
    steps_per_epoch = None

    for match in matches:
        epoch = int(match[0])
        step = int(match[1])
        total_steps = int(match[2])
        loss = float(match[3])

        if steps_per_epoch is None:
            steps_per_epoch = total_steps

        key = (epoch, step)
        if key not in step_losses:
            step_losses[key] = []
        step_losses[key].append(loss)

    return step_losses, steps_per_epoch


def plot_average_losses(step_losses, output_file, steps_per_epoch):
    """Plot the average loss for each training step."""
    steps = []
    avg_losses = []

    for key, losses in sorted(step_losses.items()):
        epoch, step = key
        overall_step = step + epoch * steps_per_epoch
        steps.append(overall_step)
        avg_losses.append(sum(losses) / len(losses))

    plt.figure(figsize=(10, 6))
    plt.plot(steps, avg_losses)
    plt.title('Training Loss Curve')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()


# Main execution
if __name__ == "__main__":
    log_file = 'train_sdaa_3rd.log'
    output_file = 'train_sdaa_3rd.png'

    step_losses, steps_per_epoch = parse_log_file(log_file)
    if steps_per_epoch is None:
        steps_per_epoch = 1252  # Default if not found in log

    plot_average_losses(step_losses, output_file, steps_per_epoch)