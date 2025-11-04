import torch
import torch.nn as nn
import torchvision
import os
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import glob
import net_selfattention


# Function for dehazing
def dehaze_image(image_path):
    # Open the input image
    data_hazy = Image.open(image_path)
    original_image = data_hazy.copy()  # Keep a copy of the original image

    # Convert the image to a numpy array and normalize
    data_hazy = (np.asarray(data_hazy) / 255.0)

    # Convert to tensor
    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 0, 1)  # Change to (C, H, W) format
    data_hazy = data_hazy.cuda().unsqueeze(0)  # Add batch dimension

    # Load the trained model
    dehaze_net = net.dehaze_net().cuda()
    dehaze_net.load_state_dict(torch.load('snapshots/dehazer.pth'))

    # Perform dehazing
    clean_image = dehaze_net(data_hazy)

    # Convert dehazed image to numpy array and then to PIL Image
    clean_image = clean_image.squeeze().cpu().detach().numpy()
    clean_image = np.transpose(clean_image, (1, 2, 0)) * 255.0
    clean_image = np.clip(clean_image, 0, 255).astype(np.uint8)
    clean_image_pil = Image.fromarray(clean_image)

    return original_image, clean_image_pil


# Function to load and dehaze image
def load_and_dehaze_image():
    # Open file dialog to choose an image
    image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])

    if image_path:
        # Perform dehazing and get both the original and dehazed images
        original_image, clean_image = dehaze_image(image_path)

        # Display the original image
        original_image_tk = ImageTk.PhotoImage(original_image)
        original_label.config(image=original_image_tk)
        original_label.image = original_image_tk

        # Display the dehazed image
        clean_image_tk = ImageTk.PhotoImage(clean_image)
        clean_label.config(image=clean_image_tk)
        clean_label.image = clean_image_tk

        # Save the dehazed image to results folder
        save_path = "results/" + os.path.basename(image_path)
        clean_image.save(save_path)  # Save the dehazed image


# Create the GUI window
root = tk.Tk()
root.title("Image Dehazing")

# Set up the window size
root.geometry("1200x600")  # Adjusted width to fit two images side by side

# Add a button to load and dehaze the image
load_button = tk.Button(root, text="Upload Image", command=load_and_dehaze_image)
load_button.pack(pady=20)

# Add a label to display the original image
original_label = tk.Label(root)
original_label.pack(side=tk.LEFT, padx=10)  # Display on the left

# Add a label to display the dehazed image
clean_label = tk.Label(root)
clean_label.pack(side=tk.LEFT, padx=10)  # Display on the right

# Start the GUI main loop
root.mainloop()
