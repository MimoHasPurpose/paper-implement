import matplotlib.pyplot as plt
def test_img_print(image):
    # Step 1: Remove batch dimension
    image_tensor = image.squeeze(0)  # Shape: [3, 480, 480]
    
    # Step 2: Permute dimensions to [height, width, channels]
    image_tensor = image_tensor.permute(1, 2, 0)  # Shape: [480, 480, 3]
    
    # Step 3: Convert to NumPy
    image_np = image_tensor.numpy()
    
    # Step 4: Normalize if necessary (assuming values are not in [0, 1] or [0, 255])
    # If the tensor is already normalized (e.g., [0, 1]), skip this step
    # If the tensor is in a different range, adjust accordingly, e.g.:
    # image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    
    # Plot the image
    plt.imshow(image_np)
    plt.show()
    exit()