import matplotlib
import matplotlib.pyplot as plt
from skimage import color
import torch
import bins

matplotlib.use("Agg") 

# Show matrix image
def show_plot(matrix, axis, title=""):
    """Shows matrix image using matplotlib.

    Args:
        matrix (numpy.ndarray): A valid image matrix.
        axis (matplotlib.pyplot.axis): A matplotlib axis to add the image to.
        title (str, optional): Plot title. Defaults to "".
    """
    axis.imshow(matrix, cmap='gray')
    if title:
        axis.set_title(title)
    axis.axis('off')
    
def to_rgb(L, ab):
    lab = torch.cat([(L + 1) * 50, ab], dim=1)
    lab = lab.permute(0,2 , 3, 1).cpu().numpy()
    return color.lab2rgb(lab)

def save_samples(model, device, test_set, centers, path, n=6):
    model.eval()
    picks = [i * len(test_set) // n for i in range(n)]
    L = torch.stack([test_set[i][0] for i in picks]).to(device)
    ab = torch.stack([test_set[i][1] for i in picks]).to(device)
    
    with torch.no_grad():
        predicted_ab = bins.decode(model(L), centers)
        
    grayscale = to_rgb(L, torch.zeros_like(ab))
    predicted = to_rgb(L, predicted_ab)
    truth = to_rgb(L, ab)
    
    fig, axes = plt.subplots(n, 3, figsize=(6, 2 * n))
    for row in range(n):
        show_plot(grayscale[row], axes[row, 0], "Input" if row == 0 else "")
        show_plot(predicted[row], axes[row, 1], "Predicted" if row == 0 else "")
        show_plot(truth[row], axes[row, 2], "Truth" if row == 0 else "")
    
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)