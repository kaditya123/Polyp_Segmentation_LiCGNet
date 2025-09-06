import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_prediction(image, ground_truth_mask, predicted_mask, image_name):
    """
    Function to visualize the image, ground truth mask, and predicted mask.
    Args:
    - image: Input image
    - ground_truth_mask: Ground truth segmentation mask
    - predicted_mask: Predicted segmentation mask
    """
    # Minimal, robust display
    image = np.array(image)
    gt = np.array(ground_truth_mask)
    pred = np.array(predicted_mask)

    # Normalize image to uint8 [0,255]
    if image.dtype != np.uint8:
        im = image.astype(np.float32)
        im = (im - im.min()) / max(1e-6, (im.max() - im.min()))
        image_disp = (im * 255).astype(np.uint8)
    else:
        image_disp = image

    # If channels-first, convert to HWC
    if image_disp.ndim == 3 and image_disp.shape[0] in (1, 3) and image_disp.shape[0] != image_disp.shape[2]:
        image_disp = np.transpose(image_disp, (1, 2, 0))

    if image_disp.ndim == 2:
        image_disp = cv2.cvtColor(image_disp, cv2.COLOR_GRAY2RGB)

    # Prepare binary masks (0 or 255)
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    if pred.ndim == 3:
        pred = pred[:, :, 0]
    gt_mask = (gt > (0 if gt.max() <= 1 else (gt.max() / 2))).astype(np.uint8) * 255
    pred_mask = (pred > (0 if pred.max() <= 1 else (pred.max() / 2))).astype(np.uint8) * 255

    H, W = image_disp.shape[:2]
    if gt_mask.shape != (H, W):
        gt_mask = cv2.resize(gt_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    if pred_mask.shape != (H, W):
        pred_mask = cv2.resize(pred_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # Overlay prediction in red
    overlay = image_disp.copy()
    red = np.array([255, 0, 0], dtype=np.uint8)
    idx = pred_mask > 127
    overlay[idx] = (overlay[idx].astype(np.float32) * 0.5 + red.astype(np.float32) * 0.5).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(image_disp); axes[0].set_title('Image'); axes[0].axis('off')
    axes[1].imshow(gt_mask, cmap='gray'); axes[1].set_title('GT'); axes[1].axis('off')
    axes[2].imshow(pred_mask, cmap='gray'); axes[2].set_title('Pred'); axes[2].axis('off')
    axes[3].imshow(overlay); axes[3].set_title('Overlay'); axes[3].axis('off')
    plt.tight_layout(); plt.show()
    # plt.close(fig)
    # gc.collect()