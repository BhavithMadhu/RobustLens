import numpy as np
import pandas as pd
import torch

def compute_color_focus_score(attribution, mask):
    if torch.is_tensor(attribution):
        attribution = attribution.squeeze().detach().cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.squeeze().detach().cpu().numpy()
        
    if attribution.ndim == 3:
        attribution = np.sum(np.abs(attribution), axis=0)
    else:
        attribution = np.abs(attribution)
        
    color_region = attribution[mask > 0]
    total_attribution = np.sum(attribution)
    
    if total_attribution == 0:
        return 0.0
    return np.sum(color_region) / total_attribution

def compute_shape_focus_score(attribution, original_grayscale):
    if torch.is_tensor(attribution):
        attribution = attribution.squeeze().detach().cpu().numpy()
    if torch.is_tensor(original_grayscale):
        original_grayscale = original_grayscale.squeeze().detach().cpu().numpy()
        
    if attribution.ndim == 3:
        attribution = np.sum(np.abs(attribution), axis=0)
    else:
        attribution = np.abs(attribution)
        
    shape_mask = original_grayscale > 0.5
    shape_region = attribution[shape_mask]
    total_attribution = np.sum(attribution)
    
    if total_attribution == 0:
        return 0.0
    return np.sum(shape_region) / total_attribution

def compute_background_focus_score(attribution, original_grayscale):
    if torch.is_tensor(attribution):
        attribution = attribution.squeeze().detach().cpu().numpy()
    if torch.is_tensor(original_grayscale):
        original_grayscale = original_grayscale.squeeze().detach().cpu().numpy()
        
    if attribution.ndim == 3:
        attribution = np.sum(np.abs(attribution), axis=0)
    else:
        attribution = np.abs(attribution)
        
    background_mask = original_grayscale <= 0.1
    bg_region = attribution[background_mask]
    total_attribution = np.sum(attribution)
    
    if total_attribution == 0:
        return 0.0
    return np.sum(bg_region) / total_attribution

def compare_focus_scores(biased_model, debiased_model, images, labels):
    from .integrated_gradients import IntegratedGradients
    
    device = images.device
    ig_biased = IntegratedGradients(biased_model, device)
    ig_debiased = IntegratedGradients(debiased_model, device)
    
    records = []
    
    for i in range(images.size(0)):
        img = images[i:i+1]
        label = labels[i].item()
        
        attr_b = ig_biased.compute(img, label)
        attr_d = ig_debiased.compute(img, label)
        
        if img.shape[1] == 3:
            gray = torch.mean(img, dim=1, keepdim=True)
            color_mask = torch.max(img, dim=1, keepdim=True)[0] - torch.min(img, dim=1, keepdim=True)[0] > 0.1
        else:
            gray = img
            color_mask = torch.zeros_like(img)
            
        records.append({
            'Model': 'Biased',
            'Image_ID': i,
            'Label': label,
            'Color_Focus': compute_color_focus_score(attr_b, color_mask),
            'Shape_Focus': compute_shape_focus_score(attr_b, gray),
            'Background_Focus': compute_background_focus_score(attr_b, gray)
        })
        
        records.append({
            'Model': 'Debiased',
            'Image_ID': i,
            'Label': label,
            'Color_Focus': compute_color_focus_score(attr_d, color_mask),
            'Shape_Focus': compute_shape_focus_score(attr_d, gray),
            'Background_Focus': compute_background_focus_score(attr_d, gray)
        })
        
    return pd.DataFrame(records)
