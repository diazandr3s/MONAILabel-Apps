"""
@author: MONAI 
(https://github.com/Project-MONAI/tutorials/blob/master/modules/dynunet_pipeline/create_network.py)
"""

def dynUNet_strides_kernels(patch_size, target_spacing):
    sizes, spacings = patch_size, target_spacing
    strides, kernels = [], []
    
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [
            2 if ratio <= 2 and size >= 8 else 1
            for (ratio, size) in zip(spacing_ratio, sizes)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return strides, kernels

