# kernel, padding
def autopad(kernal_size, padding=None):
    # Pad to 'same'
    if padding is None:
        padding = kernal_size//2 if isinstance(kernal_size, int) else [x//2 for x in kernal_size]
    return padding