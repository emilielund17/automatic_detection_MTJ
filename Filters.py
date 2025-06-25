import numpy as np
from scipy.signal import convolve2d
from multiprocessing.pool import ThreadPool, Pool

def generate_sticks_filters(n: int):

    # Initialize filters and filter values
    s = np.zeros((n, n, 2 * n - 2))
    s0 = 1/n
    m = n // 2
    l = 1

    # Generate filters
    # Horizontal and diagonal filters
    s[m,:,0] = s0
    s[:,m, 1] = s0
    np.fill_diagonal(s[:, :, 2], s0)
    np.fill_diagonal(np.fliplr(s[:, :, 3]), s0)

    # Mirror filters to create the remaining filters
    angles = np.linspace(0, np.pi / 4, int((2*n-6)/4+2))[1:-1]

    for i, angle in enumerate(angles):
        i = (i + 1)*4

        x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
        matrix = np.abs((x - m) * np.tan(angle) - (y - m))

        flat = matrix.flatten()  # Flatten matrix into a 1D array
        threshold = np.partition(flat, n-1)[n-1]  # Get the nth smallest value
        mask = matrix <= threshold  # Mask for values that are among the n smallest
        result = np.zeros_like(matrix)  # Initialize output matrix with zeros
        result[mask] = s0  # Assign 1/n to the n smallest values
        s[:, :, i] = result

        s[:, :, i+1] = np.fliplr(s[:, :, i])
        s[:, :, i+2] = np.rot90(s[:, :, i], -1)
        s[:, :, i+3] = np.fliplr(np.rot90(s[:, :, i], -1))


    return s

def apply_sticks_filter(image: np.ndarray, n: int, k: int):
    
    def apply_filter(args):
        image, sticks = args
        return convolve2d(image, sticks, mode='same')

    S = generate_sticks_filters(n)
    m = 2 * n - 2  # Number of filters
    args = [(image, S[:, :, i]) for i in range(m)]

    with ThreadPool(8) as thpool:
        filtered_images = thpool.map(apply_filter, args)
        # filtered_images = [convolve2d(image, S[:, :, i], mode='same') for i in range(m)]
    
    filtered_images = np.array(filtered_images)
    new_frame = np.max(filtered_images, axis=0).astype(np.uint8)
    
    return new_frame