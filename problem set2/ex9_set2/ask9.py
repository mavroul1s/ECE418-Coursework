import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

F1 = np.array([[ 0, -1,  0],
               [-1,  8, -1],
               [ 0, -1,  0]])

F2 = np.array([[ 0,  1,  0],
               [ 1,  4,  1],
               [ 0,  1,  0]])

F3 = np.array([[-1, -1, -1],
               [-1,  8, -1],
               [-1, -1, -1]])

filters = [F1, F2, F3]
filter_names = ["F1 (Edge Detection)", "F2 (Blurring)", "F3 (Sharp Edge)"]

def toeplitz_transform_matrix(input_shape, kernel):
    h_in, w_in = input_shape
    h_k, w_k = kernel.shape
    h_out, w_out = h_in - h_k + 1, w_in - w_k + 1
    
    matrix_rows = h_out * w_out
    matrix_cols = h_in * w_in
    
    H = np.zeros((matrix_rows, matrix_cols))
    
    for i in range(h_out):
        for j in range(w_out):
            row_idx = i * w_out + j
            for ki in range(h_k):
                for kj in range(w_k):
                    r_in = i + ki
                    c_in = j + kj
                    col_idx = r_in * w_in + c_in
                    H[row_idx, col_idx] = kernel[ki, kj]
    return H


img = cv2.imread('cat.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Warning: 'cat.png' not found. Using dummy noise image.")
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

results_conv = []
times_conv = []
for f in filters:
    t0 = time.time()
    res = convolve2d(img, f, mode='valid')
    t1 = time.time()
    results_conv.append(res)
    times_conv.append(t1 - t0)

img_small = cv2.resize(img, (20, 20))
times_matrix = []
for f in filters:
    t0 = time.time()
    H = toeplitz_transform_matrix(img_small.shape, f)
    img_vector = img_small.flatten()
    res_vector = H @ img_vector
    t1 = time.time()
    times_matrix.append(t1 - t0)

plt.figure(figsize=(6, 6))
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(filters[i], cmap='coolwarm')
    for (r, c), val in np.ndenumerate(filters[i]):
        plt.text(c, r, f'{val}', ha='center', va='center', color='black', fontsize=12, weight='bold')
    plt.title(filter_names[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(results_conv[i], cmap='gray')
    plt.title(f"Output of {filter_names[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

dummy_shape = (6, 6)
H_demo = toeplitz_transform_matrix(dummy_shape, F1) 

plt.figure(figsize=(8, 8))
plt.imshow(H_demo, cmap='Blues', interpolation='nearest')
plt.title("Visual of the Toeplitz Matrix Structure\n(Notice the Diagonal Bands)", fontsize=14)
plt.xlabel("Input Pixels (Flattened)", fontsize=12)
plt.ylabel("Output Pixels (Flattened)", fontsize=12)
plt.colorbar(label="Weight Value")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

scale_factor = (img.shape[0]*img.shape[1]) / (20*20)
estimated_matrix_times = [t * scale_factor for t in times_matrix]

x_pos = np.arange(len(filter_names))
width = 0.35

plt.bar(x_pos - width/2, times_conv, width, label='Convolution (Direct)', color='#2ca02c')
plt.bar(x_pos + width/2, estimated_matrix_times, width, label='Matrix Mult (Projected)', color='#d62728')

plt.ylabel('Time (Seconds) - Log Scale', fontsize=12)
plt.title('Computation Time: Convolution vs Matrix Multiplication', fontsize=14)
plt.xticks(x_pos, ['Filter 1', 'Filter 2', 'Filter 3'])
plt.legend(fontsize=12)
plt.yscale('log') # Log scale is essential here
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()