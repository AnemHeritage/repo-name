# Import library
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('TkAgg')  # Backend interaktif

# Folder berisi gambar
image_folder = 'dataset_images'
output_folder = 'output_images'

# Buat folder output jika belum ada
os.makedirs(output_folder, exist_ok=True)

# Ambil semua file gambar dalam folder
image_files = [f for f in os.listdir('D:\\Project_Skripsi\\dataset_images') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if len(image_files) == 0:
    print('Tidak ada gambar dalam folder.')
    exit()

# Fungsi untuk menghitung PSNR
def calculate_psnr(original, processed):
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Fungsi untuk deteksi tepi dan menampilkan hasil
def detect_edges(image_path):
    # Membaca gambar berwarna
    image_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_color is None:
        print(f'Gagal memuat gambar: {image_path}')
        return None, None, None, None, None

    # Konversi gambar ke grayscale
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    # Deteksi tepi menggunakan Sobel
    sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)

    # Deteksi tepi menggunakan Canny
    canny_edges = cv2.Canny(image_gray, 100, 200)

    # Hitung PSNR
    psnr_sobel = calculate_psnr(image_gray, sobel_combined)
    psnr_canny = calculate_psnr(image_gray, canny_edges)

    print(f'Processing: {os.path.basename(image_path)}')
    print(f'PSNR (Sobel): {psnr_sobel:.2f} dB')
    print(f'PSNR (Canny): {psnr_canny:.2f} dB')

    return image_color, image_gray, sobel_combined, canny_edges, (psnr_sobel, psnr_canny)

# Proses hanya 10 gambar pertama
for idx, img_file in enumerate(image_files[:10], start=1):  # Ambil 10 gambar pertama
    img_path = os.path.join(image_folder, img_file)
    print(f'[{idx}/10] Processing: {img_file}')
    
    # Panggil fungsi detect_edges
    image_color, image_gray, sobel_combined, canny_edges, psnr_values = detect_edges(img_path)
    if image_color is None:
        continue

    psnr_sobel, psnr_canny = psnr_values

    # Buat figure untuk setiap gambar
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))  # Konversi BGR ke RGB untuk Matplotlib
    plt.title(f'Original (Color)\n{os.path.basename(img_file)}')

    plt.subplot(1, 4, 2)
    plt.imshow(image_gray, cmap='gray')
    plt.title('Grayscale')

    plt.subplot(1, 4, 3)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title(f'Sobel\nPSNR: {psnr_sobel:.2f} dB')

    plt.subplot(1, 4, 4)
    plt.imshow(canny_edges, cmap='gray')
    plt.title(f'Canny\nPSNR: {psnr_canny:.2f} dB')

    # Simpan gambar ke folder output_images
    output_path = os.path.join(output_folder, f'{os.path.splitext(img_file)[0]}_comparison.png')
    plt.tight_layout()
    plt.savefig(output_path)  # Simpan gambar ke file
    print(f'Hasil disimpan di {output_path}')
    plt.close()

