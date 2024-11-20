import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# MNISTデータセットの読み込み
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 塩胡椒ノイズを加える関数
def add_salt_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    total_pixels = image.size
    
    # 塩ノイズ（白）を加える
    salt = int(total_pixels * salt_prob)
    salt_coords = [np.random.randint(0, i - 1, salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255
    
    # 胡椒ノイズ（黒）を加える
    pepper = int(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i - 1, pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0
    
    return noisy_image

# ノイズの密度
salt_prob = 0.02  # 塩ノイズの確率
pepper_prob = 0.02  # 胡椒ノイズの確率

# 最初の画像にノイズを加える例
noisy_image = add_salt_pepper_noise(x_train[0], salt_prob, pepper_prob)

# 結果を表示
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# オリジナル画像
ax[0].imshow(x_train[0], cmap='gray')
ax[0].set_title("Original Image")
ax[0].axis('off')

# ノイズ付き画像
ax[1].imshow(noisy_image, cmap='gray')
ax[1].set_title("Image with Salt & Pepper Noise")
ax[1].axis('off')

plt.show()
