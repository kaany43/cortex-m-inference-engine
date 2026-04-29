import numpy as np
import tensorflow as tf

# Küçük MNIST modeli indir ve INT8 export et
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# MNIST veriyi yükle ve eğit
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
model.fit(x_train, y_train, epochs=3, verbose=1)

# Ağırlıkları INT8 olarak kaydet
for i, layer in enumerate(model.layers):
    weights = layer.get_weights()
    if weights:
        w = weights[0]
        scale = np.max(np.abs(w))
        w_int8 = np.clip(np.round(w / scale * 127), -128, 127).astype(np.int8)
        np.save(f'layer{i}_weights.npy', w_int8)
        print(f'Layer {i}: shape={w_int8.shape}, scale={scale:.4f}')


(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test / 255.0

# İlk test görüntüsünü al, düzleştir, INT8'e çevir
img = x_test[0].flatten()
img_int8 = np.clip(np.round(img * 127), -128, 127).astype(np.int8)

with open('test_image.h', 'w') as f:
    f.write('#pragma once\n#include <stdint.h>\n\n')
    f.write(f'// Label: {y_test[0]}\n')
    f.write(f'#define TEST_LABEL {y_test[0]}\n\n')
    f.write('static const int8_t test_image[] = {\n  ')
    f.write(', '.join(str(x) for x in img_int8))
    f.write('\n};\n')

print(f'Kaydedildi. Gerçek rakam: {y_test[0]}')