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

# Ağırlıkları INT8 olarak kaydet + bias export
bias_list = []
for i, layer in enumerate(model.layers):
    weights = layer.get_weights()
    if weights:
        w = weights[0]  # shape: (input, output)
        b = weights[1]  # bias
        scale = np.max(np.abs(w))
        # Transpose: TF gives (input, output), C needs (output, input)
        w_T = w.T
        w_int8 = np.clip(np.round(w_T / scale * 127), -128, 127).astype(np.int8)
        np.save(f'layer{i}_weights.npy', w_int8)
        # Bias: scale to match quantized output domain (bias_float * 127)
        bias_int32 = np.round(b * 127).astype(np.int32)
        bias_list.append((i, bias_int32))
        print(f'Layer {i}: shape={w_int8.shape}, scale={scale:.4f}, bias_range=[{b.min():.4f}, {b.max():.4f}]')


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



# 100 test görüntüsü
imgs = x_test[:100].reshape(100, -1)
imgs_int8 = np.clip(np.round(imgs * 127), -128, 127).astype(np.int8)
labels = y_test[:100]

with open('test_batch.h', 'w') as f:
    f.write('#pragma once\n#include <stdint.h>\n\n')
    f.write('#define TEST_COUNT 100\n\n')
    f.write('static const int8_t test_batch[] = {\n  ')
    f.write(', '.join(str(x) for x in imgs_int8.flatten()))
    f.write('\n};\n\n')
    f.write('static const uint8_t test_labels[] = {\n  ')
    f.write(', '.join(str(x) for x in labels))
    f.write('\n};\n')

print('test_batch.h yazıldı')

# Bias header dosyasını oluştur
with open('bias.h', 'w') as f:
    f.write('#pragma once\n#include <stdint.h>\n\n')
    for idx, (layer_idx, bias) in enumerate(bias_list):
        f.write(f'static const int32_t layer{idx+1}_bias[] = {{\n  ')
        f.write(', '.join(str(x) for x in bias))
        f.write('\n};\n\n')
print('bias.h yazıldı')

# Scale değerlerini kaydet
scales = []
for i, layer in enumerate(model.layers):
    weights = layer.get_weights()
    if weights:
        w = weights[0]
        scale = np.max(np.abs(w))
        scales.append(scale)

with open('scales.h', 'w') as f:
    f.write('#pragma once\n#include <stdint.h>\n\n')
    for i, scale in enumerate(scales):
        scale_fixed = int((scale / 127) * 65536)  # 16-bit shift
        f.write(f'#define LAYER{i+1}_SCALE_FIXED {scale_fixed}\n')

print('scales.h yazıldı:', scales)