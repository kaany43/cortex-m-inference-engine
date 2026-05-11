import numpy as np
import tensorflow as tf
import os

OUT_DIR = os.path.join('model', 'headers')
os.makedirs(OUT_DIR, exist_ok=True)

# Train the CNN model (similar to train_conv.py)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[..., np.newaxis] / 255.0
x_test = x_test[..., np.newaxis] / 255.0

model = tf.keras.models.Sequential([
    # use_bias=False ekliyoruz çünkü C kodundaki conv2d fonksiyonu bias almıyor.
    tf.keras.layers.Conv2D(1, (3,3), activation='relu', input_shape=(28,28,1), use_bias=False),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, verbose=1)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f'Test accuracy: {acc:.4f}')

# Ağırlıkları INT8 olarak kaydet + bias export
bias_list = []
scales = []
header_includes = []

dense_idx = 1

for i, layer in enumerate(model.layers):
    weights = layer.get_weights()
    if not weights:
        continue
    
    w = weights[0]
    scale = np.max(np.abs(w))
    scales.append(scale)
    
    if isinstance(layer, tf.keras.layers.Conv2D):
        w_int8 = np.clip(np.round(w / scale * 127), -128, 127).astype(np.int8)
        with open(os.path.join(OUT_DIR, 'conv1_weights.h'), 'w') as f:
            f.write('#pragma once\n#include <stdint.h>\n\n')
            f.write(f'static const int8_t conv1_weights[] = {{\n  ')
            f.write(', '.join(str(x) for x in w_int8.flatten()))
            f.write('\n};\n')
        header_includes.append('conv1_weights.h')
        print(f'Layer {i} (Conv2D): shape={w_int8.shape}, scale={scale:.4f}')
        
    elif isinstance(layer, tf.keras.layers.Dense):
        w_T = w.T
        w_int8 = np.clip(np.round(w_T / scale * 127), -128, 127).astype(np.int8)
        
        filename = f'layer{dense_idx}_weights.h'
        with open(os.path.join(OUT_DIR, filename), 'w') as f:
            f.write('#pragma once\n#include <stdint.h>\n\n')
            f.write(f'#define LAYER{dense_idx}_ROWS {w.shape[1]}\n')
            f.write(f'#define LAYER{dense_idx}_COLS {w.shape[0]}\n\n')
            f.write(f'static const int8_t layer{dense_idx}_weights[] = {{\n  ')
            f.write(', '.join(str(x) for x in w_int8.flatten()))
            f.write('\n};\n')
        header_includes.append(filename)
        
        b = weights[1]
        bias_int32 = np.round(b * 127).astype(np.int32)
        bias_list.append((f'layer{dense_idx}', bias_int32))
        
        print(f'Layer {i} (Dense): shape={w_int8.shape}, scale={scale:.4f}')
        dense_idx += 1

# test_batch and images
img = x_test[0].flatten()
img_int8 = np.clip(np.round(img * 127), -128, 127).astype(np.int8)

with open(os.path.join(OUT_DIR, 'test_image.h'), 'w') as f:
    f.write('#pragma once\n#include <stdint.h>\n\n')
    f.write(f'#define TEST_LABEL {y_test[0]}\n\n')
    f.write('static const int8_t test_image[] = {\n  ')
    f.write(', '.join(str(x) for x in img_int8))
    f.write('\n};\n')

imgs = x_test[:100].reshape(100, -1)
imgs_int8 = np.clip(np.round(imgs * 127), -128, 127).astype(np.int8)
labels = y_test[:100]

with open(os.path.join(OUT_DIR, 'test_batch.h'), 'w') as f:
    f.write('#pragma once\n#include <stdint.h>\n\n')
    f.write('#define TEST_COUNT 100\n\n')
    f.write('static const int8_t test_batch[] = {\n  ')
    f.write(', '.join(str(x) for x in imgs_int8.flatten()))
    f.write('\n};\n\n')
    f.write('static const uint8_t test_labels[] = {\n  ')
    f.write(', '.join(str(x) for x in labels))
    f.write('\n};\n')

with open(os.path.join(OUT_DIR, 'bias.h'), 'w') as f:
    f.write('#pragma once\n#include <stdint.h>\n\n')
    for name, bias in bias_list:
        f.write(f'static const int32_t {name}_bias[] = {{\n  ')
        f.write(', '.join(str(x) for x in bias))
        f.write('\n};\n\n')

with open(os.path.join(OUT_DIR, 'scales.h'), 'w') as f:
    f.write('#pragma once\n#include <stdint.h>\n\n')
    dense_scale_idx = 1
    for i, scale in enumerate(scales):
        scale_fixed = int((scale / 127) * 65536)
        if i == 0: 
            f.write(f'#define CONV1_SCALE_FIXED {scale_fixed}\n')
        else:
            f.write(f'#define LAYER{dense_scale_idx}_SCALE_FIXED {scale_fixed}\n')
            dense_scale_idx += 1

with open(os.path.join(OUT_DIR, 'model.h'), 'w') as f:
    f.write('#pragma once\n\n')
    f.write('// Model weights and biases\n')
    for h in header_includes:
        f.write(f'#include "{h}"\n')
    f.write('#include "bias.h"\n')
    f.write('#include "scales.h"\n\n')
    f.write('// Test data\n')
    f.write('#include "test_image.h"\n')
    f.write('#include "test_batch.h"\n')

print(f'Tüm header dosyaları başarıyla {OUT_DIR} klasörüne çıkarıldı.')

