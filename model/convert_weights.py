import numpy as np
import os

weights_dir = r'C:\Users\KuruU Bilgisayar\Desktop\inference\model'

for i, shape in [(1, (784, 32)), (2, (32, 10))]:
    w = np.load(os.path.join(weights_dir, f'layer{i}_weights.npy'))
    w = w.reshape(shape)
    flat = w.flatten()
    
    with open(f'layer{i}_weights.h', 'w') as f:
        f.write(f'#pragma once\n')
        f.write(f'#include <stdint.h>\n\n')
        f.write(f'#define LAYER{i}_ROWS {w.shape[1]}\n')
        f.write(f'#define LAYER{i}_COLS {w.shape[0]}\n\n')
        f.write(f'static const int8_t layer{i}_weights[] = {{\n  ')
        f.write(', '.join(str(x) for x in flat))
        f.write('\n};\n')
    
    print(f'layer{i}_weights.h yazıldı: {len(flat)} eleman')