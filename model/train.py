# # **Edge Device를 위한 TensorFlow Lite CNN 모델 최적화**
 
 
import tensorflow as tf
import numpy as np
import gzip
from tinymlgen import port
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
print("TensorFlow version:", tf.__version__)
 
 
# emnist를 받아 데이터셋 구성
 
 
 
 
# !wget https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip
# !unzip gzip.zip
 
def load_emnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        # 첫 16바이트는 헤더 정보이므로 건너뛴다
        f.read(16)
        # 나머지 데이터를 읽어들여 이미지 데이터로 변환
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        # 데이터를 28x28 이미지로 재구성
        data = data.reshape(-1, 28, 28)
    return data
 
def load_emnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(8)
        # 나머지 데이터를 읽어들여 레이블 데이터로 변환
        buf = f.read()
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels
 
# 훈련 데이터 로드
train_data = load_emnist_images('gzip/emnist-letters-train-images-idx3-ubyte.gz')
train_label = load_emnist_labels('gzip/emnist-letters-train-labels-idx1-ubyte.gz')
 
# 테스트 데이터 로드
test_data = load_emnist_images('gzip/emnist-letters-test-images-idx3-ubyte.gz')
test_label = load_emnist_labels('gzip/emnist-letters-test-labels-idx1-ubyte.gz')
 
 
# 왜 255로 나누는가? 이미지 픽셀 값을 정규화. 왜 정규화 ? Gradient 폭팔/소실 방지?
# sigmoid, relu등이 효과적으로 동작함 
train_data, test_data = train_data / 255.0, test_data/ 255.0
 
 
test_num = 10
plt.matshow(train_data[test_num])
print(chr(train_label[test_num] + 64))
 
 
# 모델 구성
 
 
model = models.Sequential([
    # 입력 레이어 명시 (TFLite 변환 시 입력 형태가 고정되어야 오류가 없습니다)
    layers.Input(shape=(28, 28, 1)),
 
    # 1st Block
    layers.Conv2D(8, kernel_size=(3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    
    # 2nd Block (크기 축소: 28x28 -> 14x14)
    layers.Conv2D(16, kernel_size=(3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # 3rd Block (크기 축소: 14x14 -> 7x7)
    layers.Conv2D(32, kernel_size=(3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # 과적합 방지
    layers.Dropout(0.25),
    
    # Flatten layer
    layers.Flatten(),
    # 최종 출력 (다중 분류이므로 softmax 활성화 함수 사용)
    layers.Dense(27, activation='softmax')
])
 
model.summary()
 
 
# 모델 컴파일 및 훈련
 
 
model.compile(optimizer='adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
 
model.fit(train_data, train_label, epochs=5)
 
 
# 모델 평가
 
 
model.evaluate(test_data, test_label, verbose=2)
 
 
# 모델 저장
 
c_code = port(model, variable_name="testmodel2", pretty_print=True, optimize=False )
print(c_code)
 
# 양자화 없이 모델을 텐서플로우 라이트 형식으로 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
 
# 모델을 디스크에 저장
open("model.tflite", "wb").write(tflite_model)

# xxd -i 기능을 Python으로 구현 (shell 명령어 대체)
def xxd_i(input_file, output_file):
    with open(input_file, 'rb') as f:
        data = f.read()
    var_name = input_file.replace('.', '_').replace('-', '_')
    with open(output_file, 'w') as f:
        f.write(f'unsigned char {var_name}[] = {{\n')
        for i in range(0, len(data), 12):
            chunk = data[i:i+12]
            hex_vals = ', '.join(f'0x{b:02x}' for b in chunk)
            f.write(f'  {hex_vals},\n')
        f.write(f'}};\nunsigned int {var_name}_len = {len(data)};\n')

# 파일을 C 소스파일로 저장
xxd_i('model.tflite', 'model.cc')
# 소스파일을 출력
with open('model.cc') as f:
    print(f.read())

# 양자화하여 모델을 텐서플로우 라이트 형식으로 변환 (INT8 양자화, representative dataset 사용)
converter = tf.lite.TFLiteConverter.from_keras_model(model)

def representative_dataset():
    for i in range(300):
        sample = train_data[i:i+1].astype(np.float32)
        sample = sample.reshape(1, 28, 28, 1)
        yield [sample]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# 모델을 디스크에 저장
open("quantized_model.tflite", "wb").write(tflite_model)
xxd_i('quantized_model.tflite', 'qmodel.cc')
print("done")
