# # **Edge Device를 위한 TensorFlow Lite CNN 모델 최적화**
 
 
import tensorflow as tf
import numpy as np
import gzip
from tinymlgen import port
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
print("TensorFlow version:", tf.__version__)
 
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
 
# 모델 구조
# model = models.Sequential([
#     # 입력: 28x28 흑백 이미지
#     layers.Input(shape=(28, 28, 1), batch_size=1),
    
#     layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'),
#     layers.MaxPooling2D(pool_size=(2, 2)), # 14x14
    
#     layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
#     layers.MaxPooling2D(pool_size=(2, 2)), # 7x7
    
#     # Classifier
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(27, activation='softmax') # 27개 클래스 분류
# ])
 
model = models.Sequential([
    layers.Input(shape=(28, 28, 1), batch_size=1),
    layers.Conv2D(4, (3, 3), strides=2, padding='same', activation='relu'),  # 14x14
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
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
open("lite_model.tflite", "wb").write(tflite_model)

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
xxd_i('lite_model.tflite', 'lite_model.cc')

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
open("lite_quantized_model.tflite", "wb").write(tflite_model)
xxd_i('lite_quantized_model.tflite', 'lite_qmodel.cc')
print("done")

# ── 최종 평가 ──────────────────────────────────────────
print("\n=== 모델 성능 비교 ===")

# 1. Keras float 모델
print("\n[Float model]")
_, keras_acc = model.evaluate(test_data, test_label, verbose=0)
print(f"  accuracy: {keras_acc:.4f}")

# 2. 양자화 TFLite 모델 (int8)
print("\n[Quantized(int8) model]")
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
scale, zero_point = input_details[0]['quantization']

correct = 0
for i in range(len(test_data)):
    sample   = test_data[i].reshape(1, 28, 28, 1).astype(np.float32)
    q_sample = (sample / scale + zero_point).astype(np.int8)
    interpreter.set_tensor(input_details[0]['index'], q_sample)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    if np.argmax(output) == test_label[i]:
        correct += 1

quant_acc = correct / len(test_data)
print(f"  accuracy: {quant_acc:.4f}")
