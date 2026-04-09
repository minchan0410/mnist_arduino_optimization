import tvm
from tvm import relay
from tvm.relay.backend import Runtime, Executor
import tflite
import os
import shutil
import tarfile

# 1. TFLite 모델 로드
with open("quantized_model.tflite", "rb") as f:
    tflite_model_buf = f.read()

tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

# 2. 입력 정보 확인
subgraph = tflite_model.Subgraphs(0)
input_tensor = subgraph.Tensors(subgraph.Inputs(0))
input_name = input_tensor.Name().decode("utf-8")
print(f"입력 텐서 이름: {input_name}")

input_shape = (1, 28, 28, 1)
input_dtype = "int8"
shape_dict = {input_name: input_shape}
dtype_dict = {input_name: input_dtype}

# 3. Relay IR 변환
mod, params = relay.frontend.from_tflite(
    tflite_model,
    shape_dict=shape_dict,
    dtype_dict=dtype_dict
)
print("Relay 변환 완료")

# 4. nRF52840 (Cortex-M4F) 타겟
target = tvm.target.Target("c -keys=arm_cpu -mcpu=cortex-m4 -march=armv7e-m+fp")
runtime = Runtime("crt")
executor = Executor("aot", {"unpacked-api": True, "interface-api": "c"})

# 5. 컴파일
print("컴파일 중...")
with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
    lib = relay.build(
        mod,
        target=target,
        runtime=runtime,
        executor=executor,
        params=params
    )
print("컴파일 완료")

# 6. tar로 저장 후 압축 해제
output_tar = "model_cortexm4.tar"
lib.export_library(output_tar)

output_dir = "./tvm_output"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

with tarfile.open(output_tar) as tar:
    tar.extractall(output_dir)
    print("\n생성된 파일:")
    for name in tar.getnames():
        print(f"  {name}")

# 7. 생성된 헤더에서 함수 시그니처 출력
print("\n=== TVM 생성 함수 시그니처 ===")
for fname in os.listdir(output_dir):
    if fname.endswith(".h"):
        with open(os.path.join(output_dir, fname)) as f:
            for line in f:
                if "tvmgen" in line:
                    print(line.rstrip())
