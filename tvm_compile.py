import pathlib
import shutil
import tvm
from tvm import relay
from tvm.relay.backend import Executor, Runtime
import tflite

# ──────────────────────────────────
# 설정 (본인 환경에 맞게 수정)
# ──────────────────────────────────
TFLITE_MODEL_PATH = "/home/minchan0410/arduino/quantized_model.tflite"
PROJECT_DIR = pathlib.Path("/home/minchan0410/arduino/mnist_arduino_tvm")

INPUT_NAME  = "input_1"
INPUT_SHAPE = (1, 28, 28, 1)
INPUT_DTYPE = "int8"

# ──────────────────────────────────
# 1. TFLite 모델 로드
# ──────────────────────────────────
with open(TFLITE_MODEL_PATH, "rb") as f:
    tflite_model_buf = f.read()
tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

mod, params = relay.frontend.from_tflite(
    tflite_model,
    shape_dict={INPUT_NAME: INPUT_SHAPE},
    dtype_dict={INPUT_NAME: INPUT_DTYPE},
)

# tvmgen_default.h 생성에 필요한 이름 추출
relay_input_name = mod["main"].params[0].name_hint   # e.g. serving_default_input_1_0
subgraph = tflite_model.Subgraphs(0)
_out_raw = subgraph.Tensors(subgraph.Outputs(0)).Name().decode("utf-8")
relay_output_name = _out_raw.replace(":", "_").replace("/", "_").replace(";", "_")

# ──────────────────────────────────
# 2. nRF52840 (Cortex-M4) 컴파일
#    unpacked-api + interface-api=c 를 써야 generate_project 템플릿과 호환됨
#    packed API (옵션 없이 Executor("aot")) 쓰면 model.c 호출 규약 불일치로 크래시
# ──────────────────────────────────
TARGET   = tvm.target.Target("c -keys=cpu -mcpu=cortex-m4")
RUNTIME  = Runtime("crt")
EXECUTOR = Executor("aot", {"unpacked-api": True, "interface-api": "c"})

with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
    module = relay.build(
        mod,
        target=TARGET,
        params=params,
        runtime=RUNTIME,
        executor=EXECUTOR,
    )
print("컴파일 완료")

# ──────────────────────────────────
# 3. Arduino 프로젝트 생성
# ──────────────────────────────────
template_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("arduino"))

if PROJECT_DIR.exists():
    shutil.rmtree(PROJECT_DIR)

project_options = {
    "board": "nano33ble",
    "project_type": "example_project",
    "arduino_cli_cmd": "/home/minchan0410/arduino/bin/arduino-cli",
}

project = tvm.micro.generate_project(
    template_path,
    module,
    PROJECT_DIR,
    project_options,
)

print(f"프로젝트 생성 완료: {PROJECT_DIR}")
print(f"Arduino IDE에서 .ino 파일을 열어 업로드하세요.")
