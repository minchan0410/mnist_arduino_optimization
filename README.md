# EMNIST Letters 추론 최적화 Cortex-m4 (Arduino Nano 33 BLE)

## 런타임별 추론 시간 비교

> 측정 환경: Arduino Nano 33 BLE (nRF52840, Cortex-M4F 64 MHz)

| 모델 | 런타임 | 정밀도 | 추론 시간 |
|:----:|:------:|:------:|----------:|
| m1 | Arduino TFLite | float32 | 427.5 ms |
| m1 | Arduino TFLite (CMSIS-NN) | int8 | 133.7 ms |
| m1 | TVM AOT | float32 | 278.8 ms |
| m1 | TVM AOT | int8 | 341.0 ms |
| **m2** | **Arduino TFLite (CMSIS-NN)** | **int8** | **12.0 ms** |
| **m2** | **TVM AOT + CMSIS-NN** | **int8** | **6.6 ms** |

<br><br>
![추론 시간 비교 차트](_etc/inference_time_chart.png)
<br><br>
![모델 정확도 비교 차트](_etc/acc_chart.png)

---

## 모델 아키텍처

<table>
<tr>
  <th align="center">m1 ( Base 모델 )</th>
  <th align="center">m2 ( 경량 모델 )</th>
</tr>
<tr>
  <td align="center" valign="top">
    <img src="_etc/model.tflite.png" width="200"/><br>
  </td>
  <td align="center" valign="top">
    <img src="_etc/lite_quantized_model.tflite.png" width="200"/><br>
  </td>
</tr>
</table>

---
정확도는 93% --> 88%로 약 5% 감소하였지만 m2_int는 양자화 및 모델 경량화를 통해 기존(m1_float)대비 **98.45%** 감소된 속도를 통해 추론을 수행할 수 있었다.
