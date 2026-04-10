#include <Arduino.h>

// C-callable profiling helpers for TVM generated C code
extern "C" {
  void arduino_prof_begin(const char* tag, unsigned long* out_start) {
    *out_start = millis();
  }

  void arduino_prof_end(const char* tag, unsigned long start) {
    unsigned long elapsed = millis() - start;
    Serial.print("[PROF] ");
    Serial.print(tag);
    Serial.print(": ");
    Serial.print(elapsed);
    Serial.println(" ms");
  }
}
