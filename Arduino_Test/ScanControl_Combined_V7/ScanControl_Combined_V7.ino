// === ScanControlRebootFifoPredictWithControl.ino ===

#include <Arduino.h>
#include <Preferences.h>
#include "SH1106Display.h"
#include "ScanControl.h"
#include "TFLite_V1.h"
#include "TFLite_taVNS.h"      // taVNS参数预测模型
#include "Nurosym_Control.h"   // 你的控制函数头文件
#include "SPIFFSLogger.h"

// 实例化 OLED 对象，使用 I²C 地址 0x3C、SDA=39、SCL=38、无复位引脚
SH1106Display oled;

#define MAX_READINGS 12
#define BUTTON_PIN 0
#define DUMP_TIMEOUT_MS  3000  // 等待 dump 命令的超时时间（毫秒）

// NVS for raw readings
Preferences preferences;
// NVS for predicted readings
Preferences predictPref;

// 环形缓冲存储原始读数
uint32_t nextIndex      = 0;
uint32_t totalCount     = 0;
uint16_t storedReadings[MAX_READINGS];

// 环形缓冲存储预测值
uint32_t predNextIndex  = 0;
uint32_t predTotalCount = 0;
uint16_t predictedReadings[MAX_READINGS];


/**
 * 从 NVS 载入原始读数缓存
 */
void loadStoredData() {
  preferences.begin("dexcom", false);
  nextIndex  = preferences.getUInt("nextIdx", 0);
  totalCount = preferences.getUInt("totalCount", 0);
  if (totalCount > MAX_READINGS) totalCount = MAX_READINGS;
  char key[16];
  for (uint32_t i = 0; i < MAX_READINGS; i++) {
    sprintf(key, "r%u", i);
    storedReadings[i] = preferences.getUInt(key, 0);
  }
}

/**
 * 将本次读数写入原始读数环形缓冲
 */
void storeReading(uint16_t reading) {
  char key[16];
  sprintf(key, "r%u", nextIndex);
  preferences.putUInt(key, reading);

  nextIndex = (nextIndex + 1) % MAX_READINGS;
  preferences.putUInt("nextIdx", nextIndex);

  if (totalCount < MAX_READINGS) {
    totalCount++;
    preferences.putUInt("totalCount", totalCount);
  }
}

/**
 * 从 NVS 载入预测值缓存
 */
void loadPredictedData() {
  predictPref.begin("predict", false);
  predNextIndex  = predictPref.getUInt("predNextIdx", 0);
  predTotalCount = predictPref.getUInt("predTotalCount", 0);
  if (predTotalCount > MAX_READINGS) predTotalCount = MAX_READINGS;
  char key[16];
  for (uint32_t i = 0; i < MAX_READINGS; i++) {
    sprintf(key, "p%u", i);
    predictedReadings[i] = predictPref.getUInt(key, 0);
  }
}

/**
 * 将本次预测写入预测值环形缓冲
 */
void storePredictedReading(uint16_t reading) {
  char key[16];
  sprintf(key, "p%u", predNextIndex);
  predictPref.putUInt(key, reading);
  predictedReadings[predNextIndex] = reading;

  predNextIndex = (predNextIndex + 1) % MAX_READINGS;
  predictPref.putUInt("predNextIdx", predNextIndex);

  if (predTotalCount < MAX_READINGS) {
    predTotalCount++;
    predictPref.putUInt("predTotalCount", predTotalCount);
  }
}

/**
 * 清空预测值缓存，准备下一轮
 */
void clearPredictedData() {
  char key[16];
  for (uint32_t i = 0; i < MAX_READINGS; i++) {
    sprintf(key, "p%u", i);
    predictPref.putUInt(key, 0);
  }
  predNextIndex = 0;
  predTotalCount = 0;
  predictPref.putUInt("predNextIdx", predNextIndex);
  predictPref.putUInt("predTotalCount", predTotalCount);
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }
  Serial.println("=== SCAN & PREDICT & CONTROL ===");

  // —— 新增：挂载 SPIFFS —— 
  if (!SPIFFSLogger::begin()) {    
    Serial.println("SPIFFS initialization failure, stopping the service");
    while (1) delay(1000);
  }

  // 监听 'dump' 
  Serial.printf("In %u ms input \"dump\" ,logs will be exported...\n", DUMP_TIMEOUT_MS);
  unsigned long start = millis();
  while (millis() - start < DUMP_TIMEOUT_MS) {
    if (Serial.available()) {
      String cmd = Serial.readStringUntil('\n');
      cmd.trim();
      if (cmd.equalsIgnoreCase("dump")) {
        Serial.println("Receive dump, start exporting the log:");
        SPIFFSLogger::exportAll();
        break;
      }
    }
    delay(10);
  }

  // 1. 载入历史缓存
  loadStoredData();
  loadPredictedData();
  Serial.printf("Loaded %u raw readings (nextIndex=%u)\n", totalCount, nextIndex);
  Serial.printf("Loaded %u predicted readings (predNextIndex=%u)\n",
                predTotalCount, predNextIndex);

  // 2. 初始化 BLE & 模型
  ScanControlInit();
  TFLiteV1Init();
  TFLite_taVNS_Init();

  // —— 新增 OLED 初始化 —— 
  if (!oled.begin()) {
    Serial.println(F("OLED fail"));
    while (true) delay(100);
  }
  oled.updateStatus("Connect...");
  //oled.showIntro(500);  // 欢迎页面

  // 3. 扫描一次并读取
  uint16_t glucose = 0;
  if (ScanControlScanOnce()) {
    while (!ScanControlIsDataReady()) {
      delay(10);
    }
    glucose = ScanControlGetReading();
    Serial.printf("New raw reading: %u mg/dL\n", glucose);
    ScanControlClearDataReady();
    oled.updateStatus("Connected");

  } else {
    Serial.println("Scan failed!");
    oled.updateStatus("Connect...");
  }

  // 4. 存储本次原始读数
  storeReading(glucose);
  storedReadings[(nextIndex + MAX_READINGS - 1) % MAX_READINGS] = glucose;

  //保存，显示预测原始读数
  SPIFFSLogger::logRaw(glucose);
  char buf1[8], buf2[8], buf3[16];
  sprintf(buf1, "%.2f", (float)glucose);
  oled.updateLastReading(buf1);
  

  // 5. 若原始缓存已满，准备模型输入并预测
  if (totalCount >= MAX_READINGS) {
    Serial.println("=== Last 12 Raw Readings (FIFO) ===");
    uint32_t start = nextIndex;
    for (uint32_t i = 0; i < MAX_READINGS; i++) {
      uint32_t idx = (start + i) % MAX_READINGS;
      Serial.printf(" [%2u] %u mg/dL\n", i + 1, storedReadings[idx]);
    }

    float inputs[MAX_READINGS];
    Serial.print("Model input: [");
    for (int i = 0; i < MAX_READINGS; i++) {
      uint32_t idx = (start + i) % MAX_READINGS;
      inputs[i] = (float)storedReadings[idx];
      Serial.print(inputs[i], 2);
      if (i < MAX_READINGS - 1) Serial.print(", ");
    }
    Serial.println("]");

    float pred = TFLiteV1Predict(inputs);
    Serial.printf("Predicted glucose: %.2f mg/dL\n", pred);
    uint16_t predInt = (uint16_t)(pred + 0.5f);
    //保存，显示预测读数
    SPIFFSLogger::logPred(pred);
    sprintf(buf2, "%.2f", pred);
    oled.updatePredict(buf2);

    // 6. 存储一次预测
    storePredictedReading(predInt);
    Serial.printf("Stored predicted[%u] = %u (totalPred=%u)\n",
                  (predNextIndex + MAX_READINGS - 1) % MAX_READINGS,
                  predInt, predTotalCount);
    
    // taVNS 参数计算......
    /////////////////////////
    // 使用相同的血糖序列进行taVNS参数预测
    taVNS_Params tavns_params = TFLite_taVNS_Predict(inputs);
    
    // 获取适合控制例程使用的参数
    int taVNS_time = 10;  // 默认值
    int taVNS_AMP = 10;   // 默认值
    TFLite_taVNS_GetControlParams(tavns_params, &taVNS_time, &taVNS_AMP);
    /////////////////////////
    char time1[8], AMP1[8];
    sprintf(time1, "%u", taVNS_time);
    sprintf(AMP1, "%u", taVNS_AMP);

    // 显示并等待用户10s
    Serial.println("Waiting User Input......(10s)");
    oled.updateStatus("Wait...");
    bool accepted = oled.updateTimeAmp(time1, AMP1, BUTTON_PIN);
    
    // 记录控制决策
    SPIFFSLogger::logControl(accepted, taVNS_time, taVNS_AMP);

    if (accepted) {
      // 用户接受或超时
      Serial.println("User accepted, running control routine");
      oled.updateStatus("Begin...");
      nurosymControlRoutine(taVNS_time, taVNS_AMP);
    } else {
      // 用户在 3s 内按下了按钮，表示拒绝
      Serial.println("User rejected, skipping control routine");
    }

    // 7. 若预测缓存已满，调用控制例程并清空预测缓存
    if (predTotalCount >= MAX_READINGS) {
      Serial.println(">>> Predicted buffer full—running control routine");
      clearPredictedData();
      Serial.println("Cleared predicted buffer, waiting next batch...");
    }
  } else {
    Serial.printf("Need %u more raw readings to fill model input\n",
                  MAX_READINGS - totalCount);
  }

  // 8. 结束 NVS，延迟后重启
  preferences.end();
  predictPref.end();

  Serial.println("Waiting 2s before reboot...");
  oled.updateStatus("Next...");
  delay(2000);
  Serial.println("Rebooting now...");
  esp_restart();
}

void loop() {
  // 所有流程在 setup() 中完成
}
