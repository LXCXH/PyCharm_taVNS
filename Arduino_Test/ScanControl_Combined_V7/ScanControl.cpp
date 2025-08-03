// ScanControl.cpp
#include "ScanControl.h"

#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEScan.h>
#include <mbedtls/aes.h>
#include "esp_system.h"

// ---------- 用户配置区 ----------
static std::string transmitterID = "810005";
static BLEUUID serviceUUID       = BLEUUID("f8083532-849e-531c-c594-30f1f86a4ea5");
static BLEUUID authUUID          = BLEUUID("f8083535-849e-531c-c594-30f1f86a4ea5");
static BLEUUID controlUUID       = BLEUUID("f8083534-849e-531c-c594-30f1f86a4ea5");
static const uint8_t bothOn[]    = {0x03, 0x00}; // 启用 notify + indicate
static const bool useAltChannel  = true;         // 使用备用通道
// ----------------------------------

static uint16_t  _lastReading = 0;
static bool      _dataReady   = false;

// ----------------------------------

static std::string AuthCallbackResponse;
static std::string ControlCallbackResponse;
static volatile bool bondingFinished = false;

// —— BLE 安全回调 —— 
class MySecurity : public BLESecurityCallbacks {
  uint32_t onPassKeyRequest() override { return 123456; }
  void onPassKeyNotify(uint32_t) override {}
  bool onConfirmPIN(uint32_t) override { return true; }
  bool onSecurityRequest() override { return true; }
  void onAuthenticationComplete(esp_ble_auth_cmpl_t auth) override {
    //Serial.print("→ Bonding ");
    //Serial.println(auth.success ? "succeeded" : "failed");
    bondingFinished = auth.success;
  }
};

class MyClientCallbacks : public BLEClientCallbacks {
  void onConnect(BLEClient*) override {
    //Serial.println("→ Connected to Dexcom device");
  }
  void onDisconnect(BLEClient*) override {
    //Serial.println("→ Disconnected from Dexcom device");
  }
};

// —— AUTH indication 回调 —— 
static void indicateAuthCallback(BLERemoteCharacteristic*, uint8_t* data, size_t len, bool) {
  AuthCallbackResponse.assign((char*)data, len);
}

// —— CONTROL notify 回调 —— 
static void indicateControlCallback(BLERemoteCharacteristic*, uint8_t* data, size_t len, bool) {
  ControlCallbackResponse.assign((char*)data, len);
}

// —— 等待 AUTH indication —— 
static bool AuthWaitToReceive(uint32_t timeout_ms = 2000) {
  unsigned long start = millis();
  while (millis() - start < timeout_ms) {
    if (!AuthCallbackResponse.empty()) return true;
    delay(10);
  }
  return false;
}

// —— 执行认证逻辑 —— 

static bool authenticate(BLERemoteCharacteristic* pAuthChar) {
  //Serial.println("=== AUTHENTICATION PHASE START ===");
  //Serial.println("→ Registering for AUTH indication...");

  pAuthChar->registerForNotify(indicateAuthCallback, false);
  pAuthChar->getDescriptor(BLEUUID((uint16_t)0x2902))
           ->writeValue((uint8_t*)bothOn, sizeof(bothOn), true);

  std::string authReq = {
    0x01,0x19,0xF3,0x89,0xF8,0xB7,0x58,0x41,0x33,
    (char)(useAltChannel ? 0x01 : 0x02)
  };
  //Serial.println("→ Sending Auth request...");
  pAuthChar->writeValue((uint8_t*)authReq.data(), authReq.size(), true);

  //Serial.println("→ Waiting for Auth challenge...");
  if (!AuthWaitToReceive()) {
    //Serial.println("!! Auth challenge not received within timeout");
    return false;
  }
  std::string challengeMsg = AuthCallbackResponse;
  AuthCallbackResponse.clear();

  if (challengeMsg.size() != 17 || (uint8_t)challengeMsg[0] != 0x03) {
    //Serial.println("!! Invalid auth challenge");
    return false;
  }
  //Serial.println("→ Received valid Auth challenge");

  std::string challenge = challengeMsg.substr(9, 8);
  challenge += challenge;
  std::string key = "00" + transmitterID + "00" + transmitterID;

  //Serial.println("→ Computing AES hash from challenge...");
  unsigned char outbuf[16];
  mbedtls_aes_context aes;
  mbedtls_aes_init(&aes);
  mbedtls_aes_setkey_enc(&aes, (const unsigned char*)key.data(), key.size()*8);
  mbedtls_aes_crypt_ecb(&aes, MBEDTLS_AES_ENCRYPT,
                        (const unsigned char*)challenge.data(),
                        outbuf);
  mbedtls_aes_free(&aes);

  std::string hashResp((char*)outbuf, 16);
  hashResp.resize(8);

  std::string authResp = std::string({0x04}) + hashResp;
  //Serial.println("→ Sending Auth response...");
  pAuthChar->writeValue((uint8_t*)authResp.data(), authResp.size(), true);

  //Serial.println("→ Waiting for Auth status...");
  if (!AuthWaitToReceive()) {
    //Serial.println("!! No auth status received");
    return false;
  }
  std::string statusMsg = AuthCallbackResponse;
  //Serial.print("→ Auth status: ");
  //Serial.println((uint8_t)statusMsg[1] == 0x01 ? "SUCCESS" : "FAIL");
  return ((uint8_t)statusMsg[1] == 0x01);
}

// —— 发送 CONTROL 命令并等待 notify —— 
bool ControlSendAndWait(BLERemoteCharacteristic* pCtrl,
                        const std::string& cmd,
                        const char* label) {
  ControlCallbackResponse.clear();
  //Serial.print("→ Sending ");
  //Serial.println(label);
  pCtrl->writeValue((uint8_t*)cmd.data(), cmd.size(), true);

  unsigned long start = millis();
  while (millis() - start < 2000) {
    if (!ControlCallbackResponse.empty()) {
      //Serial.print("→ ");
      //Serial.print(label);
      //Serial.print(" response: ");
      for (uint8_t c : ControlCallbackResponse) {
        char buf[4];
        sprintf(buf, "%02X ", c);
        //Serial.print(buf);
      }
      //Serial.println();
      return true;
    }
    delay(10);
  }
  //Serial.print("!! No response for ");
  //Serial.println(label);
  return false;
}

// —— 解析时间信息 —— 
void decodeTime(const std::string& resp) {
  //Serial.println("--- Decoding Time Info ---");
  if (resp.size() != 17 || (uint8_t)resp[0] != 0x25) {
    //Serial.println("Invalid Time response");
    return;
  }
  uint8_t status       = resp[1];
  uint32_t currentTime = resp[2] | resp[3]<<8 | resp[4]<<16 | resp[5]<<24;
  uint32_t sessionStart= resp[6] | resp[7]<<8 | resp[8]<<16 | resp[9]<<24;

  //Serial.printf("Status:              %d\n", status);
  //Serial.printf("Since activation:    %u sec\n", currentTime);
  //Serial.printf("Since session start: %u sec\n", sessionStart);
}

// —— 解析电池信息 ——
void decodeBattery(const std::string& resp) {
  //Serial.println("--- Decoding Battery Info ---");
  if (resp.size() < 10 || (uint8_t)resp[0] != 0x23) {
    //Serial.println("Invalid Battery response");
    return;
  }
  uint8_t status = resp[1];
  uint16_t voltA = resp[2] | resp[3]<<8;
  uint16_t voltB = resp[4] | resp[5]<<8;

  //Serial.printf("Status:      %d\n", status);
  //Serial.printf("Voltage A:   %u mV\n", voltA);
  //Serial.printf("Voltage B:   %u mV\n", voltB);

  if (resp.size() >= 12) {
    uint16_t res = resp[6] | resp[7]<<8;
    uint8_t run  = resp[8];
    uint8_t tmp  = resp[9];
    //Serial.printf("Resistance:  %u\n", res);
    //Serial.printf("Runtime:     %u min\n", run);
    //Serial.printf("Temperature: %u °C\n", tmp);
  }
}

// —— 解析血糖信息 —— 
void decodeGlucose(const std::string& resp) {
  //Serial.println("--- Decoding Glucose Info ---");
  uint8_t opcode = resp[0];
  if (resp.size() < 14 || (opcode != 0x31 && opcode != 0x4F)) {
    //Serial.println("Invalid Glucose response");
    return;
  }
  uint8_t status = resp[1];
  uint32_t seq   = resp[2] | resp[3]<<8 | resp[4]<<16 | resp[5]<<24;
  uint32_t ts    = resp[6] | resp[7]<<8 | resp[8]<<16 | resp[9]<<24;
  uint16_t raw   = resp[10] | resp[11]<<8;
  bool disp      = (raw & 0xF000) != 0;
  uint16_t gl    = raw & 0x0FFF;

  // Serial.printf("Status:       %d\n", status);
  // Serial.printf("Sequence:     %u\n", seq);
  // Serial.printf("Timestamp:    %u\n", ts);
  // Serial.printf("DisplayOnly:  %s\n", disp ? "Yes" : "No");
  // Serial.printf("Glucose:      %u mg/dL\n", gl);
  // Serial.printf("Sensor state: %u\n", resp[12]);
  // Serial.printf("Trend:        %d mg/dL/min\n", (int8_t)resp[13]);
}

static bool scanAndConnect() {
  //Serial.println("=== DEVICE SCAN & CONNECTION START ===");
  String txID = String(transmitterID.c_str());
  String expectedName = "Dexcom" + txID.substring(4, 6);
  Serial.print("→ Looking for device: ");
  Serial.println(expectedName);

  BLEScan* scan = BLEDevice::getScan();
  scan->setInterval(100);
  scan->setWindow(99);
  scan->setActiveScan(true);

  while (true) {
    BLEScanResults* results = scan->start(5);
    for (int i = 0; i < results->getCount(); i++) {
      auto dev = results->getDevice(i);
      if (dev.haveName() && dev.getName() == expectedName) {
        //Serial.print("→ Found device at: ");
        //Serial.println(dev.getAddress().toString().c_str());

        BLEClient* client = BLEDevice::createClient();
        client->setClientCallbacks(new MyClientCallbacks());
        client->connect(&dev);

        auto svc   = client->getService(serviceUUID);
        auto cAuth = svc->getCharacteristic(authUUID);
        auto cCtrl = svc->getCharacteristic(controlUUID);

        if (!authenticate(cAuth)) {
          //Serial.println("Authentication failed. Disconnecting...");
          client->disconnect();
          return false;
        }

        //Serial.println("=== BONDING PHASE START ===");
        BLEDevice::setEncryptionLevel(ESP_BLE_SEC_ENCRYPT_MITM);
        BLEDevice::setSecurityCallbacks(new MySecurity());
        cAuth->writeValue((uint8_t[]){0x06,0x19}, 2, true);
        cAuth->writeValue((uint8_t[]){0x07}, 1, true);
        //Serial.println("→ Waiting for bonding to complete...");
        //----------------------------------------//
        //while (!bondingFinished) delay(10);
        //----------------------------------------//
        //Serial.println("→ Bonding successful");

        cCtrl->registerForNotify(indicateControlCallback, false);
        cCtrl->getDescriptor(BLEUUID((uint16_t)0x2902))
             ->writeValue((uint8_t*)bothOn, sizeof(bothOn), true);

        //Serial.println("=== CONTROL PHASE START ===");

        ControlSendAndWait(cCtrl, std::string({0x24,0xE6,0x64}), "Time");
        decodeTime(ControlCallbackResponse);

        ControlSendAndWait(cCtrl, std::string({0x22,0x20,0x04}), "Battery");
        decodeBattery(ControlCallbackResponse);

        ControlSendAndWait(cCtrl, std::string({0x30,0x53,0x36}), "Glucose");
        decodeGlucose(ControlCallbackResponse);

        // 抽取原始值第10/11字节
        auto &r = ControlCallbackResponse;
        uint16_t v = (r.size()>=12)
          ? ((uint8_t)r[10] | ((uint8_t)r[11]<<8)) & 0x0FFF
          : 0;
        _lastReading = v;
        _dataReady   = true;

        //Serial.println("→ Disconnecting...");

        client->disconnect();
        Serial.println("=== SESSION COMPLETE ===");
        return true;
      }
    }
    Serial.println("→ Device not found. Retrying...");
    delay(200);
  }
  return false;
}


// —— 对外接口 —— 
void ScanControlInit() {
  Serial.begin(115200);
  delay(2000);
  //Serial.println("=== DEXCOM BLE INIT ===");
  BLEDevice::init("");
}

bool ScanControlScanOnce() {
  _dataReady = false;
  return scanAndConnect();
}

bool ScanControlIsDataReady() {
  return _dataReady;
}

uint16_t ScanControlGetReading() {
  return _lastReading;
}

void ScanControlClearDataReady() {
  _dataReady = false;
}
