// === Nurosym_Control.cpp ===

#include "Nurosym_Control.h"

// 包装后的主控制流程
void nurosymControlRoutine(int time, int strength) {
  // 初始化 GPIO
  pinMode(GPIO_POWER, OUTPUT);
  pinMode(GPIO_ENTER, OUTPUT);
  pinMode(GPIO_UP, OUTPUT);
  pinMode(GPIO_DOWN, OUTPUT);

  digitalWrite(GPIO_POWER, LOW);
  digitalWrite(GPIO_ENTER, LOW);
  digitalWrite(GPIO_UP, LOW);
  digitalWrite(GPIO_DOWN, LOW);

  // 约束参数到合法范围
  int finalTime     = constrain(time, 1, MAX_DURATION); // [1–60]
  int finalStrength = constrain(strength, 0, 15);       // [0–15]

  Serial.print("[Config] Duration = ");
  Serial.print(finalTime);
  Serial.print(" min, Strength = ");
  Serial.println(finalStrength);

  // 开机按键
  Serial.println("Power Event Triggered (On)");
  digitalWrite(GPIO_POWER, HIGH);
  delay(3000);
  digitalWrite(GPIO_POWER, LOW);

  // 设置“持续时间”值
  int currentTimeSetting = 1;
  while (currentTimeSetting <= finalTime) {
    delay(200);
    digitalWrite(GPIO_UP, HIGH);
    delay(100);
    digitalWrite(GPIO_UP, LOW);
    Serial.print("GPIO UP Triggered, Time: "); 
    Serial.println(currentTimeSetting);
    currentTimeSetting++;
  }

  // 确认时间
  Serial.println("GPIO ENTER Triggered (Confirm Time)");
  delay(500);
  digitalWrite(GPIO_ENTER, HIGH);
  delay(100);
  digitalWrite(GPIO_ENTER, LOW);

  int currentStrengthSetting = 0;
  while (currentStrengthSetting < finalStrength) {
    delay(200);
    digitalWrite(GPIO_UP, HIGH);
    delay(100);
    digitalWrite(GPIO_UP, LOW);
    Serial.print("GPIO UP Triggered (Strength), Level: "); 
    Serial.println(currentStrengthSetting);
    currentStrengthSetting++;
  }

  // 确认强度
  Serial.println("GPIO ENTER Triggered (Confirm Strength)");
  delay(500);
  digitalWrite(GPIO_ENTER, HIGH);
  delay(100);
  digitalWrite(GPIO_ENTER, LOW);

  Serial.print("Output Begin, Time: ");
  Serial.print(finalTime);
  Serial.print(", Strength: ");
  Serial.println(finalStrength);
  delay(1000);

  // 复位流程：关机、开机、调回持续时间
  Serial.println("Reset Event Triggered (Off)");
  digitalWrite(GPIO_POWER, HIGH);
  delay(100);
  digitalWrite(GPIO_POWER, LOW);
  delay(100);
  digitalWrite(GPIO_POWER, HIGH);
  delay(3000);
  digitalWrite(GPIO_POWER, LOW);
  delay(1000);

  Serial.println("Reset Event Triggered (On)");
  digitalWrite(GPIO_POWER, HIGH); 
  delay(3000);
  digitalWrite(GPIO_POWER, LOW);

  // 降回持续时间档位
  while (finalTime > 0) {
    delay(200);
    digitalWrite(GPIO_DOWN, HIGH);
    delay(100);
    digitalWrite(GPIO_DOWN, LOW);
    Serial.print("GPIO DOWN Triggered (Time), Count: "); 
    Serial.println(finalTime);
    finalTime--;
  }

  delay(1000);
  Serial.println("Power Event Triggered");
  digitalWrite(GPIO_POWER, HIGH);
  delay(3000);
  digitalWrite(GPIO_POWER, LOW);

  Serial.println("Reset finish!!!");
}
