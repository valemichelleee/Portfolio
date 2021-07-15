//This example code is in the Public Domain (or CC0 licensed, at your option.)
//By Evandro Copercini - 2018
//
//This example creates a bridge between Serial and Classical Bluetooth (SPP)
//and also demonstrate that SerialBT have the same functionalities of a normal Serial

#include "BluetoothSerial.h"

const int pin = 34;
const int wet = 1838;
const int dry = 2500;

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Please run `make menuconfig` to and enable it
#endif

BluetoothSerial SerialBT;

void setup() {
  Serial.begin(115200);
  SerialBT.begin("ESP32"); //Bluetooth device name
  Serial.println("The device started, now you can pair it with bluetooth!");
}

void loop() {
  int sensorVal = analogRead(pin);
  int percentageHumidity = map(sensorVal, wet, dry, 100, 0);
  Serial.print(percentageHumidity);
  Serial.println("%");
  if (percentageHumidity <= 10) {
    SerialBT.println("Your plant is thirsty!");
  }
  delay(1000);
}
