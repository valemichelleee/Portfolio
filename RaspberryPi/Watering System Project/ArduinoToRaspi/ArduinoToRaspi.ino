int pin = 34;
int wet = 1600; //needs to be calibrated
int dry = 2920; //needs to be calibrated

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
}

void loop() {
  // put your main code here, to run repeatedly:
  int sensorVal = analogRead(pin);
  int percentageHumidity = map(sensorVal, wet, dry, 100, 0);
  Serial.println(percentageHumidity);
  delay(500);
}
