const int LED = 18;
const int BUTTON = 19;

int STATUS = 0;

void setup() {
  pinMode (LED,OUTPUT);
  pinMode (BUTTON, INPUT_PULLUP);
}

void loop() {
  // put your main code here, to run repeatedly:
  STATUS = digitalRead(BUTTON);
  if (STATUS == LOW)
  {
    digitalWrite(LED,HIGH);
  }
  else
  {
    digitalWrite(LED,LOW);  
  }
}
