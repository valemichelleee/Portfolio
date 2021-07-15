#include <SPI.h>
#include <MFRC522.h>
#include <LiquidCrystal.h>
#include <Servo.h>

#define RST_PIN 9
#define SS_PIN 10

const int LED = 19;
const int BUTTON = 18;

byte readCard[4];
char* myTags[100] = {};
int tagsCount = 0;
String tagID = "";
boolean successRead = false;
boolean correctTag = false;
boolean doorOpened = false;
int state = 0;

MFRC522 mfrc522(SS_PIN, RST_PIN);
LiquidCrystal lcd(2, 3, 4, 5, 6, 7);
Servo myServo;

void setup() {
  // put your setup code here, to run once:
  SPI.begin();
  mfrc522.PCD_Init();
  lcd.begin(16,2);
  myServo.attach(8);
  pinMode (BUTTON, INPUT_PULLUP);
  pinMode (LED, OUTPUT);

  myServo.write(10);
  lcd.print("-No Master Tag!-");
  lcd.setCursor(0,1);
  lcd.print("   SCAN NOW");

  while (!successRead) {
    successRead = getID();
    if (successRead == true) {
      myTags[tagsCount] = strdup(tagID.c_str());
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Master Tag Set!");
      tagsCount = 1;
    }
  }
  successRead = false;
  printNormalModeMessage();
}

void loop() {
  // put your main code here, to run repeatedly:
  state = digitalRead(BUTTON);
  if (state == HIGH) {
    digitalWrite(LED,HIGH);
    if(!mfrc522.PICC_IsNewCardPresent()){
      return;
    }
    if(!mfrc522.PICC_ReadCardSerial()){
      return;
    }
    tagID ="";
    for(uint8_t i = 0; i<4; i++){
      readCard[i] = mfrc522.uid.uidByte[i];
      tagID.concat(String(mfrc522.uid.uidByte[i], HEX));
    }
    tagID.toUpperCase();
    mfrc522.PICC_HaltA();

    correctTag = false;

    if (tagID == myTags[0]){
      lcd.clear();
      lcd.print("Program mode:");
      lcd.setCursor(0,1);
      lcd.print("Add/Remove Tag");
      while (!successRead){
        successRead = getID();
        if(successRead == true){
          for (int i=1; i<100; i++){
            if (tagID == myTags[i]){
              myTags[i]= "";
              lcd.clear();
              lcd.setCursor(0,0);
              lcd.print("Tag Removed!");
              printNormalModeMessage();
              return;
              }
            }
            myTags[tagsCount] = strdup(tagID.c_str());
            lcd.clear();
            lcd.setCursor(0,0);
            lcd.print("Tag Added!");
            printNormalModeMessage();
            tagsCount = tagsCount + 1;
            return;
            }
          }
        }
        successRead = false;
        for (int i = 0; i<100; i++){
        if (tagID == myTags[i]){
          lcd.clear();
          lcd.setCursor(0,0);
          lcd.print("Acces Granted!");
          myServo.write(170);
          printNormalModeMessage();
          correctTag = true;
          i = 101;
          } 
        }
        if (correctTag == false){
          lcd.clear();
          lcd.setCursor(0,0);
          lcd.print("Accses Denied!");
          printNormalModeMessage();
        }
  }
  else {
    digitalWrite(LED,LOW);
    lcd.clear();
    lcd.setCursor (0,0);
    lcd.print("Door Opened!");
    while(!doorOpened){
      digitalWrite(LED,LOW);
      state = digitalRead(BUTTON);
      if (state == HIGH){
        doorOpened = true;
      }
    }
  doorOpened = false;
  delay(500);
  myServo.write(10);
  printNormalModeMessage();
 }
}

uint8_t getID() {
  if (!mfrc522.PICC_IsNewCardPresent()){
    return 0;
  }
  if (!mfrc522.PICC_ReadCardSerial()){
    return 0;
  }
  tagID = "";
  for(uint8_t i = 0; i<4; i++){
    readCard[i] = mfrc522.uid.uidByte[i];
    tagID.concat(String(mfrc522.uid.uidByte[i], HEX));
  }
  tagID.toUpperCase();
  mfrc522.PICC_HaltA();
  return 1;
}

void printNormalModeMessage(){
  delay(1500);
  lcd.clear();
  lcd.print("-Acces Control-");
  lcd.setCursor(0,1);
  lcd.print("Scan your Tag!");
}
