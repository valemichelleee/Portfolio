// Include Libraries
#include <LiquidCrystal.h>

// Initialize the library with the numbers of the interface pins
LiquidCrystal lcd(8, 9, 10, 11, 12, 13);

// Pin and Variable definition
double Temp = 0;
int IN4 = 2 ;
int IN3 =3;
int en = 6;
int tempPin = A0;
int led = 5;
int value = 0;
int speed = 0;

void setup()
{
  // Initialize GPIO Pins as output
  pinMode(led,OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(en, OUTPUT);

  // Start LCD
  lcd.begin(16,2);
  lcd.clear();
  delay(1000);
  lcd.print("Hi! Wilkommen :)");
  lcd.setCursor(0,1);
  lcd.print("zur Auto Luefter");
  delay(5000);
  lcd.clear();
}

void loop() {
  lcd.setCursor(0,0);                       //set cursor for the lcd
  lcd.print("Temp. berechnen");             //print title on lcd
  lcd.setCursor(2, 1);                      //set cursor for the lcd
  lcd.write(byte(1));                       
  for (int i=1;i<15;i++)                    //for loop for creating "loading" symbol
  {
    lcd.setCursor(i,1);
    lcd.write(byte(2));
    delay(200);
   }
  delay(1000);
  lcd.clear();

  value = analogRead(tempPin);                 // reading temperature 

  // Script for interface using Temperature Sensor
//  float T= (value/1024.0)*5.0;
//  float Temp = (T-0.5)*100;

  // Script for interface using thermistor
  Temp=log(10000.0*((1024.0/value-1)));
  Temp=1/(0.001129148+(0.000234125+(0.0000000876741*Temp*Temp))*Temp);
  Temp=Temp-273.15;
  
  lcd.setCursor(2, 0);
  lcd.print("Temperatur :");
  lcd.setCursor(2,1);
  lcd.print(Temp);                            // print the temperature that has been read
  lcd.print(" Grad C");
  delay(3000);
  lcd.clear();

  // Output for DC Motor 
  if(Temp<20){                                // When the temperature is low
    speed=0;
    poweroff(speed);
    lcd.print("Ventilator aus");
    delay(2000);
    lcd.clear();
  }

  //When temperature needs to be lowered
  else if(Temp>=25 && Temp<29)
  {
    speed=80;
    poweron(speed);
    lcd.print("Speed Stufe: 1");
    delay(4000);
    lcd.clear();
  }
  
  else if(Temp>=29 && Temp<33)
  { 
    speed = 120;
    poweron(speed);
    lcd.print("Speed Stufe: 2");
    delay(4000);
    lcd.clear();
  }
  
  else if(Temp>=33 && Temp<37)
  { 
    speed = 160;
    poweron(speed);
    lcd.print("Speed Stufe: 3");
    delay(4000);
    lcd.clear();
  }
  
//  else if(Temp>=33 && Temp<36)
//  { 
//    speed = 200;
//    poweron(speed);
//    lcd.print("Speed Stufe: 4");
//    delay(4000);
//    lcd.clear();
  }
//  else if(Temp>=100 && Temp<115)
//  { 
//    speed = 250;
//    poweron(speed);
//    lcd.print("Speed Stufe: 5");
//    delay(4000);
//    lcd.clear();
//  }

  // When temperature is too high
  else{
    speed=0;
    poweroff(speed);
    lcd.print("Temperatur");
    lcd.setCursor(0,1);
    lcd.print("viel zu Heiss");
    digitalWrite(led, HIGH);                    //turning on the LED
    delay(4000);
    lcd.clear();
    digitalWrite(led, LOW);
  }
}

// Function for turning on the DC Motor with the specific given speed
void poweron(int spd){
  analogWrite(en, spd);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

// Funtion for turning off the DC Motor
void poweroff(int spd){
  analogWrite(en, spd);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}
