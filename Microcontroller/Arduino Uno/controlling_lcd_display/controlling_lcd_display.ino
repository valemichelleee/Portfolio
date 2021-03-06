#include <LiquidCrystal.h>
const int rs = 12, en = 11, d4 = 5, d5 = 4, d6 = 3, d7 = 2;
LiquidCrystal lcd(2, 3, 4, 5, 6, 7);
void setup() {
                         // set up the LCD's number of columns and rows:
  lcd.begin(16, 2);
                      // Print a message to the LCD.
  lcd.print("counting (s)");
}

void loop() {
  // set the cursor to column 0, line 1
  // (note: line 1 is the second row, since counting begins with 0):
  lcd.setCursor(8, 1);
  // print the number of seconds since reset:
  lcd.print(millis() / 1000);
}
