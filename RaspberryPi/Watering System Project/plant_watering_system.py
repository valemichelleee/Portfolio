'''Plant Watering System using NodeMCU-ESP32'''

from time import sleep
import serial
from gpiozero import LED

RELAY = LED(23)

while True:
    if __name__ == '__main__':
        SER = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
        SER.flush()
        if SER.in_waiting > 0:
            LINE = SER.readline().decode('utf-8').rstrip()
            LINE_INT = int(LINE)
            print(LINE_INT)
            if LINE_INT <= 10:
                print('watering')
                RELAY.on()
                sleep(5)
                RELAY.off()
                sleep(300)
            else:
                print("the plant has been watered")
                RELAY.off()
                sleep(60)
