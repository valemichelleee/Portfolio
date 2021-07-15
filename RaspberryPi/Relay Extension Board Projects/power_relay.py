#!/bin/env python3

import argparse
from time import sleep
from gpiozero import DigitalInputDevice, OutputDevice


class Power:
    '''A class to represent a relay and the action applied'''

    def __init__(self, num: int):
        '''initializing relays'''
        device_gpio_map = [
            [23, 17, 27],
            [24, 22, 10],
            [25, 9, 11],
            [8, 0, 5],
            [7, 6, 13],
            [1],
            [12],
            [16]
        ]

        if num < 1 or num > 9:
            raise ValueError("unknown relay number (1 to 8 allowed)")

        self.status = "none"

        index = num - 1

        self.device_has_led = len(device_gpio_map[index]) > 1
        self.power_button = OutputDevice(device_gpio_map[index][0])

        if self.device_has_led:
            self.green_led = DigitalInputDevice(device_gpio_map[index][1])
            self.red_led = DigitalInputDevice(device_gpio_map[index][2])
        else:
            print(f"Device {num} has no LED support, no status checks available")

    def main(self, act):
        '''main process'''
        self.status_led()

        self.mode_option(act)

        self.waiting_led(self.status)
        self.status_led()

    def status_led(self):
        '''check the led status'''
        if not self.device_has_led:
            return

        if self.green_led.value and not self.red_led.value:
            print("status led : red")
            self.status = "red"
        elif self.red_led.value and not self.green_led.value:
            print("status led : green")
            self.status = "green"
        else:
            print("status led : off")
            self.status = "off"

    def mode_option(self, action):
        '''action applied to the relay'''
        if action == "toggle":
            self.toggle_power_btn()
        elif action == "hardoff":
            self.hardoff()
        elif action == "none":
            return 0
        else:
            raise ValueError("unknown action requested")

    def toggle_power_btn(self):
        '''toggle relay'''
        print("toggling power button")
        self.power_button.on()
        sleep(0.5)
        self.power_button.off()
        sleep(1)

    def hardoff(self):
        '''hardoff relay'''
        print("longpress power button (this block 6 secs)")
        self.power_button.on()
        sleep(6)
        self.power_button.off()

    def waiting_led(self, state):
        '''waiting led until specific order occured'''
        if not self.device_has_led:
            return

        if state == "off":
            print("device is turning on")
            print("waiting for the green LED...")
            self.green_led.wait_for_inactive(
                timeout=120) and self.red_led.wait_for_active(timeout=120)
            if self.red_led.value and not self.green_led.value:
                print("device has been turned on")
            else:
                print("Timeout : DEVICE ERROR. Waited for 2 minutes")
        else:
            print("device is turning off")
            print("waiting for the green and red LED...")
            self.green_led.wait_for_active(
                timeout=None) and self.red_led.wait_for_active(timeout=None)
            print("device has been turned off")


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    PARSER.add_argument("number", type=int, help="number of relay")
    PARSER.add_argument("action", default="none",
                        nargs="?", help="toggle, hardoff")

    ARGS = PARSER.parse_args()
    POWER = Power(ARGS.number)
    POWER.main(ARGS.action)
