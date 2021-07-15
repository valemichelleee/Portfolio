'''Turning on or off, toggling and checking the status' of a specific relay'''

#!/bin/env python3

from time import sleep
from gpiozero import LED

RELAYS = [
    LED(23),
    LED(24),
    LED(25),
    LED(8),
    LED(7),
    LED(1),
    LED(12),
    LED(16)
]


def on_action(relay_option, number):
    '''To turn on the chosen relay'''
    relay_option.on()
    print(f"relay {number} is turning on")


def off_action(relay_option, number):
    '''To turn off the chosen relay'''
    relay_option.off()
    print(f"relay {number} is turning off")


def toggle_action(relay_option, number):
    '''To toggle the chosen relay'''
    print(f"relay {number} is toggling")
    relay_option.on()
    sleep(0.5)
    relay_option.off()
    sleep(0.5)


def print_help():
    '''Print/show help for informations of the required parameter'''
    print('''
Description

Arguments:
  number      number of relay 1 to 8
  action      on, off, or toggle

optional arguments:
  h  show this help message and exit
    ''')


def options():
    '''Input the relay number or show help and check the input'''
    input_str = input("Which relay? ")
    while True:
        if input_str == 'h':
            print_help()
            return

        index = int(input_str) - 1
        if 0 <= index <= 7:
            relay_status(RELAYS[index], input_str)
            relay_action(RELAYS[index], input_str)
            relay_status(RELAYS[index], input_str)
            return
        else:
            print("index out of range")
            return


def relay_action(relay_number, num):
    '''Do the given order(turn on, turn off, toggle) or raise error'''
    action = input("Which action? ")
    while True:

        try:
            return {
                'on': on_action,
                'off': off_action,
                'toggle': toggle_action
            }[action](relay_number, num)
        except KeyError:
            print("Try again")
            return relay_action(relay_number, num)


def relay_status(relay_number, number):
    '''Check initial relay's status'''
    if relay_number.value == 1:
        print(f"relay {number} is on")
    else:
        print(f"relay {number} is off")


while True:
    options()
    sleep(1)
