#!/bin/env python3

import argparse
from gpiozero import Button


class ReadStatus:
    '''A class represent the LED pins and check their status'''
    def __init__(self):
        '''initializing led pins'''
        self.pins = [
            Button(17),
            Button(27),
            Button(22),
            Button(10),
            Button(9),
            Button(11),
            Button(0),
            Button(5),
            Button(6),
            Button(13)
        ]

    def state(self, num):
        '''checking the status of a specific'''
        int_num = int(0 if num is None else num)
        if int_num < 1 or int_num > 10:
            raise ValueError("unknown pin number (1 to 10 allowed)")

        index = int_num - 1
        pin = self.pins[index]

        if pin.is_pressed:
            print(f"{int_num} is high")
        else:
            print(f"{int_num} is low")


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    PARSER.add_argument("number", type=int, nargs="?",
                        help="pin number from 1 to 10")
    PARSER.add_argument('--all', action='store_true',
                        help="print out all LEDs state")

    ARGS = PARSER.parse_args()
    READ_STATUS = ReadStatus()

    if ARGS.all:
        for i in range(1, 11):
            READ_STATUS.state(i)
    else:
        READ_STATUS.state(ARGS.number)
