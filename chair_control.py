#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Program containing useful functions to interact with the game
(accelerate forward or backward, boost, launch object, rescue, turn using the joystick)

Functions that involve pressing and releasing a button are launched via a thread
to avoid impacting the main loop.
"""

import time
import threading
from pynput.keyboard import Key, Controller
from multiprocessing import Value
from oscpy.server import OSCThreadServer


import numpy as np
import vgamepad as vg

sens = 2.5
joy_curve_power = 1.15

def play_accel(accel_state):
    if accel_state == 0:
        keyboard.release(Key.up)
        keyboard.release(Key.down)
    
    elif accel_state == 1:
        keyboard.press(Key.up)
        keyboard.release(Key.down)
    
    elif accel_state == -1:
        keyboard.press(Key.down)
        keyboard.release(Key.up) 
        
    return accel_state


def press_boost(duration):
    def boost_thread(duration):
        keyboard.press('n')
        time.sleep(duration)
        keyboard.release('n')
    
    threading.Thread(target=boost_thread, args=(duration,)).start()
    
def press_object():
    def object_thread():
        keyboard.press(Key.space)
        time.sleep(0.016)
        keyboard.release(Key.space)
        
    
    threading.Thread(target=object_thread, args=()).start()
    
def press_rescue():
    def rescue_thread():
        keyboard.press(Key.backspace)
        time.sleep(0.016)
        keyboard.release(Key.backspace)
        
    
    threading.Thread(target=rescue_thread, args=()).start()
    

def release_direction():
    keyboard.release(Key.left)
    keyboard.release(Key.right)
    


def counter_drift(drift_direction, current_steer_command):
    """
    Function executed in a thread for drifting
    Drift is triggered when raising the left or right hand.
    To improve setup performance, this drift has a direction based on the raised hand.
    The drift automatically lasts 1 second to allow the yellow boost after the drift.
    This thread knows the current steering value "current_steer_command," which is a shared variable.
    This value is used to adjust steering so that turning in the direction
    opposite to the drift is easier (hence the name counter_steer).
    """
    
    _dir = (Key.left if drift_direction == -1 else Key.right)
    counter_dir = (Key.right if drift_direction == -1 else Key.left)
    
    release_direction()
    time.sleep(0.016)
    keyboard.press(_dir)
    time.sleep(0.1)
    keyboard.press("v")
    time.sleep(0.016)
    keyboard.release(_dir)
    
    start_drift_time = time.time()
    # Time for the yellow turbo to charge
    while (time.time() - start_drift_time) < 1.0:
        current_steer = current_steer_command.value
        
        # Left drift
        if drift_direction == -1:
            # Steering changes to make it easier to turn in the direction opposite to the drift
            counter_steer = (current_steer + 0.5) * 2
        # Right drift
        elif drift_direction == 1:
            counter_steer = (current_steer - 0.5) * 2
            
            
        counter_steer = np.clip(counter_steer, -1, 1)
        joystick_play(counter_steer)
        time.sleep(0.016)
        
    
    keyboard.release("v")

def apply_steer_curve(val, deadzone=0.1, power=1):
    """
    Allows adding a dead zone to the joystick for easier straight-line driving,
    applying a power curve (e.g., x**1.2) for more precision near the center,
    and clipping values between -1 and 1 (-1 = fully left, 1 = fully right).
    """
    d = deadzone
    val = np.clip(val, -1, 1)
    # I used the site desmos.com to help find this function
    new_val = max(0, (abs(val)**power - d) * ((1+d) / (1-d**2))) * np.sign(val)
    return new_val  
        
def orient_to_steer(orient, joystick_deadzone):
    """The phone's orientation is converted into a steering value."""
    steer = -sens*orient
    steer = apply_steer_curve(steer, deadzone=joystick_deadzone, power=joy_curve_power)
    return steer

def joystick_play(steer):
    """To turn using the virtual joystick."""
    gamepad.left_joystick_float(x_value_float=steer, y_value_float=0.0) 
    gamepad.update()
    
def new_matrix_state(*matrix_state):
    """Receives phone orientation via OSC."""
    global phone_orient

    phone_orient = matrix_state[1]

gamepad = vg.VX360Gamepad()
keyboard = Controller()

if __name__ == '__main__':
    # If we just want to test direction via phone angle
    
    joystick_deadzone = 0.1
    osc = OSCThreadServer()
    
    sock = osc.listen(address='0.0.0.0', port=8000, default=True)
    
    osc.bind(b'/gyrosc/rmatrix', new_matrix_state)
    
    phone_orient = 0
    while True:
        steer = orient_to_steer(phone_orient, joystick_deadzone)
    osc.stop()  # Stop the default socket
