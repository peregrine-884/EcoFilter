import RPi.GPIO as GPIO
import time

output_pins = {
  'JETSON_XAVIER': 18
  'JETSON_NANO': 32,
}
output_pins = output_pins.get(GPIO.model, None)
if output_pins is None:
  raise Exception('PWM not supported on this board')

GPIO.setmode(GPIO.BOARD)
GPIO.setup(output_pins, GPIO.OUT, initial=GPIO.HIGH)
p = GPIO.PWM(output_pins, 50)

print('move_static')
p.start(9.0)
time.sleep(3)

p.stop()
GPIO.cleanup()