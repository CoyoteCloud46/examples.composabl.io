from composabl import Sensor

x = Sensor("x", "this is the x position of the starship")
x_speed = Sensor("x_speed", "this is the speed in the x direction that the starship is headed")
y = Sensor("y", "this is the y position of the starship")
y_speed = Sensor("y_speed", "this is the speed in the y direction that the starship is headed")
angle = Sensor("angle", "this is the angle of the starship")
ang_speed = Sensor("ang_speed", "this is rate that the angle is changing")

sensors = [x, x_speed, y, y_speed, angle, ang_speed]
