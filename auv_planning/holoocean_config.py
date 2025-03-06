scenario = {
    "name": "Hovering",
    "world": "OpenWater",
    "package_name": "Ocean",
    "main_agent": "auv0",
    "ticks_per_sec": 100,
    "frames_per_sec": False,
    "agents":[
        {
            "agent_name": "auv0",
            "agent_type": "HoveringAUV",
            "sensors": [
                {
                    "sensor_type": "PoseSensor",
                    "socket": "COM",
                    "Hz": 100
                },
                {
                    "sensor_type": "VelocitySensor",
                    "socket": "COM",
                    "Hz": 100
                },
                {
                    "sensor_type": "RotationSensor",
                    "socket": "COM",
                    "Hz": 100
                },
                {
                    "sensor_type": "RangeFinderSensor",
                    "sensor_name": "HorizontalRangeSensor",
                    "socket": "COM",
                    "Hz": 100,
                    "configuration": {
                        "LaserCount":8,
                        "LaserDebug":True
                    }
                },
                {
                    "sensor_type": "RangeFinderSensor",
                    "sensor_name": "UpRangeSensor",
                    "socket": "COM",
                    "Hz": 100,
                    "configuration": {
                        "LaserCount":1,
                        "LaserAngle":90,
                        "LaserDebug":True
                    }
                },
                {
                    "sensor_type": "RangeFinderSensor",
                    "sensor_name": "DownRangeSensor",
                    "socket": "COM",
                    "Hz": 100,
                    "configuration": {
                        "LaserCount":1,
                        "LaserAngle":-90,
                        "LaserDebug":True
                    }
                },
                {
                    "sensor_type": "RangeFinderSensor",
                    "sensor_name": "UpInclinedRangeSensor",
                    "socket": "COM",
                    "Hz": 100,
                    "configuration": {
                        "LaserCount":2,
                        "LaserAngle":45,
                        "LaserDebug":True
                    }
                },
                {
                    "sensor_type": "RangeFinderSensor",
                    "sensor_name": "DownInclinedRangeSensor",
                    "socket": "COM",
                    "Hz": 100,
                    "configuration": {
                        "LaserCount":2,
                        "LaserAngle":-45,
                        "LaserDebug":True
                    }
                }
            ],
            "control_scheme": 2,
            "location": [0, 0, -10],
            "rotation": [0.0, 0.0, 0.0]
        }
    ],

    "window_width":  1680,
    "window_height": 980
}