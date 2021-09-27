from datetime import datetime


class Sit2stand:
    def __init__(self, reps):
        now = datetime.now()  # current date and time
        self.datetime = now.strftime("%m/%d/%Y, %H:%M:%S")
        self.reps = reps

    def asdict(self):
        return {'datetime': self.datetime, 'reps': self.reps}


class HipAbduction:
    def __init__(self, lr_med, hip_r_med, hip_l_med):
        now = datetime.now()  # current date and time
        self.datetime = now.strftime("%m/%d/%Y, %H:%M:%S")
        self.lr_med = lr_med
        self.hip_r_med = hip_r_med
        self.hip_l_med = hip_l_med

    def asdict(self):
        return {'datetime': self.datetime, 'lr_med': self.lr_med, 'hip_r_med': self.hip_r_med,
                'hip_l_med': self.hip_l_med}


class CervRot:
    def __init__(self, l_med, r_med):
        now = datetime.now()  # current date and time
        self.datetime = now.strftime("%m/%d/%Y, %H:%M:%S")

        self.l_med = l_med
        self.r_med = r_med

    def asdict(self):
        return {'datetime': self.datetime, 'l_med': self.l_med, 'r_med': self.r_med}


class HipRot:

    def __init__(self, lr_med, l_med, r_med):
        now = datetime.now()  # current date and time
        self.datetime = now.strftime("%m/%d/%Y, %H:%M:%S")
        self.lr_med = lr_med
        self.l_med = l_med
        self.r_med = r_med

    def asdict(self):
        return {'datetime': self.datetime, 'lr_med': self.lr_med, 'r_med': self.r_med, 'l_med': self.l_med}


class ShoulderFlex:
    def __init__(self, l_med, r_med):
        now = datetime.now()  # current date and time
        self.datetime = now.strftime("%m/%d/%Y, %H:%M:%S")
        self.l_med = l_med
        self.r_med = r_med

    def asdict(self):
        return {'datetime': self.datetime, 'l_med': self.l_med, 'r_med': self.r_med}


class LumbarFlex:
    def __init__(self, l_med, r_med):
        now = datetime.now()  # current date and time
        self.datetime = now.strftime("%m/%d/%Y, %H:%M:%S")
        self.l_med = l_med
        self.r_med = r_med

    def asdict(self):
        return {'datetime': self.datetime, 'l_med': self.l_med, 'r_med': self.r_med}


class SideFlex:
    def __init__(self, l_med, r_med):
        now = datetime.now()  # current date and time
        self.datetime = now.strftime("%m/%d/%Y, %H:%M:%S")
        self.l_med = l_med
        self.r_med = r_med

    def asdict(self):
        return {'datetime': self.datetime, 'l_med': self.l_med, 'r_med': self.r_med}


class Tragus:
    def __init__(self, l_med, r_med):
        now = datetime.now()  # current date and time
        self.datetime = now.strftime("%m/%d/%Y, %H:%M:%S")
        self.l_med = l_med
        self.r_med = r_med

    def asdict(self):
        return {'datetime': self.datetime, 'l_med': self.l_med, 'r_med': self.r_med}


class Umovements:
    def __init__(self, email):
        self.email = email
        self.sit2stand = []
        self.hip_abduction = []
        self.cerv_rot = []
        self.hip_int_rot = []
        self.lumbar_flex = []
        self.shoulder_flex = []
        self.side_flex = []
        self.tragus = []

    def asdict(self):
        return {'email': self.email, 'sit2stand': self.hip_abduction, 'cerv_rot': self.cerv_rot,
                'hip_int_rot': self.hip_int_rot, 'lumber_flex': self.lumbar_flex, 'shoulder_flex': self.shoulder_flex,
                'side_flex': self.side_flex, 'tragus': self.tragus}


