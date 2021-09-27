from flask_pymongo import pymongo
from model import *
from bson.json_util import dumps
from bson.json_util import loads

CONNECTION_STRING = "STRING"
client = pymongo.MongoClient(CONNECTION_STRING)
db = client.get_database('iuk')

testDB = db.testn
userDB = db.users


def add_movement_record(movement, data, email):
    global status
    status = 0
    try:
        userDB.update_one({'email': email}, {'$push': {movement: data}})
        # cursor = list(userDB.find())
        # print(loads(dumps(cursor)))
        status = 1
    except Exception as e:
        print("Update DB error: "+str(e))
        status = 0
    return status
