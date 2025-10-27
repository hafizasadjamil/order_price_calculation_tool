#db file
from pymongo import MongoClient
import certifi
from app.config import MONGODB_URI, DB_NAME, MENU_COLL

client = MongoClient(
    MONGODB_URI,
    tls=True,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=30000,
)

db     = client[DB_NAME]
menus  = db[MENU_COLL]
