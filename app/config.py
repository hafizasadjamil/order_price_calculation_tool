import os

TAX_RATE: float = float(os.getenv("TAX_RATE", "0.06"))
#configuration
MONGODB_URI = os.getenv(
    "MONGODB_URI")
DB_NAME      = os.getenv("DB_NAME", "voice_agents")
MENU_COLL    = os.getenv("MENU_COLLECTION", "menus")
