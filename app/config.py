import os

TAX_RATE: float = float(os.getenv("TAX_RATE", "0.06"))

MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb+srv://hasadjamil_db_user:aF35vqPHKS0wSQ9I@cluster0.fzfg52u.mongodb.net/portfolioDB?retryWrites=true&w=majority&appName=Cluster0"
)
DB_NAME      = os.getenv("DB_NAME", "voice_agents")
MENU_COLL    = os.getenv("MENU_COLLECTION", "menus")
