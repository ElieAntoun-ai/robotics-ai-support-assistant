import sqlite3
from langchain_core.tools import tool

@tool
def get_robot_status(robot_id: str):
    """
    Get the current operational status of a robot using its robot ID.
    """

    connection = sqlite3.connect(
        "data/operational/robot_data.db"
    )

    cursor = connection.cursor()

    cursor.execute(
    "SELECT * FROM robots WHERE robot_id = ?",
    (robot_id,)
)
    result = cursor.fetchone()
    connection.close()
    return result



