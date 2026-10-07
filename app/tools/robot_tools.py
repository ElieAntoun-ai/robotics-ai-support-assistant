import sqlite3
from langchain_core.tools import tool

@tool
def get_robot_status(robot_id: str):
    """
    Retrieve current operational data for a specific robot,
    including battery level, operating status, and error code.
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

    if result is None:
        return {
            "error": f"Robot {robot_id} not found"
        }

    return {
        "robot_id": result[0],
        "battery_level": result[1],
        "status": result[2],
        "error_code": result[3]
    }


