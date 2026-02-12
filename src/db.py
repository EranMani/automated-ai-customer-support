"""
    This module contains the mock database data for simulating a customer support system
"""

class MockDB():
    """
        Mock database class that includes sample user and order data
    """
    def __init__(self):
        self.users = {
            "user1@gmail.com": {
                "tier": "Free"
            },
            "user2@gmail.com": {
                "tier": "Premium"
            }
        }

        self.orders = {
            "#123": {
                "user": "user1@gmail.com",
                "price": 100.0,
                "description": "A case for an android phone",
                "status": "WIP"
            },
            "#124": {
                "user": "user2@gmail.com",
                "price": 200.0,
                "description": "A case for an iphone",
                "status": "Refund Processing"
            }
        }

    def get_order_status(self, order_id) -> str:
        """Fetch the status of an order from the database"""
        return self.orders[order_id]["status"]

    def get_user_tier(self, email: str) -> str:
        """Fetch the tier level of a user from the database"""
        return self.users[email]["tier"]
