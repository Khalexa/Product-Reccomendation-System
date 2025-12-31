# backend/interaction.py

def log_interaction(user_id, product_id, interaction_type):
    #Simulate logging a user interaction.
    interaction = {
        "user_id": user_id,
        "product_id": product_id,
        "interaction": interaction_type
    }
    return interaction
