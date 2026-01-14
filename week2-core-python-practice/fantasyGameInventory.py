def display_inventory(inventory):
    """
    Display a fantasy game inventory stored as a dictionary.

    Prints each item with its quantity and shows the total number of items
    in the inventory.
    """
    print("Inventory:")
    item_total = 0

    for item, count in inventory.items():
        print(f"{count} {item}")
        item_total += count

    print(f"Total number of items: {item_total}")

def add_to_inventory(inventory, added_items):
    """
    Add a list of items to a player's inventory dictionary.

    Parameters:
        inventory (dict): Current inventory with item names as keys and counts as values.
        added_items (list): List of items to add to the inventory.

    Returns:
        dict: Updated inventory with added items counted.
    """
    for item in added_items:
        if item in inventory:
            inventory[item] += 1
        else:
            inventory[item] = 1
    return inventory

if __name__ == "__main__":
    stuff = {
        'rope': 1,
        'torch': 6,
        'gold coin': 42,
        'dagger': 1,
        'arrow': 12
    }

    display_inventory(stuff)

    inv = {'gold coin': 42, 'rope': 1}
    dragon_loot = ['gold coin', 'dagger', 'gold coin', 'gold coin', 'ruby']

    inv = add_to_inventory(inv, dragon_loot)
    display_inventory(inv)
