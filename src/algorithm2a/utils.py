import math

# radius of the Earth
R = 6373.0


def compute_distance(place_1, place_2):
    """
    compute the distance between two lat-long coordinates

    Input:

    place_1 - tuple : (latitude, longitude) of place 1
    place_2 - tuple : (latitude, longitude) of place 2

    Returns:
    distance - float number
    """

    distance_lat = place_2[0] - place_1[0]
    distance_long = place_2[1] - place_1[1]

    # Haverinse formula
    x = math.sin(distance_lat / 2) ** 2 + math.cos(place_1[0]) * math.cos(place_2[0]) * math.sin(distance_long / 2) ** 2

    y = 2 * math.atan2(math.sqrt(x), math.sqrt(1 - x))

    return R * y
