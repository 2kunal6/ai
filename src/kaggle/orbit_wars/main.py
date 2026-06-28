import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet


def nearest_planet_sniper(obs):
    SUN_X, SUN_Y, SUN_RADIUS = 50, 50, 50
    '''def predict_position(planet, omega, distance_to_planet):
        t = distance_to_planet / obs.angular_velocity

        r = math.hypot(planet.x - SUN_X, planet.y - SUN_Y)
        theta = math.atan2(planet.y - SUN_Y, planet.x - SUN_X)

        future_theta = theta + omega * t

        x = SUN_X + r * math.cos(future_theta)
        y = SUN_Y + r * math.sin(future_theta)

        return x, y'''

    def is_rotating(planet):
        orbital_radius = math.hypot(
            planet.x - SUN_X,
            planet.y - SUN_Y
        )
        return orbital_radius + planet.radius < SUN_RADIUS

    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    planets = [Planet(*p) for p in raw_planets]

    # Separate our planets from targets
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]

    if not targets:
        return moves

    def touches_sun(x, y, a):
        cx, cy = 50, 50
        r = 10
        dx = math.cos(a)
        dy = math.sin(a)
        vx = cx - x
        vy = cy - y

        t = vx * dx + vy * dy

        if t < 0:
            return False

        closest_x = x + t * dx
        closest_y = y + t * dy

        dist2 = (closest_x - cx) ** 2 + (closest_y - cy) ** 2

        return dist2 <= r * r

    for mine in my_planets:
        targets.sort(key=lambda p: math.sqrt((p.x - mine.x)**2 + (p.y - mine.y)**2))
        max_ships = 0
        max_ships_t = None
        for t in targets:
            dist = math.sqrt((mine.x - t.x) ** 2 + (mine.y - t.y) ** 2)
            if (dist > 20):
                break
            if(t.ships >= mine.ships):
                continue
            #if(is_rotating(t)):
                #continue
            if(t.ships > max_ships):
                max_ships_t = t
                max_ships = t.ships
        if(max_ships_t==None):
            min_dist = float('inf')
            for t in targets:
                dist = math.sqrt((mine.x - t.x) ** 2 + (mine.y - t.y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    nearest = t

            if nearest is None:
                continue

            # How many ships do we need? Target's garrison + 1
            ships_needed = max(nearest.ships + 1, 20)

            # Only send if we have enough
            if mine.ships >= ships_needed:
                # Calculate angle from our planet to the target
                angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
                moves.append([mine.id, angle, ships_needed])
        else:
            angle = math.atan2(max_ships_t.y - mine.y, max_ships_t.x - mine.x)
            #if(touches_sun(mine.x, mine.y, angle)==True):
                #continue
            moves.append([mine.id, angle, max_ships_t.ships+1])

    return moves
