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

    for mine in my_planets:
        targets.sort(key=lambda p: math.sqrt((p.x - mine.x)**2 + (p.y - mine.y)**2))
        for t in targets:
            if(t.ships >= mine.ships):
                continue
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

                dist2 = (closest_x - cx)**2 + (closest_y - cy)**2

                return dist2 <= r * r
            if(is_rotating(t)):
                continue

            dist = math.sqrt((mine.x - t.x)**2 + (mine.y - t.y)**2)
            angle = math.atan2(t.y - mine.y, t.x - mine.x)
            if(touches_sun(mine.x, mine.y, angle)==True):
                continue
            moves.append([mine.id, angle, t.ships+1])    

    return moves
