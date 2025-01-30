from typing import List, Tuple, Callable
import csv
import random
import time
import numpy as np

Point = Tuple[float, float]

# Configuration
number_of_points = 10000  # Change this value as needed
number_of_runs = 100      # Change this value as needed (y rows)
output_filename = f"{number_of_points}_points.csv"

# Helper function to find orientation of the triplet
def orientation(p: Point, q: Point, r: Point) -> int:
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0  # Collinear
    return 1 if val > 0 else 2  # Clockwise or Counterclockwise

# Gift Wrapping (Jarvis March)
def convex_hull_jarvis(points: List[Point]) -> List[Point]:
    n = len(points)
    if n < 3:
        return points.copy()

    points = sorted(points)
    start = points[0]
    hull = [start]
    current = start

    while True:
        next_point = points[0]
        for p in points[1:]:
            o = orientation(current, next_point, p)
            if o == 2 or (o == 0 and (
                (p[0] - current[0])**2 + (p[1] - current[1])**2 >
                (next_point[0] - current[0])**2 + (next_point[1] - current[1])**2
            )):
                next_point = p
        if next_point == start:
            break
        hull.append(next_point)
        current = next_point

    return hull

# Graham's Scan
def convex_hull_graham(points: List[Point]) -> List[Point]:
    if len(points) <= 1:
        return points.copy()

    def polar_angle(o: Point, p: Point) -> float:
        return np.arctan2(p[1]-o[1], p[0]-o[0])

    pivot = min(points, key=lambda p: (p[1], p[0]))
    sorted_points = sorted(points, key=lambda p: (polar_angle(pivot, p), -p[1], p[0]))

    hull = []
    for p in sorted_points:
        while len(hull) >= 2 and orientation(hull[-2], hull[-1], p) != 2:
            hull.pop()
        hull.append(p)
    return hull

# Andrew's Monotone Chain
def convex_hull_monotone_chain(points: List[Point]) -> List[Point]:
    points = sorted(points)
    if len(points) <= 1:
        return points.copy()

    lower = []
    for p in points:
        while len(lower) >= 2 and orientation(lower[-2], lower[-1], p) != 2:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and orientation(upper[-2], upper[-1], p) != 2:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]

# Quickhull with Monotone Chain ordering
def convex_hull_quickhull(points: List[Point]) -> List[Point]:
    points = list(set(points))  # Remove duplicates
    if len(points) < 3:
        return points.copy()

    hull_points = set()

    def cross(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0])*(b[1] - o[1]) - (a[1] - o[1])*(b[0] - o[0])

    def quick_hull(pts: List[Point], p: Point, q: Point):
        if not pts:
            return
        farthest = max(pts, key=lambda x: abs(cross(p, q, x)))
        hull_points.add(farthest)
        left_p = [x for x in pts if cross(p, farthest, x) > 0]
        left_q = [x for x in pts if cross(farthest, q, x) > 0]
        quick_hull(left_p, p, farthest)
        quick_hull(left_q, farthest, q)

    leftmost = min(points, key=lambda x: x[0])
    rightmost = max(points, key=lambda x: x[0])
    hull_points.update([leftmost, rightmost])

    upper = [x for x in points if cross(leftmost, rightmost, x) > 0]
    lower = [x for x in points if cross(leftmost, rightmost, x) < 0]

    quick_hull(upper, leftmost, rightmost)
    quick_hull(lower, rightmost, leftmost)

    return convex_hull_monotone_chain(list(hull_points))

# Compute convex layers
def compute_convex_layers(points: List[Point], algorithm: Callable) -> List[List[Point]]:
    remaining = points.copy()
    layers = []
    while len(remaining) >= 3:
        hull = algorithm(remaining)
        if len(hull) < 3:
            break
        hull_set = set(hull)
        remaining = [p for p in remaining if p not in hull_set]
        layers.append(hull)
    return layers

# Generate unique random points
def generate_points(n: int) -> List[Point]:
    points = set()
    while len(points) < n:
        points.add((random.randint(0, 100), random.randint(0, 100)))
    return list(points)

# Main function to generate timing data
def generate_timing_data():
    algorithms = {
        "Jarvis March": convex_hull_jarvis,
        "Graham's Scan": convex_hull_graham,
        "Monotone Chain": convex_hull_monotone_chain,
        "Quickhull": convex_hull_quickhull,
    }

    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(algorithms.keys())  # Write header

        for _ in range(number_of_runs):
            points = generate_points(number_of_points)
            row = []

            for algo in algorithms.values():
                start_time = time.perf_counter()
                compute_convex_layers(points, algo)
                elapsed = time.perf_counter() - start_time
                row.append(f"{elapsed:.9f}")

            writer.writerow(row)

if __name__ == "__main__":
    generate_timing_data()
    print(f"Generated timing data for {number_of_runs} runs with {number_of_points} points in {output_filename}")