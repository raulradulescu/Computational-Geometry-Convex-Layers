from typing import List, Tuple, Optional, Set, Callable
import matplotlib.pyplot as plt
import random
import numpy as np
import time

Point = Tuple[float, float]
number_of_points = 10000
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

    points = sorted(points)  # Ensure consistent ordering for leftmost selection
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

    # Find the point with the lowest y-coordinate (and x if tie)
    pivot = min(points, key=lambda p: (p[1], p[0]))

    # Sort by polar angle and distance
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
    points = list(set(points))  #Remove duplicates
    if len(points) < 3:
        return points.copy()

    hull_points = set()

    def cross(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0])*(b[1] - o[1]) - (a[1] - o[1])*(b[0] - o[0])

    def quick_hull(pts: List[Point], p: Point, q: Point):
        if not pts:
            return
        # Find the farthest point from pq
        farthest = max(pts, key=lambda x: abs(cross(p, q, x)))
        hull_points.add(farthest)
        # Partition points into left of p-farthest and left of farthest-q
        left_p = [x for x in pts if cross(p, farthest, x) > 0]
        left_q = [x for x in pts if cross(farthest, q, x) > 0]
        quick_hull(left_p, p, farthest)
        quick_hull(left_q, farthest, q)

    # Find initial leftmost and rightmost points
    leftmost = min(points, key=lambda x: x[0])
    rightmost = max(points, key=lambda x: x[0])
    hull_points.update([leftmost, rightmost])

    # Split points into upper and lower
    upper = [x for x in points if cross(leftmost, rightmost, x) > 0]
    lower = [x for x in points if cross(leftmost, rightmost, x) < 0]

    quick_hull(upper, leftmost, rightmost)
    quick_hull(lower, rightmost, leftmost)

    # Order the collected hull points using Monotone Chain
    return convex_hull_monotone_chain(list(hull_points))

# Compute convex layers efficiently
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

# Plot convex layers with improved styling
def plot_convex_layers(ax, layers: List[List[Point]], points: List[Point], title: str):
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    colors = plt.cm.tab20(np.linspace(0, 1, len(layers)))
    for i, layer in enumerate(layers):
        closed_layer = layer + [layer[0]]  # Close the hull
        ax.plot(*zip(*closed_layer), color=colors[i], linewidth=1.5, alpha=0.8)
    ax.scatter(*zip(*points), s=20, c='black', alpha=0.6)


# Compare algorithms with timing and visualization
def compare_algorithms(points: List[Point]):
    algorithms = {
        "Jarvis March": convex_hull_jarvis,
        "Graham's Scan": convex_hull_graham,
        "Monotone Chain": convex_hull_monotone_chain,
        "Quickhull": convex_hull_quickhull,
    }

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Convex Hull Algorithm Comparison for {number_of_points} points", fontsize=14)

    for ax, (name, algo) in zip(axs.flatten(), algorithms.items()):
        start_time = time.perf_counter()
        layers = compute_convex_layers(points, algo)
        elapsed = time.perf_counter() - start_time
        plot_convex_layers(ax, layers, points, f"{name}\nTime: {elapsed:.9f}s")

    plt.tight_layout()
    plt.show()

# Generate points and compare
points = generate_points(number_of_points)
compare_algorithms(points)