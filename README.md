# Computational-Geometry-Convex-Layers

A collection of Python scripts for comparing different convex hull algorithms' performance and visualizing their results.

Scripts:

```conv_hull_visualiser.py``` - Visualizes how different algorithms compute convex hulls

```csv_generator.py``` - Generates timing data for algorithm comparisons.

```points_visualiser.py``` - Creates statistical plots from the timing data.

Prerequisites:

```pip install numpy matplotlib pandas scipy```

Usage

Generate timing data:


```python csv_generator.py```

This will create CSV files with timing data for different numbers of points (10, 100, 1000, 10000).


Visualize the algorithms in action:

```python conv_hull_visualiser.py```

Shows an interactive visualization of how each algorithm computes convex hulls.

Analyze performance:

```python points_visualiser.py 10_points.csv```

Creates bar plots showing mean execution times and standard deviations for each algorithm.


Supported Algorithms: Jarvis March (Gift Wrapping), Graham's Scan, Monotone Chain, Quickhull

Output:

CSV files with timing data;

PNG plots showing performance comparisons;

Interactive visualizations of the convex hull computation process;

The scripts can be modified to test different numbers of points by changing number_of_points in the respective files.
