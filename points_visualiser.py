import pandas as pd
import matplotlib.pyplot as plt
import sys

filename = "10_points.csv"
def process_csv(filename):
    # Read the CSV file
    df = pd.read_csv(filename)

    # Calculate mean and standard deviation for each algorithm
    means = df.mean()
    stds = df.std()

    # Print the results
    print("Mean execution times (seconds):")
    print(means.to_string())
    print("\nStandard deviations:")
    print(stds.to_string())

    # Plotting the results
    plt.figure(figsize=(10, 6))
    bars = plt.bar(means.index, means, yerr=stds, capsize=5, alpha=0.7)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title(f'Performance Comparison of Convex Hull Algorithms ({filename})', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout and save the plot
    plt.tight_layout()
    plot_filename = filename.replace('.csv', '_plot.png')
    plt.savefig(plot_filename)
    print(f"\nPlot saved as '{plot_filename}'")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze.py <csv_filename>")
        sys.exit(1)
    csv_file = sys.argv[1]
    process_csv(csv_file)