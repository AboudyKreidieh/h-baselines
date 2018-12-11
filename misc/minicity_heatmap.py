# Saves mean vehicle speed heatmap plots of varying granularity from minicity csv
# Can customize time range and heatmap granularities to try

# WARNING: There is a pd.pivot_table() bug in Pandas!!
# Bug mixes up values and sorting for Categorical groupbys like pd.cut() bins.
# Run pip install --upgrade pandas if the heatmap
# pixels generated are all scrambled/ out of order.
# Verified bug in Pandas version 0.23.0
# Verified working in Pandas version 0.23.4

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Global constants. Change them here
T_START = 0 # Inclusive.
T_END = -1  # Inclusive. Set to -1 to include all time steps to end of run
NUM_OF_TILES = [25, 50, 100, 150]  # Plot hyperparameter. Granularity of tiles, 
								   # splits each axes into 25, 50, 100, etc. heat tiles
USE_LOG_SCALE = False  # Use log scale to plot mean vehicle speeds. Looks worse.

CSV_PATH = "minicity_20181210-1606311544486791.0742002-emission.csv"
X_MIN = -8  # Hardcoded Minicity (x,y) bounds. 
X_MAX = 225
Y_MIN = -8
Y_MAX = 225


# Generates and saves heatmap plots from csv_path of the average speed of vehicles
# for each granularity listed in num_of_tiles,
# each plot averaging across all data from time t_start to t_end.
#
# ARGS
# csv_path: string. Location of csv of minicity run
# t_start: int. Inclusive start of time range to average over
# t_end: int. Inclusive end of time range to average over. 
#		 Set to -1 to include all time steps to end of run.
#		 Set to t_start to show speed for single time step.
# num_of_tiles: list of ints. Each granularity hyperparameter value to plot
# log_scale: boolean. Whether to plot mean average vehicle speed using log scale.
def csv_heatmap(csv_path, t_start, t_end, num_of_tiles, log_scale=False):

	print("WARNING: Pandas has a bug that outputs incorrect results.")
	print("Please update Pandas to >=0.23.4 before running.\n")

	df = pd.read_csv(csv_path)
	df = df[['time', 'id', 'x', 'y', 'speed']]

	# Drop data from time steps not in range
	if t_start > 0:
		df = df[df['time'] >= t_start]
	if t_end != -1 and t_end >= t_start:
		df = df[df['time'] <= t_start]

	# Add log scale column as alternative. Honestly result looks worse.
	df['log speed'] = np.log(df['speed'])

	for dim in num_of_tiles:

		# Generate new columns labeling each piece of data based on x,y bins
		x_bins = np.linspace(X_MIN, X_MAX, dim)
		y_bins = np.linspace(Y_MIN, Y_MAX, dim)
		df['x_bin'] = pd.cut(df['x'], x_bins)
		df['y_bin'] = pd.cut(df['y'], y_bins)
		df['x_bin_min'] = df['x_bin'].apply(lambda interval: interval.left)
		df['y_bin_min'] = df['y_bin'].apply(lambda interval: interval.left)

		# Create a pivot table based on x,y bins
		if not log_scale:
			df_pivot = df.pivot_table(index='y_bin_min', columns='x_bin_min', \
								  	  values='speed', aggfunc=np.mean)
		else:
			df_pivot = df.pivot_table(index='y_bin_min', columns='x_bin_min', \
								  	  values='log speed', aggfunc=np.mean)

		# Fill NaNs with 0 for areas with no cars recorded
		df_pivot = df_pivot.fillna(0)

		# Sort pivot table so that (0,0) is at bottom left and axes are correct
		df_pivot.sortlevel(axis=0, ascending=False, sort_remaining=False, inplace=True)
		df_pivot.sortlevel(axis=1, ascending=True, sort_remaining=False, inplace=True)

		# Print preview of heatmap values
		print(df_pivot.head())

		# Format and export heatmap as png. Hides bin labels along axes, since it gets crowded
		fig=plt.figure()
		ax=plt.subplot(111, aspect='equal')
		sns.heatmap(df_pivot, vmin=0.0, cbar_kws={'label': 'Mean Vehicle Speed'}, \
				    xticklabels=False, yticklabels=False, )
		plt.title("Heatmap of Minicity Mean Vehicle Speeds\n" + \
				  "from t={} to t={}, granularity={} x {}".format(t_start, t_end, dim, dim))
		plt.xlabel("x")
		plt.ylabel("y")
		plt.tight_layout()
		if not log_scale:
			filename = 'heatmap_gran{}_tstart{}_tend{}.png'.format(dim, t_start, t_end)
		else:
			filename = 'heatmap_log_gran{}_tstart{}_tend{}.png'.format(dim, t_start, t_end)
		fig.savefig(filename)
		plt.close(fig)


if __name__ == '__main__':
	csv_heatmap(CSV_PATH, T_START, T_END, NUM_OF_TILES, USE_LOG_SCALE)

