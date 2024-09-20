import pandas as pd
import numpy as np
import pybaseball as pyb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

pyb.cache.enable()

features_to_keep_main = [
    'pitch_type', 
    'game_date', 
    'release_speed', 
    'release_pos_x', 
    'release_pos_z', 
    'player_name', 
    'pitch_name',
    'events', 
    'zone', 
    'balls', 
    'strikes', 
    'game_year', 
    'pfx_x', 
    'pfx_z', 
    'plate_x', 
    'plate_z', 
    'outs_when_up', 
    'inning', 
    'inning_topbot', 
    'release_spin_rate', 
    'release_extension', 
    'delta_home_win_exp', 
    'delta_run_exp'
]

start_time_main = '2015-04-01'
end_time_main = '2020-07-15'


def get_pitchers_info(firstname, lastname, features_to_keep=features_to_keep_main, start_time=start_time_main, end_time=end_time_main):
    player_info = pyb.playerid_lookup(lastname, firstname)
    
    if player_info.empty:
        raise ValueError(f"No player found for name: {firstname} {lastname}")
    
    player_id = player_info['key_mlbam'].iloc[0]
    print(f'Pitcher ID: {player_id}')
    data = pyb.statcast_pitcher(start_time, end_time, player_id=player_id)
    filtered_data = data[features_to_keep]
    filtered_data = filtered_data.dropna()

    earlist = pd.to_datetime(sorted(filtered_data.game_date.unique())[0])
    latest = pd.to_datetime(sorted(filtered_data.game_date.unique())[-1])
    print(f'Loaded data for pitcher {firstname} {lastname} from {earlist} to {latest}')
    print(f'with {filtered_data.shape[0]} data points and {filtered_data.shape[1]} features')
    print()
    return filtered_data


def plot(df, columnName, playerName, pitch_type_filter=None):
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['date_num'] = df['game_date'].map(pd.Timestamp.toordinal)
    df['year'] = df['game_date'].dt.year

    plt.figure(figsize=(10, 6))

    # Define a color palette
    unique_pitch_types = df['pitch_name'].unique()
    palette = sns.color_palette("husl", len(unique_pitch_types))

    # Create a dictionary to map each pitch type to a color
    color_map = {pitch_type: color for pitch_type, color in zip(unique_pitch_types, palette)}

    if pitch_type_filter:
        # Filter dataframe for the specified pitch type
        df = df[df['pitch_name'] == pitch_type_filter]
        unique_pitch_types = [pitch_type_filter]

    # Plot each pitch type separately
    for pitch_type in unique_pitch_types:
        subset = df[df['pitch_name'] == pitch_type]
        sns.regplot(x='date_num', y=columnName, data=subset, scatter_kws={'s': 10, 'color': color_map[pitch_type]}, line_kws={'color': color_map[pitch_type]}, lowess=True, label=pitch_type)

    ax = plt.gca()
    # Set x-ticks to years
    years = df['year'].unique()
    ax.set_xticks([pd.Timestamp(f'{year}-01-01').toordinal() for year in years])
    ax.set_xticklabels(years, rotation=45)

    plt.xlabel('Year')
    plt.ylabel(columnName)
    plt.title(f'Average {columnName} of {playerName}')
    plt.legend(title='Pitch Type')

    plt.show()


def check_correlation(df, speed_column, spin_column, pitch_type_filter=None):

    if pitch_type_filter:
        df = df[df['pitch_name'] == pitch_type_filter]
    
    correlation, p_value = pearsonr(df[speed_column], df[spin_column])
    
    print(f'Pearson correlation coefficient: {correlation}')
    print(f'P-value: {p_value}')
    
    plt.figure(figsize=(10, 6))
    sns.regplot(x=speed_column, y=spin_column, data=df, scatter_kws={'s': 10}, line_kws={'color': 'red'})
    
    plt.xlabel('Release Speed')
    plt.ylabel('Release Spin Rate')
    plt.title(f'Release Speed vs Release Spin Rate for {pitch_type_filter if pitch_type_filter else "All Pitches"}\nCorrelation: {correlation:.2f}, P-value: {p_value:.2e}')
    
    plt.show()


firstname = 'Jacob'
lastname = 'deGrom'
player_name = 'Jacob deGrom'
pat_df = get_pitchers_info(firstname, lastname)


plot(pat_df, 'release_spin_rate', player_name, '4-Seam Fastball')
plot(pat_df, 'release_speed', player_name, '4-Seam Fastball')
check_correlation(pat_df, 'release_speed', 'release_spin_rate', pitch_type_filter='4-Seam Fastball')


def plot_pitch_movement(pat_df, years):
    all_data = []

    # Create subplots for individual year scatter plots
    fig_scatter, axes_scatter = plt.subplots(2, 3, figsize=(18, 12))
    axes_scatter = axes_scatter.flatten()

    for i, year in enumerate(years):
        # Filter data for each year
        yearly_data = pat_df[pat_df['game_year'] == year].copy()
        
        # Clean and transform the data
        yearly_data.loc[:, 'pfx_x_in_pv'] = -12 * yearly_data['pfx_x']
        yearly_data.loc[:, 'pfx_z_in'] = 12 * yearly_data['pfx_z']
        yearly_data.loc[:, 'year'] = year
        all_data.append(yearly_data)
        
        # Create dynamic pitch colors
        unique_pitches = yearly_data['pitch_name'].unique()
        pitch_colors = {
            "4-Seam Fastball": "red",
            "2-Seam Fastball": "blue",
            "Sinker": "cyan",
            "Cutter": "violet",
            "Fastball": "black",
            "Curveball": "green",
            "Knuckle Curve": "purple",
            "Slider": "orange",
            "Changeup": "#7f7f7f",
            "Split-Finger": "beige",
            "Knuckleball": "gold",
            "Intentional Ball": "black",
            "Pitch Out": "brown"
        }
        
        # Plot pitch movement for each year
        sns.scatterplot(ax=axes_scatter[i], data=yearly_data, x='pfx_x_in_pv', y='pfx_z_in', hue='pitch_name', palette=pitch_colors, alpha=0.25, s=60)
        axes_scatter[i].axvline(0, color='gray', linewidth=1)
        axes_scatter[i].axhline(0, color='gray', linewidth=1)
        axes_scatter[i].set_xlim(-25, 25)
        axes_scatter[i].set_ylim(-25, 25)
        axes_scatter[i].set_aspect('equal', adjustable='box')
        axes_scatter[i].set_title(f'Jacob deGrom Pitch Movement\n{year} MLB Season | Pitcher\'s POV')
        axes_scatter[i].set_xlabel('Horizontal Break (inches)')
        axes_scatter[i].set_ylabel('Induced Vertical Break (inches)')
        axes_scatter[i].legend(title='Pitch Name')

    # Combine all years data
    cumulative_data = pd.concat(all_data)

    # Plot cumulative pitch movement in the last subplot
    sns.scatterplot(ax=axes_scatter[-1], data=cumulative_data, x='pfx_x_in_pv', y='pfx_z_in', hue='pitch_name', palette=pitch_colors, alpha=0.25, s=60)
    axes_scatter[-1].axvline(0, color='gray', linewidth=1)
    axes_scatter[-1].axhline(0, color='gray', linewidth=1)
    axes_scatter[-1].set_xlim(-25, 25)
    axes_scatter[-1].set_ylim(-25, 25)
    axes_scatter[-1].set_aspect('equal', adjustable='box')
    axes_scatter[-1].set_title('Jacob deGrom Pitch Movement\n2015-2019 MLB Seasons | Pitcher\'s POV')
    axes_scatter[-1].set_xlabel('Horizontal Break (inches)')
    axes_scatter[-1].set_ylabel('Induced Vertical Break (inches)')
    axes_scatter[-1].legend(title='Pitch Name')

    plt.tight_layout()
    plt.show()

def fetch_degrom_data():
    # Fetch the player's ID
    player_lookup = pyb.playerid_lookup('deGrom', 'Jacob')
    degrom_id = player_lookup['key_mlbam'].values[0]

    # Define years to loop through
    years = [2015, 2016, 2017, 2018, 2019]
    all_data = []

    # Initialize an empty DataFrame to store cumulative results
    cumulative_pitch_counts = pd.DataFrame()

    for year in years:
        # Fetch the data for each year
        degrom_data = pyb.statcast_pitcher(f'{year}-03-01', f'{year}-12-01', degrom_id)
        
        # Clean and transform the data
        degrom_cleaned_data = degrom_data.dropna(subset=['pitch_name', 'type'])
        
        # Calculate ball and strike counts per pitch type
        pitch_counts = degrom_cleaned_data.groupby(['pitch_name', 'type']).size().unstack(fill_value=0)
        pitch_counts['Total Count'] = pitch_counts.sum(axis=1)
        pitch_counts['Strike Ratio'] = pitch_counts['S'] / pitch_counts['Total Count']
        pitch_counts.reset_index(inplace=True)
        
        # Add year information
        pitch_counts['Year'] = year
        all_data.append(pitch_counts)
        
        # Append to cumulative results using pd.concat
        cumulative_pitch_counts = pd.concat([cumulative_pitch_counts, pitch_counts], ignore_index=True)

    # Combine all years data
    cumulative_data = pd.concat(all_data)

    return cumulative_pitch_counts, cumulative_data



years = [2015, 2016, 2017, 2018, 2019]

plot_pitch_movement(pat_df, years)


def fetch_degrom_data():
    # Fetch the player's ID
    player_lookup = pyb.playerid_lookup('deGrom', 'Jacob')
    degrom_id = player_lookup['key_mlbam'].values[0]

    # Define years to loop through
    years = [2015, 2016, 2017, 2018, 2019]
    all_data = []

    # Initialize an empty DataFrame to store cumulative results
    cumulative_pitch_counts = pd.DataFrame()

    for year in years:
        # Fetch the data for each year
        degrom_data = pyb.statcast_pitcher(f'{year}-03-01', f'{year}-12-01', degrom_id)
        
        # Clean and transform the data
        degrom_cleaned_data = degrom_data.dropna(subset=['pitch_name', 'type'])
        
        # Calculate ball and strike counts per pitch type
        pitch_counts = degrom_cleaned_data.groupby(['pitch_name', 'type']).size().unstack(fill_value=0)
        pitch_counts['Total Count'] = pitch_counts.sum(axis=1)
        pitch_counts['Strike Ratio'] = pitch_counts['S'] / pitch_counts['Total Count']
        pitch_counts.reset_index(inplace=True)
        
        # Add year information
        pitch_counts['Year'] = year
        all_data.append(pitch_counts)
        
        # Append to cumulative results using pd.concat
        cumulative_pitch_counts = pd.concat([cumulative_pitch_counts, pitch_counts], ignore_index=True)

    # Combine all years data
    cumulative_data = pd.concat(all_data)

    return cumulative_pitch_counts, cumulative_data


def plot_strike_ratio(cumulative_pitch_counts, pitch_type_filter=None):
    # Prepare data for plotting Strike Ratio by year
    plot_data = cumulative_pitch_counts[['pitch_name', 'Strike Ratio', 'Year']]

    # Apply pitch type filter if provided
    if pitch_type_filter:
        plot_data = plot_data[plot_data['pitch_name'] == pitch_type_filter]
        plot_data_pivot = plot_data.pivot(index='Year', columns='pitch_name', values='Strike Ratio')
    else:
        # Pivot the data to get pitch types as columns
        plot_data_pivot = plot_data.pivot(index='Year', columns='pitch_name', values='Strike Ratio')

    # Define a consistent color palette
    pitch_types = plot_data_pivot.columns
    colors = sns.color_palette('tab20', len(pitch_types))
    color_map = dict(zip(pitch_types, colors))

    # Create plot
    plt.figure(figsize=(12, 8))
    for column in plot_data_pivot.columns:
        plt.plot(plot_data_pivot.index, plot_data_pivot[column], marker='o', label=column, color=color_map[column])
    
    plt.title('Strike Ratio by Pitch Type and Year for Jacob deGrom (2015-2019)')
    plt.xlabel('Year')
    plt.ylabel('Strike Ratio')
    plt.legend(title='Pitch Type')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Fetch data
cumulative_pitch_counts, cumulative_data = fetch_degrom_data()

# Plot Strike Ratio without filter
plot_strike_ratio(cumulative_pitch_counts)

# Plot Strike Ratio with filter
plot_strike_ratio(cumulative_pitch_counts, pitch_type_filter='4-Seam Fastball')



player_lookup = pyb.playerid_lookup('deGrom', 'Jacob')
degrom_id = player_lookup['key_mlbam'].values[0]

years = [2015, 2016, 2017, 2018, 2019]
overall_ratios = []

for year in years:
    degrom_data = pyb.statcast_pitcher(f'{year}-03-01', f'{year}-12-01', degrom_id)

    degrom_cleaned_data = degrom_data.dropna(subset=['type'])

    total_counts = degrom_cleaned_data['type'].value_counts()
    bad_ball_count = total_counts.get('B', 0) + total_counts.get('X', 0)
    strike_count = total_counts.get('S', 0)
    total_count = bad_ball_count + strike_count

    bad_ball_ratio = bad_ball_count / total_count if total_count != 0 else 0
    strike_ratio = strike_count / total_count if total_count != 0 else 0

    overall_ratios.append({
        'Year': year,
        'Bad Ball Ratio': bad_ball_ratio,
        'Strike Ratio': strike_ratio
    })

overall_ratios_df = pd.DataFrame(overall_ratios)

print("Overall Bad Ball and Strike Ratios by Year")
print(overall_ratios_df)


all_data = []


cumulative_pitch_counts = pd.DataFrame()

for year in years:
    degrom_data = pyb.statcast_pitcher(f'{year}-03-01', f'{year}-12-01', degrom_id)

    degrom_cleaned_data = degrom_data.dropna(subset=['pitch_name', 'type'])

    pitch_counts = degrom_cleaned_data.groupby(['pitch_name', 'type']).size().unstack(fill_value=0)
    pitch_counts['Total Count'] = pitch_counts.sum(axis=1)
    pitch_counts['Bad Ball Ratio'] = (pitch_counts['B'] + pitch_counts['X']) / pitch_counts['Total Count']
    pitch_counts['Strike Ratio'] = pitch_counts['S'] / pitch_counts['Total Count']
    pitch_counts.reset_index(inplace=True)

    pitch_counts.columns.name = None
    pitch_counts.rename(columns={'B': 'Ball Count', 'S': 'Strike Count'}, inplace=True)

    pitch_counts['Year'] = year
    all_data.append(pitch_counts)

    print(f"Ball and Strike Ratio for {year}")
    print(pitch_counts)
    print("\n")

    cumulative_pitch_counts = pd.concat([cumulative_pitch_counts, pitch_counts], ignore_index=True)

cumulative_counts = cumulative_pitch_counts.groupby(['pitch_name']).sum()
cumulative_counts['Bad Ball Ratio'] = (cumulative_counts['Ball Count'] + cumulative_counts['X']) / cumulative_counts['Total Count']
cumulative_counts['Strike Ratio'] = cumulative_counts['Strike Count'] / cumulative_counts['Total Count']
cumulative_counts.reset_index(inplace=True)

print("Cumulative Ball and Strike Ratio from 2015 to 2019")
print(cumulative_counts)
