import pandas as pd

df = pd.read_csv("mancala_simulations.csv")

# Overall win rates
win_rates = df.groupby(['Player1_Strategy', 'Player2_Strategy'])['Winner'].value_counts(normalize=True)
print(win_rates)

# Average game duration by strategy pair
avg_times = df.groupby(['Player1_Strategy', 'Player2_Strategy'])['Time_Seconds'].mean()
print(avg_times)

# Score differences
df['Score_Diff'] = df['Player1_Score'] - df['Player2_Score']
score_stats = df.groupby(['Player1_Strategy', 'Player2_Strategy'])['Score_Diff'].describe()
print(score_stats)