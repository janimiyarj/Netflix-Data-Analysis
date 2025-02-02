"""""
Columns and Descriptions:
- show_id: Unique identifier for each show (s1, s2).
- type: Specifies whether the title is a "Movie" or "TV Show".
- title: The name of the Netflix title.
- director: The director of the title
- cast: The main actors involved in the title.
- country: The country where the title was produced.
- date_added: The date when the title was added to Netflix.
- release_year: The year the title was originally released.
- rating: The content rating ("PG-13", "TV-MA").
- duration: Duration of the movie (in minutes) or the number of seasons for TV shows.
- listed_in: Categories or genres the title falls under ("Documentaries", "TV Dramas").
- description: The summary description

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

# Load the dataset
df = pd.read_csv('/Users/janimiyashaik/Desktop/My Projects/Netflix_Movies_Titles_Data_Analysis/netflix_titles.csv')

# Display dataset info
print("First 5 rows of the dataset:")
print(df.head())
print("__________________________________________________________________")
print("\nColumn names:")
print(df.columns)
print("__________________________________________________________________")
print("\nMissing values per column:")
print(df.isnull().sum())
print("__________________________________________________________________")
print("\nDataset shape:")
print(df.shape)
print("__________________________________________________________________")
print("\nDataset Info:")
print(df.info())
print("__________________________________________________________________")
print("\nStatistical Summary of Numerical Columns:")
print(df.describe())
print("__________________________________________________________________")
# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Fill missing values
for col in numerical_cols:
    df[col].fillna(df[col].mean(), inplace=True)

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Handle the 'date_added' column safely
df['date_added'] = df['date_added'].fillna("Unknown, Unknown")

# Function to clean and split the 'date_added' column
def extract_date_components(date_str):
    try:
        if date_str == "Unknown, Unknown":
            return ("Unknown", np.nan, np.nan)
        parts = date_str.split(",")
        year = int(parts[1].strip())
        month_day = parts[0].split(" ")
        month = month_day[0]
        day = int(month_day[1].strip())
        return (month, day, year)
    except Exception:
        return ("Unknown", np.nan, np.nan)

# Apply the function to extract components
df[['Month', 'Date_of_Month', 'Year']] = df['date_added'].apply(extract_date_components).apply(pd.Series)

# Map months to numerical values
month_mapping = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
    "Unknown": 0  # Handling cases where the month is unknown
}

# EDA: Number of movies/shows by month added
print("\nNumber of movies/shows by month added:")
print(df['Month'].value_counts().sort_values(ascending=False))

df['Month'] = df['Month'].map(month_mapping)

print("__________________________________________________________________")
# Ensure the new columns are numerical
df['Date_of_Month'] = df['Date_of_Month'].fillna(0).astype(int)
df['Year'] = df['Year'].fillna(0).astype(int)

print("__________________________________________________________________")
# Display the updated DataFrame
print("\nUpdated DataFrame with extracted date components:")
print(df[['date_added', 'Month', 'Date_of_Month', 'Year']].head())

print("__________________________________________________________________")
# EDA: Number of movies/shows by year added
print("\nNumber of movies/shows by year added:")
print(df['Year'].value_counts().sort_values(ascending=False))

print("__________________________________________________________________")
# EDA: Number of movies/shows by month added
print("\nNumber of movies/shows by month added:")
print(df['Month'].value_counts().sort_values(ascending=False))

print("__________________________________________________________________")
# EDA: Countries mostly used
print("\nCountries mostly used:")
print(df['country'].value_counts().sort_values(ascending=False).head(10))

print("__________________________________________________________________")
# EDA: Type of shows
print("\nTypes of shows (Movies/TV Shows):")
print(df['type'].value_counts())

print("__________________________________________________________________")
#EDA: Top 10 dates on which normally the shows are seen
print("\nTop 10 dates of a Month on which normally the shows are seen:")
print(df['Date_of_Month'].value_counts().sort_values(ascending=False).head(10))

print("__________________________________________________________________")

# Group by type and count
type_counts = df.groupby('type').size()
print("\nNumber of Movies and TV Shows:")
print(type_counts.sort_values(ascending=False))

print("__________________________________________________________________")

# Group by year and count titles
titles_per_year = df.groupby('Year').size().sort_values(ascending=False)
print("\nNumber of titles added per year:")
print(titles_per_year)

print("__________________________________________________________________")

# Group by country and count titles
titles_per_country = df.groupby('country').size().sort_values(ascending=False).head(10)
print("\nTop 10 countries with the most titles:")
print(titles_per_country)

print("__________________________________________________________________")

# Group by month and count titles
titles_per_month = df.groupby('Month').size().sort_values(ascending=False)
print("\nNumber of titles added per month:")
print(titles_per_month.sort_values(ascending=False))

print("__________________________________________________________________")
# Group by year and type, then unstack and sort
type_year_counts = df.groupby(['Year', 'type']).size().unstack()

# Sorting rows (by Year) in ascending order
type_year_counts_sorted_by_year = type_year_counts.sort_index()

print("\nMovies and TV Shows added per year (sorted by Year):")
print(type_year_counts_sorted_by_year)

# Sorting columns (alphabetically by type)
type_year_counts_sorted_by_type = type_year_counts.sort_index(axis=1)

print("\nMovies and TV Shows added per year (sorted by type):")
print(type_year_counts_sorted_by_type)

# Sorting by total count across both types
type_year_counts['Total'] = type_year_counts.sum(axis=1)  # Add a total column for sorting
type_year_counts_sorted_by_total = type_year_counts.sort_values(by='Total', ascending=False)

print("\nMovies and TV Shows added per year (sorted by total count):")
print(type_year_counts_sorted_by_total.drop(columns='Total'))  # Remove the helper column for display

print("__________________________________________________________________")

# Group by country and type to count movies and TV shows for each country
country_type_counts = df.groupby(['country', 'type']).size().unstack().fillna(0)
country_type_counts['Total'] = country_type_counts.sum(axis=1)
country_type_counts_sorted_by_total = country_type_counts.sort_values(by='Total', ascending=True)

print("\nMovies and TV Shows per country (sorted by total count of Type):")
print(country_type_counts_sorted_by_total.head(10))

country_type_counts.drop(columns='Total', inplace=True)
print("\nMovies and TV Shows per country:")
print(country_type_counts.head(10))  # Display top 10 countries

print("__________________________________________________________________")

print(df.head(10))

# Count of Movies vs TV Shows
type_counts = df['type'].value_counts()

plt.figure(figsize=(6, 4))
sns.barplot(x=type_counts.index, y=type_counts.values, palette='viridis')
plt.title('Distribution of Movie and TV Show Titles')
plt.xlabel('Type')
plt.ylabel('Count')
plt.show()

# Titles added per year
titles_per_year = df['Year'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
sns.lineplot(x=titles_per_year.index, y=titles_per_year.values, marker='o', color='blue')
plt.title('Number of Titles Added Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Titles')
plt.grid(True)
plt.show()


# Top 10 countries with the most titles
top_countries = df['country'].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_countries.values, y=top_countries.index, palette='coolwarm')
plt.title('Top 10 Countries with Most Titles')
plt.xlabel('Number of Titles')
plt.ylabel('Country')
plt.show()

# Titles added per month
titles_per_month = df['Month'].value_counts().sort_index()

plt.figure(figsize=(8, 5))
sns.barplot(x=titles_per_month.index, y=titles_per_month.values, palette='mako')
plt.title('Number of Titles Added by Month')
plt.xlabel('Month')
plt.ylabel('Number of Titles')
plt.xticks(range(13), ['Unknown', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
plt.show()

# Group data by Year and Type
type_year_counts = df.groupby(['Year', 'type']).size().unstack()

plt.figure(figsize=(12, 6))
type_year_counts.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Set3')
plt.title('Distribution of Movies and TV Shows Added Per Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend(title='Type', loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Combine all genres into a single string
all_genres = ' '.join(df['listed_in'].dropna())

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_genres)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Genres')
plt.show()

# Content rating distribution
rating_counts = df['rating'].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=rating_counts.index, y=rating_counts.values, palette='pastel')
plt.title('Distribution of Content Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# # Filter movies and extract duration in minutes
# movies = df[df['type'] == 'Movie']

# # Extract numerical durations for movies
# movies['duration'] = movies['duration'].str.replace(' min', '').astype(float)

# plt.figure(figsize=(10, 6))
# sns.histplot(movies['duration'], bins=30, kde=True, color='skyblue')
# plt.title('Distribution of Movie Durations')
# plt.xlabel('Duration (minutes)')
# plt.ylabel('Frequency')
# plt.show()

# Count of Movies vs TV Shows
type_counts = df['type'].value_counts().reset_index()
type_counts.columns = ['Type', 'Count']

fig = px.pie(type_counts, values='Count', names='Type', 
             title='Distribution of Movies and TV Shows', 
             color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()

# Group by Year and Month
heatmap_data = df.groupby(['Year', 'Month']).size().reset_index(name='Count')

fig = px.density_heatmap(heatmap_data, x='Month', y='Year', z='Count', 
                         title='Titles Added Per Month and Year',
                         color_continuous_scale='Blues')
fig.update_layout(yaxis=dict(type='category'), xaxis=dict(tickmode='linear'))
fig.show()

