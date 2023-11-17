import csv

import lyricsgenius

# Your credentials
client_access_token = "TE8QoM5zhGXueHUL9m-Kb8dJ8TjbQv7X8B0VkB7lt5FOPxAHuHJJbyCrr76uEZD-"

# Create a Genius client
genius = lyricsgenius.Genius(client_access_token)

# List of UK drill artists
uk_drill_artists = ['Bigare_Ben', 'Farmer12', 'mgespaz', 'RkInGmusic', 'CofiCarrera']  # Replace with actual list of artist names

# Function to fetch all lyrics for an artist and save to a CSV file
def fetch_all_lyrics_for_artist(artist_name):
    try:
        artist = genius.search_artist(artist_name, max_songs=None, sort='title')
        with open(f"./lyrics/{artist_name}_lyrics.csv", "w", newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Title", "Lyrics"])  # Writing header
            for song in artist.songs:
                writer.writerow([song.title, song.lyrics.replace('\n', ' ')])  # Replace newlines with spaces in lyrics
        print(f"Lyrics for {artist_name} saved to {artist_name}_lyrics.csv")
    except Exception as e:
        print(f"Error occurred while fetching songs for {artist_name}: {e}")

# Iterate over the list of artists and fetch their lyrics
for artist_name in uk_drill_artists:
    fetch_all_lyrics_for_artist(artist_name)
