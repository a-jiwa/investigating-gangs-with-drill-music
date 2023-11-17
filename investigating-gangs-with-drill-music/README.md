# README: Bellingcat Hackathon - London Gang Culture Investigation

## Project Overview
This repository hosts our project for the Bellingcat Hackathon, focusing on understanding gang culture in London through the lens of UK Drill music. Our team has developed a set of tools, datasets, and an article to aid researchers in exploring London gang culture and related gang violence.

### Key Components
1. **Website:** Provides a explanation of our investigation. Access it at: [[URL](https://a-jiwa.github.io/investigating-uk-drill/)]
2. **Command Line Tools:**
   - **Lyric Fetcher:** Retrieves the lyrics of all songs by a specified artist.
   - **Artist Similarity Matrix Tool:** Generates a similarity matrix for a set of artists using Machine Learning (ML) to uncover connections.
   - **Song Similarity Matrix Tool:** Creates a similarity matrix for a set of songs to discover connections, leveraging ML techniques.
   - **Aggressiveness Assessment Model:** An ML model that evaluates the aggressiveness of an artist based on their songs.
3. **Datasets:**
   - **London Gangs Dataset:** Contains information on most main London gangs, including members, descriptions, 'opps' (opponents), and allies.
   - **Slang Terms Dataset:** A comprehensive list of slang terms used in UK Drill music, with their real meanings.
   - **Gang Locations Map:** A geolocated map showing the locations of all major gangs in London.
4. **Research Article:** A written piece providing deep insights into our findings and methodologies.

### Usage
#### Installing Dependencies
- Ensure you have Python 3.x installed.
- Install necessary libraries using `pip install -r requirements.txt`.

#### Command Line Tools
- **Lyric Fetcher:** `python lyric_fetcher.py --artist <artist_name>`
- **Artist Similarity Matrix Tool:** `python artist_similarity.py --artists <artist1,artist2,...>`
- **Song Similarity Matrix Tool:** `python song_similarity.py --songs <song1,song2,...>`
- **Aggressiveness Assessment Model:** `python aggressiveness_model.py --artist <artist_name>`

#### Accessing Datasets
- All datasets are available in the `datasets` folder.
- Use any standard data analysis tool to explore these datasets.

#### Viewing the Map
- The gang locations map can be accessed via the provided URL on the website or directly opened from the `maps` folder.

### Contributing
We welcome contributions from researchers and enthusiasts. To contribute:
- Fork the repository.
- Make your changes.
- Submit a pull request with a clear description of your improvements or additions.

### Support
For any queries or support, please open an issue in this repository, and we will assist you as soon as possible.

### License
This project is licensed under the MIT License - see the `LICENSE` file for details.

### Acknowledgements
We thank Bellingcat and all participants in the Hackathon for their contributions and insights into this project.

---

This repository is a collaborative effort to shed light on a complex and critical issue. We encourage responsible and ethical use of the tools and data provided.
