# MERT

## The Team
1. **Team name** : R-Squared
2. **Team members** : Sreehari J R, Joel Varghese Oommen, Glenn Mathews
3. **Track** : Media

## Problem statement
Content creators and film-makers struggle to efficiently generate creative narratives from current events due to the lack of tools that connect user intent with relevant news insights and structured storytelling.

## Solution

## Project description

# Technical details
## Tech stack and libraries used
```bash
streamlit # web-interface
networkx # to build the connected graph object
plotly # to build the 3D interactive mind map
beautifulsoup4 # web-scraping from searched, relevant articles
requests # to send requests to api endpoints
pandas # for user-friendly visualiaztion of articles and related data
numpy # for numerical computations such ass confidence_score calculation for each article data
sentence_transformer # for computing semantice similarity between online articles
```

## Implementation
**PromptEngine.py**
**llm.py**
**main.py**
**app.py**

## Instructions to use
1. Install the dependecies
```bash
pip install -r requirements.txt
```
2. Run the streamlit app
```bash
streamlit run app.py
```
3. Within the web-interface - type in a prompt related to any news topic.
4. The generated mind map and the script/screenplay can be downloaded.

## Screenshots
**Front page**
<p align="center">
  <img src="screenshots\Screenshot 2025-07-20 134030.png" alt="Front page">
</p>

**Processing user input**
<p align="center">
  <img src="screenshots\Screenshot 2025-07-20 134226.png" alt="next1">
</p>

**Mind map of news articles**
<p align="center">
  <img src="screenshots\Screenshot 2025-07-20 134308.png" alt="next2">
</p>

**Details of extracted news articles**
<p align="center">
  <img src="screenshots\Screenshot 2025-07-20 134326.png" alt="next3">
</p>

**Summary of the extracted events**
<p align="center">
  <img src="screenshots\Screenshot 2025-07-20 134344.png" alt="next4">
</p>