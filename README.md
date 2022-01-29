# YouTube Views Predictor and Analysis

## Goal
Build a model that can predict YouTube views from the video's title and analyze the effect of key words on views.

## Background and Data Sources

[YouTube] is one of the largest platforms for viewing videos online.  Content creators can share use the platform to share important information or educate others.  They can also monetize their videos and obtain ad revenue to generate a steady income.

This project will analyze publicly available data from videos uploaded at **[LowkoTV]**.  Lowko's channel mainly features **[StarCraft 2]** (SC2), a popular military science fiction strategy game, and is known for casting matches between professional SC2 players.  Using YouTube's API, we downloaded the title, thumbnail image, duration, and number of views for roughly 1,200 videos dated between 2018 through 2021.

We also downloaded information from **[Aligulac]**, a data base containing professional SC2 player statistics.  Using Aligulac's API, the player information obtained included their country, preferred SC2 race, and recent activity.


## Data Processing

Created a pre-processor function to filter out data specific to LowkoTV's content, naming conventions, professional player names, and video games.

In the video title, the video game names are usually found before the ":" at the beginning of the title.  The processor looks at the video game titles first before cleaning the rest of the title.

YouTube titles contain engaging words, usually emphasized in all capital letters.  Fully capitalized words are saved as a separate word from their non capitalized counterparts. 

For numerical text data, numbers related to matchup, video games, tech/PC parts, and 2018-2022 years are tokenized as well.

Some examples of video title's include:
- Starcraft 2: The MOST PROMISING Pro Game of 2022! (HeroMarine vs MaxPax)
- Age of Empires 4: StarCraft Grandmasters Play 2v2! (Lowko & Winter)
- Frostpunk: Endless Mode - FINAL EPISODE!


## Exploratory Data Analysis

Most of the visualizations created below used data corresponding to key words pulled from the video title.  We aggregated the average views for each of these key words.

### Non SC2 Specific Visualizations

#### 1. Views By Video Length/Duration

The video length is measured in minutes.  Video's longer than 90 minutes usually generate less than average views.  The highest views are found in videos less than 90 minutes.

![header] (https://github.com/NelGen/NG-YouTube-Views-Analysis/blob/main/App%20Images/Views%20By%20Video%20Length.png)

#### 2. Views By Video Game

StarCraft 1 leads the average number of views by a substantial amount followed by StarCraft 2.  All other featured games average less than 100,000 views.

![header] (https://github.com/NelGen/NG-YouTube-Views-Analysis/blob/main/App%20Images/Average%20Views%20Per%20Game.png)

#### 3. Views By Certain Key Words (Capitalized vs Normal Case)

We selected some words we felt would better engage someone to view a video and compared their fully capitalized versions to the word written normally.  Words like "pro" and "cheese" generate much higher views than if they were fully capitalized.  Words like "MOST", "BUILD", and "RUSH" generate higher views when fully capitalized.

![header] (https://github.com/NelGen/NG-YouTube-Views-Analysis/blob/main/App%20Images/Average%20Views%20By%20Certain%20Key%20Words.png)

### SC2 Specific Visualizations

#### 4.  Views By Pro SC2 Player

This list of players were active during the StayAtHome Story Cup # 4, taking place in late October 2021.  However, there are many high profile Korean players fulfilling their mandatory military service missing from the list which I've added.  In addition, there are other popular non-professional players featured on LowkoTV such as Florencio.

On top of the list rivalling Serral, known as one of the greatest of all time, is Florencio who is known for strategies outside the norm.

![header] (https://github.com/NelGen/NG-YouTube-Views-Analysis/blob/main/App%20Images/Average%20Views%20By%20SC2%20Player.png)

#### 5.  Views By Pro Player's SC2 Race

Considering that the recent SC2 world champions and Lowko play Zerg, it's no surprise that videos featuring Zerg generate the highest average views.

* Z = Zerg, T = Terran, P = Protoss

![header] (https://github.com/NelGen/NG-YouTube-Views-Analysis/blob/main/App%20Images/Average%20Views%20By%20Player's%20SC2%20Race.png)

#### 6.  Views By SC2 Matchups

Similarly, matchups featuring Zerg also generage the highest views.  Protoss matches are typically defensive with few engagements throughout the game.

![header] (https://github.com/NelGen/NG-YouTube-Views-Analysis/blob/main/App%20Images/Average%20Views%20By%20SC2%20Matchups.png)

## Modeling

We trained several neural networks with 3 dense layers, relu activation functions, 2 drop out layers, and compiled with optimizer adam.  The final output layer uses a linear activation function as our desired result is the number of views.  For the loss, we compared RMSE against MAE.  


### 1. Base Model With Loss RMSE

Results:
* Avg Views: 107,394
* Train RMSE: 92,474 (86% of Avg Views)
* Test RMSE: 80,133 (75% of Avg Views)

![header] (https://github.com/NelGen/NG-YouTube-Views-Analysis/blob/main/App%20Images/Base_Model.png)

### 2. Best Model With Loss MAE and Outliers Removed

Model performance was not improving with parameter tuning.  Returning to the data, we re-trained the model after removing outliers increasing performance.

Results:
* Avg Views: 86,653
* Train RMSE: 24,838 (29% of Avg Views)
* Test RMSE: 30,935 (36% of Avg Views)

![header] (https://github.com/NelGen/NG-YouTube-Views-Analysis/blob/main/App%20Images/Best_Model.png)

## Views Predictor App

The visualizations and trained model were uploaded using **[Heroku]**.  LowkoTV can enter test video titles and obtain estimates on how they will perform.

The app returns the effectiveness of each word based on the trained data if the word was seen before.  If not, the user is advised to input the missing words in all caps.

## Next Steps

1. Try other neural network architectures
2. Continue to analyze image data to discover any impactful features
3. Train the model on newer videos over time to maintain accuracy
4. Explore different sized N-grams of the title

[YouTube]: https://www.youtube.com/
[LowkoTV]: https://www.youtube.com/c/LowkoTV
[StarCraft 2]: https://starcraft2.com/en-us/
[Aligulac]: http://aligulac.com/
[Heroku]: https://yt-views-predictor-ltv.herokuapp.com/