# Personalized-Recommendation-System-for-E-commerce
This project develops a personalized recommendation system for an e-commerce platform to enhance the customer shopping experience. The primary goal is to provide tailored product suggestions by analyzing user preferences, leading to improved product discoverability, increased sales, and a more engaging shopping experience. 
<br><br>
Dataset

## Preprocessing
Preprocessing was done for the initial product dataset to extract the targetted products and create tags for each of them.

_<h4>Extracting Clothing data</h4>_
> Extracted only the clothing data from the initial dataset.
> 
> Replaced exising product IDs with numeric product IDs.

_<h4>Handling Missing Values</h4>_
> Removed unnessasary columns that contained missing data.
> 
> Imputed missing brand names from the the corresponding descriptions.

_<h4>Cleaning the descriptions and categories</h4>_
> Removed garbage text from the description.
> 
> Seperated words that were worngly concatenated (Eg: ending word of one sentence and the begining word of the next sentence).
> 
> Concatenated words seperated by "-" to account for those words being considered as seperate words when creating tags.

_<h4>Tagging</h4>_
> Removed stopwords and non-alphabetic words.
> 
> Selected only the words with specific part-of-speech tags (adjectives, nouns and proper nouns).
> 
> Lemmatized the selected words and added to the tags column.

## Frontend Development 
The front-end interface is developed using Flask, allowing users to interact with the recommendation engine. 
<br><br>
Key features include:
* Signup and Login Pages
    > Users can register and sign in enabling themselves to access personalized recommendations based on their preferences.
* Product Recommendation Page
    > This page displays personalized product suggestions, featuring essential details such as product images, names of the products, and their prices. 
* Product Details Page
    > Contains details such as product name, categories, description, price, ratings, and review counts. Shows content-based recommendations for the specific product and related products using the similar products recommendation function based on other users’ purchases. 
* User’s Item Basket page
    > This page shows user’s already bought item by add to cart method 


## Backend Development 
The backend handles user interactions and executes the recommendation logic. 
### User Authentication
* The system supports user signup and login processes, using the MySQL database, which stores user details such as username, email, phone, and password. 

### Content-Based Filtering
> The idea is that the products which have common content (Here we consider product tags) assigned to them, are similar.
* This component uses TF-IDF vectorization to analyze product tags created using product name, category and description during the preprocessing, and then consider cosine similarity between products.
* The top recommendations are returned with their corresponding relevance scores.
* The evaluation metrics are also calculated and returned as part of the output.

### User-Based Collaborative Filtering
> The idea is that users who have made similar purchases in the past are similar to each other.
* First a user-item matrix is constructed assess similarities between users based on the ratings they have given. 
* By calculating cosine similarity among users, we identify other similar users to our current target user, and recommend products which those others have already bought, but the target user still have not.
* This method uses the collective preferences of users to generate recommendations. 

### Item-Based Collaborative Filtering
> The idea is that, considering a product which is bought by many users, if there is another product many of those in this user group have bought, then these two products are similar.
* First a user-item matrix is constructed assess similarities between the items.
* By analyzing the co-occurrence of products rated by the same users, it computes similarity scores for items.
* The similarity scores then act as weights to calculate the average with user ratings, which ultimately decides the best recommendations.
* This allows the system to recommend products similar to those the user has already shown interest in.

### Users-who-have-bought-this-prouduct-have-also-bought
> This filtering mechanism is an extension of the item-based collaborative filtering.
* When a user searches for a specific product, the system identifies other products purchased by users who have bought that same item.
* It then checks the similarity between the searched product and these other users' already purchased items.
* The product of an item’s rating and its corresponding similarity score is utilized to sort the most significant recommendations, ensuring that users are presented with items that align with both their preferences and the buying behaviour of others.
* This method enhances personalized suggestions by leveraging collective purchase trends. 

### Hybrid Model
> To improve recommendation accuracy, the backend integrates a hybrid approach that combines both the content-based and collaborative filtering techniques.
* First, each of the filtering mechanisms are utilized to get a set of recommendations for a user-product combination.
* Then Duplicates are removed to refrain from recommending the same product multiple times.
* This multi-dimensional recommendation logic allows the system to provide a diverse set of suggestions.

### Trending Products
> Highlight the most popular items in the store.
* The score is calculated by multiplying each product's Rating by its Rating Count.
* The system then ranks products based on this score, ensuring that highly rated products with more user engagement appear at the top of the recommendations.
* This approach allows customers to discover the most popular items immediately, enhancing the overall shopping experience. 


## Model Evaluation
All of the models calculate each of these metrics and return them when they are called in the backend.

### Relevance Scores
> Relevance scores measure how well a recommended product matches the user's preferences based on historical interactions (e.g., purchases). 
* The ratings each user has given for a particular product act as its relevance score with respect to this project. Rating values are in the range of 0 to 5 with 5 being the 
maximum.

### Cosine Similarity
> Cosine similarity measures the cosine of the angle between two non-zero vectors in a multi-dimensional space.
* In the context of our recommendation system, these vectors represent the TF-IDF representations of product attributes such as descriptions and tags with respect to content-based recommendations, and user-item purchases in terms of the three collaborative filtering functions. 

### Average Cumulative Gain
> Measures how a list of recommendations fares compared to if all the recommended products were perfect recommendations.
* Perfect scenario: An items relevance score is 5 and its cosine similairy is 1.

### Discounted Cumulative Gain (DCG)
> DCG Measures the quality of recommendations based on there relevance scores and the relative position in the recommendation list.
* DCG discounts the items appear later in the list prioritizing the top items in the list of recommendations.

### Normalized Discounted Cumulative Gain (NDCG)
> NDCG measures the usefulness of a recommendation based on the position of relevant items in the ranked list.
* It accounts for both the relevance of recommended items and their ranking order, rewarding systems that present relevant items earlier in the list. 

## Database Management 
### Schema Design: 
The MySQL database is structured to smoothly manage user and product information through two primary tables: 
#### Users
> This table is used for user management, containing user details such as username, email, phone, and password.
> This structure ensures secure storage and retrieval of user credentials while assisting user authentication and account management. 
#### Product purchases
> This table contains purchases made by the user and relevant product metadata such as product id and ratings given by each user.
>  By organizing product attributes in this way, the system can perform efficient queries and retrieve necessary details for generating recommendations. 
