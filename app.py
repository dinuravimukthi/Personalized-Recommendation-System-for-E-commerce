from flask import Flask, request, render_template, redirect, session, flash, url_for
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# load files
trending_products = pd.read_csv("models/trending_product.csv")
train_data = pd.read_csv("models/product_and_user_data.csv")
user_data = pd.read_csv('models/product_and_user_data.csv')
product_data = pd.read_csv('models/product_data.csv')

# Database configuration
app.secret_key = "12345678"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root:@localhost/ecommerce"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define your model class for the 'users' table
class Users(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Recomennded system function
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text


# Content Based Recommendation 
def content_based_recommendations(train_data, item_name, top_n=10):

    if item_name not in train_data['Product Name'].values:
        return pd.DataFrame()
    
    # Initialize the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Fit and transform the 'Tags' column into a TF-IDF matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(train_data['Tags'])

    # Compute cosine similarity between all products based on their TF-IDF tag vectors
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Select a specific product by name 
    product_index = train_data[train_data['Product Name'] == item_name].index[0]

    # Get similarity scores for the selected product with all other products
    sim_scores = list(enumerate(cosine_sim_matrix[product_index]))

    # Sort the products by similarity score (in descending order)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top N similar products (excluding the product itself)
    recommended_indices = [i[0] for i in sim_scores[1:top_n+1]]

    # Display the details of the recommended products
    recommended_products = train_data.iloc[recommended_indices]
    recommended_products = recommended_products.sort_values('Rating', ascending=False)

    return recommended_products

# Collaborative Filtering Recommendation
def collaborative_filtering_recommendations(train_df, target_user_id, top_n=10):

    # Step 1: Create the user-item matrix
    user_item_matrix = train_df.pivot_table(index='User Id', columns='Product Name', values='Rating', aggfunc='mean').fillna(0)

    # Step 2: Calculate the user similarity matrix
    user_similarity = cosine_similarity(user_item_matrix)

    # Step 3: Find the target user's index
    target_user_index = user_item_matrix.index.get_loc(target_user_id)

    # Step 4: Get similarity scores for the target user
    user_similarities = user_similarity[target_user_index]
    similar_users_indices = user_similarities.argsort()[::-1][1:]

    # Step 5: Find recommended items
    recommended_items = []
    for user_index in similar_users_indices:
        rated_by_similar_user = user_item_matrix.iloc[user_index]
        not_rated_by_target_user = (rated_by_similar_user > 0) & (user_item_matrix.iloc[target_user_index] == 0)
        recommended_items.extend(user_item_matrix.columns[not_rated_by_target_user][:top_n])

        if len(recommended_items) >= top_n:
            break

    # Step 6: Get recommended items' details
    recommended_items_details = train_df[train_df['Product Name'].isin(recommended_items)][['Prod Id', 'Product Name', 'Brand', 'Product Url', 'Rating']]

    return recommended_items_details.head(top_n)

# Item Based Collaborative Filtering Recommendation
def item_based_recommendation(user_id, user_data, product_data, n_top=5):
    """
    Get the item-based product recommendations for the given user-id

    Args
        user_id: user id of the user
        user_data: dataframe with user-product ratings
        product_data: dataframe with product data
        no_top: number of top recommendations to return

    Return
        Dataframe with top recommendation product data
    """
    import operator
    
    # Get mean rating and rating count by each product
    agg_ratings= user_data.groupby('Prod Id').agg(mean_rating = ('Rating', 'mean'), number_of_ratings = ('Rating', 'count')).reset_index()
    # Join each products mean_rating and number_of_ratings with the user data
    agg_ratings_df = pd.merge(user_data, agg_ratings, on='Prod Id', how='inner')

    # User-product matrix
    matrix = agg_ratings_df.pivot_table(index='Prod Id', columns='User Id', values='Rating')
    # Normalize the matrix
    matrix = matrix.subtract(matrix.mean(axis=1), axis=0)
    
    # Cosine similarity matrix
    item_similarity_cosine = cosine_similarity(matrix.fillna(0))
    # Create a dataframe for the cosine similarity matrix
    cosine_df = pd.DataFrame(item_similarity_cosine)
    cosine_df.columns = matrix.index.values
    cosine_df.index = matrix.index.values

    # Products the user has not rated
    user_not_rated = pd.DataFrame(matrix[user_id].isna()).reset_index()
    user_not_rated = user_not_rated[user_not_rated[user_id] == True]['Prod Id'].values.tolist()

    # Products the user has rated
    user_rated = pd.DataFrame(matrix[user_id].dropna(axis=0))\
    .sort_values(user_id, ascending=False)\
    .reset_index()\
    .rename(columns={user_id:'Rating'})

    # Hold not-rated products and their predicted ratings
    pred_ratings = {}
    for prod_nr in user_not_rated:
        prod_nr_similarity = cosine_df[prod_nr].reset_index().rename(columns={'index':'Prod Id', 
                                                                              prod_nr:'Similarity Score'})
        # Rank the similarities between rated products and the not-rated product
        prod_nr_and_r_similarity = pd.merge(user_rated, prod_nr_similarity, how='inner', on='Prod Id')\
        .sort_values('Similarity Score', ascending=False)
        # Predicted rating
        weights = prod_nr_and_r_similarity['Similarity Score']
        rating_arr = prod_nr_and_r_similarity['Rating']
        if sum(weights) > 0:
            pred_rating = round(np.average(rating_arr, weights=weights), 5)
        else:
            pred_rating = 0
        # Save the not-rated product and corresponding predicted rating
        if pred_rating > 0:
            pred_ratings[prod_nr] = pred_rating

    recs = sorted(pred_ratings.items(), key=operator.itemgetter(1), reverse=True)[:n_top] # Top recommendations
    recs_prod_ids = []
    for idx, val in enumerate(recs):
        recs_prod_ids.append(recs[idx][0])
        
    # Create the output dataframe
    df = product_data[product_data['Prod Id'].isin(recs_prod_ids)].reset_index(drop=True)
    
    return df

# Item Based Colleborative Filtering Recommendations For All The Products The
# Customers Who Purchased a Specific Product Bought.
def specific_item_based_recommendation(product_id, user_data, product_data, n_top=5):
    """
    Get the item-based product recommendations among the other products,
    which the useres who purchased the selected product bought.

    Args
        product_id: product id of the selected product
        user_data: dataframe with user-product ratings
        product_data: dataframe with product data
        n_top: number of top recommendations to return

    Return
        Dataframe with top recommendation product data
    """
    import operator
    
    # Get the users who bought the product
    user_ids = user_data[user_data['Prod Id'] == product_id]['User Id']
    users = user_data[user_data['User Id'].isin(user_ids.values)]
    
    # Get mean rating and rating count by each product
    agg_ratings= users.groupby('Prod Id').agg(mean_rating = ('Rating', 'mean'), number_of_ratings = ('Rating', 'count')).reset_index()
    # Join each products mean_rating and number_of_ratings with the user data
    agg_ratings_df = pd.merge(users, agg_ratings, on='Prod Id', how='inner')
    
    # User-product matrix
    matrix = agg_ratings_df.pivot_table(index='Prod Id', columns='User Id', values='Rating')
    print(user_ids)
    try:
        # Cosine similarity matrix
        item_similarity_cosine = cosine_similarity(matrix.fillna(0))
        # Create a dataframe for the cosine similarity matrix
        cosine_df = pd.DataFrame(item_similarity_cosine)
        cosine_df.columns = matrix.index.values
        cosine_df.index = matrix.index.values
    
        # Get the similairty of selected product with other products the users who
        # purchased this product bought
        similarity_with_selected_product = cosine_df[product_id]
        
        # Get the similairty of other products
        similarity_with_selected_product = pd.DataFrame(similarity_with_selected_product.drop(product_id)).reset_index()
        similarity_with_selected_product = similarity_with_selected_product.rename(columns={'index':'Prod Id', product_id:'Similarity'})
    
        weight_matrix = pd.merge(similarity_with_selected_product, agg_ratings[['Prod Id', 'mean_rating']], on='Prod Id', how='inner')
        weight_matrix['relevance_score'] = weight_matrix['Similarity'] * weight_matrix['mean_rating']
        weight_matrix = weight_matrix.sort_values('relevance_score', ascending=False).reset_index(drop=True)
    
        top_recommendations = weight_matrix[:n_top]
    
        # Get the product data for the top recommendations
        df = product_data[product_data['Prod Id'].isin(top_recommendations['Prod Id'].tolist())].reset_index(drop=True)
        df = pd.merge(df, top_recommendations[['Prod Id', 'relevance_score']], on='Prod Id', how='inner')
        df = df.sort_values('relevance_score', ascending=False)
        df = df.drop('relevance_score', axis=1)

        return top_recommendations, df

    except:
        return None, None


# Hybrid Recommendation
def hybrid_recommendations(train_data, target_user_id, item_name, top_n=10):

    # Getcontent Based recommendations
    content_based_rec = content_based_recommendations(product_data, item_name, top_n)
    # Get Collaborative filtering recommendations
    collaborative_filtering_rec = collaborative_filtering_recommendations(train_data, target_user_id, top_n)
    #print(f'user based {collaborative_filtering_rec['Prod Id']}')
    # Get Item Basesd filtering recommendations
    item_based_rec = item_based_recommendation(target_user_id, user_data, product_data, top_n)
    #print(f'item {item_based_rec['Prod Id']}')
    # Merge and deduplicate the recommendations
    hybrid_rec = pd.concat([content_based_rec, collaborative_filtering_rec, item_based_rec]).drop_duplicates()
    #print(hybrid_rec['Prod Id'])
    return hybrid_rec.head(10)

#getting random image urls
random_image_urls = [
    "static/images/img_1.jpg",
    "static/images/img_2.jpg",
    "static/images/img_3.jpg",
    "static/images/img_4.jpg",
    "static/images/img_5.jpg",
]

#routes
@app.route("/")
def index():
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100]
    
    # Pass zip function to Jinja2
    return render_template(
        'index.html',
        trending_products=trending_products.head(5),
        truncate=truncate,
        random_product_image_urls=random_product_image_urls,
        random_price=[random.choice(price) for _ in range(len(trending_products.head(5)))],
        zip=zip  # Passing zip function
    )

#routes
@app.route("/main")
def main():
    return render_template('main.html', hybrid_rec=None)

# Sign up model
@app.route("/signup", methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        phone = request.form['phone']
        password = request.form['password']

        user_ids = user_data['User Id'].tolist()
        random_user_id = random.choice(user_ids)

        # Create a new user instance
        new_user = Users(username=username, email=email, phone=phone, password=password, user_id=random_user_id)
        db.session.add(new_user)
        db.session.commit()

        random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
        price = [40, 50, 60, 70, 100]
        
        return render_template(
        'index.html',
        signup_success="True",
        trending_products=trending_products.head(5),
        truncate=truncate,
        random_product_image_urls=random_product_image_urls,
        random_price=[random.choice(price) for _ in range(len(trending_products.head(5)))],
        zip=zip # Passing zip function
    )

    return render_template('index.html',signup_success="False")

# login model
@app.route("/login", methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        # Use get method to avoid KeyError in case of missing fields
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            flash('Please enter both username and password', 'danger')
            return redirect(url_for('index'))

        # Query the database to check if the user exists
        user = Users.query.filter_by(username=username, password=password).first()

        if user:
            session['username'] = user.username
            session['user_id'] = user.user_id
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('index.html')

# logout model
@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)  
    session.pop('user_id', None)    
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('index'))  

target_user_id = 100162

@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if request.method == 'POST':
        products = request.form.get('products')

        hybrid_rec = hybrid_recommendations(train_data, target_user_id, products)

        if hybrid_rec.empty:
            message = "No recommendations available for this product."
            return render_template('main.html', message=message)

        else:
            # Create a list of random image URLs for each recommended product
            random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(hybrid_rec))]
            price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]

            return render_template('main.html', 
                                   hybrid_rec=hybrid_rec,
                                   truncate=truncate, 
                                   random_product_image_urls=random_product_image_urls,
                                   random_price=random.choice(price))


# Function to fetch product details
def fetch_product_by_id(product_id):
    if product_id in list(product_data['Prod Id']):
        return product_data[product_data['Prod Id'] == product_id].iloc[0]

# Route to render the product details page
@app.route('/product/<int:product_id>')
def product_detail(product_id):
    # Fetch product details from your data source using the product_id
    product = fetch_product_by_id(product_id)
    # Content based recommendations
    content_based_rec = content_based_recommendations(product_data, product['Product Name'], top_n=20)
    # Item based collaborative recommendations
    top_rec, specific_item_based_rec = specific_item_based_recommendation(product_id, user_data, product_data, n_top=20)
    return render_template('product.html', 
                           product=product, 
                           content_based_recommendations=content_based_rec, 
                           item_based_recommendations=specific_item_based_rec)


# routes
@app.route("/home")
def home():
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100]
    
    # Pass zip function to Jinja2
    return render_template(
        'index.html',
        trending_products=trending_products.head(5),
        truncate=truncate,
        random_product_image_urls=random_product_image_urls,
        random_price=[random.choice(price) for _ in range(len(trending_products.head(5)))],
        zip=zip  # Passing zip function
    )


if __name__=='__main__':
    app.run(debug=True)