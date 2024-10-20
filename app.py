from flask import Flask, request, render_template, redirect, session, flash, url_for, jsonify
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
user_data_df = pd.read_csv('models/product_and_user_data.csv')
product_data_df = pd.read_csv('models/product_data.csv')

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

# Define your model class for the 'users_product_data' table
class User_product_data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), nullable=False)
    product_id = db.Column(db.String(100), nullable=False)
    product_name = db.Column(db.String(255), nullable=False)
    price = db.Column(db.String(100), nullable=False)
    rating = db.Column(db.String(100), nullable=False)

# Define the model for 'product_data' table
class Product_data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prod_id = db.Column(db.String(10), nullable=True)
    product_url = db.Column(db.String(200), nullable=True)
    product_name = db.Column(db.String(200), nullable=True)
    description = db.Column(db.String(5000), nullable=True)
    filtered_description = db.Column(db.String(5000), nullable=True)
    list_price = db.Column(db.String(10), nullable=True)
    sale_price = db.Column(db.String(10), nullable=True)
    brand = db.Column(db.String(100), nullable=True)
    category = db.Column(db.String(300), nullable=True)
    filtered_category = db.Column(db.String(300), nullable=True)
    rating = db.Column(db.String(10), nullable=True)
    rating_count = db.Column(db.String(100), nullable=True)
    available = db.Column(db.String(10), nullable=True)
    tags = db.Column(db.String(2000), nullable=True)

# Recomennded system function
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text

# Trending Funtion
def get_trending_products(train_data, top_n=10):
    """
    Calculate the trending products based on Rating and Rating Count.

    Parameters:
    train_data (pd.DataFrame): DataFrame containing product data with Rating and Rating Count.
    top_n (int): Number of top trending products to return.

    Returns:
    pd.DataFrame: DataFrame containing the top trending products.
    """
    # Ensure the Rating and Rating Count columns are numeric
    train_data['Rating'] = pd.to_numeric(train_data['Rating'], errors='coerce')
    train_data['Rating Count'] = pd.to_numeric(train_data['Rating Count'], errors='coerce')

    # Calculate a trending score
    train_data['Trending Score'] = train_data['Rating'] * train_data['Rating Count']

    # Sort the products based on the Trending Score in descending order
    trending_products = train_data.sort_values(by='Trending Score', ascending=False)

    # Get the top N trending products
    trending_products_top_n = trending_products.head(top_n)

    #return trending_products_top_n[['Product Name', 'Rating', 'Rating Count', 'Trending Score']]
    return trending_products_top_n

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
    # recommended_products = recommended_products.sort_values('Rating', ascending=False)

    
    rel_scores_df = pd.DataFrame(sim_scores[1:top_n+1]).rename({0:'Index', 1:'Similarity Score'}, axis=1)
    rel_scores_df.index = rel_scores_df.Index
    rel_scores_df = rel_scores_df.join(recommended_products)
    rel_scores_df['Relevance Score'] = rel_scores_df['Similarity Score'] * rel_scores_df['Rating']
    rel_scores_df = rel_scores_df.sort_values('Relevance Score', ascending=False)

    rel_scores = [round(s, 4) for s in rel_scores_df['Relevance Score'].to_list()]

    recommended_products = rel_scores_df.drop(['Index', 'Similarity Score'], axis=1).reset_index(drop=True)
    
    # Calculate metrics
    metrics = {
        'avg_cg': round(calculate_avg_cg(rel_scores), 4),
        'dcg': round(calculate_dcg(rel_scores), 4),
        'ndcg': round(calculate_ndcg(rel_scores), 4),
    }
    
    return recommended_products, rel_scores, metrics

# Collaborative Filtering Recommendation
def collaborative_filtering_recommendations(train_df, product_df, target_user_id, top_n=10):
    
    # Step 1: Create the user-item matrix
    user_item_matrix = train_df.pivot_table(index='User Id', columns='Product Name', values='Rating', aggfunc='mean').fillna(0)

    # Step 2: Calculate the user similarity matrix
    user_similarity = cosine_similarity(user_item_matrix)

    # Step 3: Find the target user's index
    target_user_index = user_item_matrix.index.get_loc(target_user_id)

    # Step 4: Get similarity scores for the target user
    user_similarities = user_similarity[target_user_index]
    similar_users_indices = user_similarities.argsort()[::-1][1:]


    # Step 5: Find recommended items and calculate relevance scores
    recommended_items = {}

    for user_index in similar_users_indices:
        rated_by_similar_user = user_item_matrix.iloc[user_index]
        not_rated_by_target_user = (rated_by_similar_user > 0) & (user_item_matrix.iloc[target_user_index] == 0)
        
        # Get products that are not rated by target user
        for product in user_item_matrix.columns[not_rated_by_target_user]:
            if product not in recommended_items:
                # Initialize relevance score as 0
                recommended_items[product] = 0
            # Add to relevance score weighted by user similarity
            recommended_items[product] += user_similarities[user_index] * rated_by_similar_user[product]

        if len(recommended_items) >= top_n:
            break

    # Step 6: Sort items by relevance score
    recommended_items = sorted(recommended_items.items(), key=lambda x: x[1], reverse=True)

    # Step 7: Get recommended items' details
    recommended_product_names = [item[0] for item in recommended_items][:top_n]
    recommended_products_details = product_df[product_df['Product Name'].isin(recommended_product_names)]
    
    # Add relevance scores to the DataFrame
    recommended_products_details['relevance_score'] = recommended_products_details['Product Name'].apply(
        lambda x: dict(recommended_items).get(x, 0)
    )
    
    # Sort by relevance score
    recommended_products_details = recommended_products_details.sort_values('relevance_score', ascending=False)

    # Step 8: Compute metrics (Average CG, DCG, NDCG)
    relevance_scores = list(recommended_products_details['relevance_score'])

    avg_cg = calculate_avg_cg(relevance_scores)
    ndcg = calculate_ndcg(relevance_scores)

    # Return recommendations and metrics
    metrics = {
        'average_cg': avg_cg,
        'ndcg': ndcg
    }
    
    return recommended_products_details.head(top_n), metrics

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
        Dataframe with top recommendation product data,
        relevance scores, and metrics if there are recommendations
        None if not
    """
    import operator
    
    # Get mean rating and rating count by each product
    agg_ratings= user_data.groupby('Prod Id').agg(mean_rating = ('Rating', 'mean'), number_of_ratings = ('Rating', 'count')).reset_index()
    # Join each products mean_rating and number_of_ratings with the user data
    agg_ratings_df = pd.merge(user_data, agg_ratings, on='Prod Id', how='inner')

    # User-product matrix
    matrix = agg_ratings_df.pivot_table(index='Prod Id', columns='User Id', values='Rating')
    # Normalize the matrix
    # matrix = matrix.subtract(matrix.mean(axis=1), axis=0)
    
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
            # Save the not-rated product and corresponding predicted rating
            pred_ratings[prod_nr] = pred_rating

    if len(pred_ratings) > 0:
        recs = sorted(pred_ratings.items(), key=operator.itemgetter(1), reverse=True)[:n_top] # Top recommendations
        recs_prod_ids = []
        rel_scores = []
        for idx, val in enumerate(recs):
            recs_prod_ids.append(recs[idx][0])
            rel_scores.append(float(recs[idx][1]))
            
        # Create the output dataframe
        df = product_data[product_data['Prod Id'].isin(recs_prod_ids)].reset_index(drop=True)
        
        # Calculate metrics
        metrics = {
            'avg_cg': round(calculate_avg_cg(rel_scores), 4),
            'dcg': round(calculate_dcg(rel_scores), 4),
            'ndcg': round(calculate_ndcg(rel_scores), 4),
        }
        return df, rel_scores, metrics
    else:
        return None, None, None

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
        rel_scores = [round(s, 4) for s in  df['relevance_score'].to_list()]
        
        df = df.drop('relevance_score', axis=1)

        # Calculate metrics
        metrics = {
            'avg_cg': round(calculate_avg_cg(rel_scores), 4),
            'dcg': round(calculate_dcg(rel_scores), 4),
            'ndcg': round(calculate_ndcg(rel_scores), 4),
        }

        return df, rel_scores, metrics

    except:
        return None, None, None

# Hybrid Recommendation
def hybrid_recommendations(user_data, product_data, target_user_id, item_name, top_n=10):

    # Ensure that target_user_id is being processed as expected
    if isinstance(target_user_id, str):
        target_user_id = int(target_user_id)  # Convert to int if necessary

    # Getcontent Based recommendations

    content_based_rec, cb_scores, cb_metrics = content_based_recommendations(product_data, item_name, top_n)
    print(f'METRICS: Content Based Recommendations for Product {product_data[product_data['Product Name'] == item_name]['Prod Id'].to_list()[0]}: {cb_metrics}')
    
    # Get collaborative filtering recommendations
    collaborative_filtering_rec, collab_metrics = collaborative_filtering_recommendations(user_data, product_data, target_user_id, top_n)


    # Get item-based filtering recommendations
    item_based_rec, ib_scores, ib_metrics = item_based_recommendation(target_user_id, user_data, product_data, top_n)

    # Merge and deduplicate the recommendations
    if item_based_rec is not None:
        hybrid_rec = pd.concat([content_based_rec, collaborative_filtering_rec, item_based_rec]).drop_duplicates(subset=['Prod Id'])
    else:
        hybrid_rec = pd.concat([content_based_rec, collaborative_filtering_rec]).drop_duplicates(subset=['Prod Id'])
    
    # Print metrics from collaborative filtering and item-based recommendations
    print(f'METRICS: Collaborative Filtering Recommendations for User {target_user_id}: {collab_metrics}')
    if item_based_rec is not None:
        print(f'METRICS: Item-Based Recommendations for User {target_user_id}: {ib_metrics}')

    # Sort by rating and return top N results
    hybrid_rec = hybrid_rec.sort_values('Rating', ascending=False)
    
    return hybrid_rec.head(10)

# Function to calculate average CG
def calculate_avg_cg(relevance_scores):
    return float(np.sum(relevance_scores) / (5 * len(relevance_scores)))

# Function to calculate DCG
def calculate_dcg(relevance_scores):
    return float(np.sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores)]))

# Function to calculate NDCG
def calculate_ndcg(relevance_scores):
    # Calculate DCG
    dcg_value = calculate_dcg(relevance_scores)
    # Calculate IDCG
    relevance_scores.sort(reverse=True)
    idcg_value = calculate_dcg(relevance_scores)
    return float(dcg_value / idcg_value) if idcg_value != 0 else 0

#getting random image urls
random_image_urls = [
    "static/images/img_1.jpg",
    "static/images/img_2.jpg",
    "static/images/img_3.jpg",
    "static/images/img_4.jpg",
    "static/images/img_5.jpg",
]

@app.route("/")
def index():
    # Get the top 5 trending products
    product_data = get_product_data()

    trending_products = get_trending_products(product_data, top_n=5)

    # Create a DataFrame with the trending products and their prices
    trending_prices = product_data.loc[product_data['Product Name'].isin(trending_products['Product Name']), ['Product Name','List Price']]
    
    # Convert to a dictionary for easier access in the template
    price_dict = trending_prices.set_index('Product Name')['List Price'].to_dict()

    # Generate random product image URLs for each trending product
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]

    # Pass zip function to Jinja2
    return render_template(
        'index.html',
        trending_products=trending_products,
        truncate=truncate,
        random_product_image_urls=random_product_image_urls,
        price_dict=price_dict,  # Pass the dictionary of prices
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
        user_data = get_user_and_product_data()

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
        prices = train_data['List Price'].head(5).tolist()
        
        return render_template(
        'index.html',
        signup_success="True",
        trending_products=trending_products.head(5),
        truncate=truncate,
        random_product_image_urls=random_product_image_urls,
        random_price=prices,
        zip=zip # Passing zip function
    )

    return render_template('index.html',signup_success="False")

# login model
# Login route
@app.route("/login", methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    if not username or not password:
        return jsonify({'success': False, 'message': 'Please enter both username and password'})

    # Query the database to check if the user exists
    user = Users.query.filter_by(username=username, password=password).first()

    if user:
        session['username'] = user.username
        session['user_id'] = user.user_id
        return jsonify({'success': True, 'message': 'Login successful!'})
    else:
        return jsonify({'success': False, 'message': 'No user found, please enter a valid username'})

# logout model
@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)  
    session.pop('user_id', None)    
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('index'))  

# Recommendations
@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if request.method == 'POST':
        products = request.form.get('products')
        target_user_id = request.form.get('user_id')
        product_data = get_product_data()

        # Check if user is logged in (i.e., target_user_id exists)
        if target_user_id:
            user_data = get_user_and_product_data()
            # Use hybrid recommendation if user is logged in
            hybrid_rec = hybrid_recommendations(user_data, product_data, target_user_id, products)
        else:
            # Use content-based recommendation only if user is not logged in
            hybrid_rec = content_based_recommendations(product_data, products)

        if hybrid_rec.empty:
            message = "No recommendations available for this product."
            return render_template('main.html', message=message)
        else:
            # Create a list of random image URLs for each recommended product
            random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(hybrid_rec))]
            prices = train_data['List Price'].head(5).tolist()

            return render_template('main.html', 
                                   hybrid_rec=hybrid_rec,
                                   truncate=truncate, 
                                   random_product_image_urls=random_product_image_urls,
                                   random_price=prices)

       
# add to cart
@app.route("/add_to_cart", methods=['POST'])
def add_to_cart():
    if 'user_id' in session:  # Check if the user is logged in
        user_id = session['user_id']
        product_id = request.form['product_id']
        product_name = request.form['product_name']
        rating = request.form['rating']
        price = request.form['price']

        # Create a new Cart entry
        cart_item = User_product_data(user_id=user_id, product_id=product_id, product_name=product_name, rating=rating, price=price)
        db.session.add(cart_item)
        db.session.commit()

        # Return a JSON response indicating success
        return jsonify({'success': True, 'message': 'Product added to cart!'})

    else:
        # Return a JSON response indicating failure
        return jsonify({'success': False, 'message': 'Please log in to add items to your cart.'})

# Update product rating and rating count
@app.route("/update_product_rating", methods=['POST'])
def update_product_rating():
    product_id = int(request.form['product_id'])

    new_user_product_data = get_user_and_product_data()

    agg_ratings = new_user_product_data.groupby('Prod Id').agg(mean_rating = ('Rating', 'mean'), rating_count=('Rating', 'count')).reset_index()
    agg_rating_for_product = agg_ratings[agg_ratings['Prod Id'] == product_id]
    new_rating = list(agg_rating_for_product['mean_rating'])[0]
    new_rating_count = list(agg_rating_for_product['rating_count'])[0]

    # Updata the product_data entry
    product = Product_data.query.filter_by(prod_id=str(product_id)).first()
    
    if product:
        product.rating = str(round(new_rating, 2))
        product.rating_count = str(new_rating_count)
        db.session.commit()
        print(f'Updated rating and rating count for product {product_id}')
    else:
        print(f"Product {product_id} not found")
    # Return a JSON response indicating success
    return jsonify({'success': True, 'message': 'Updated product rating and rating count!'})


# cart
@app.route("/cart")
def cart():
    if 'user_id' in session:
        user_id = session['user_id']
        cart_items = User_product_data.query.filter_by(user_id=user_id).all()
        return render_template('cart.html', cart_items=cart_items)
    else:
        flash('Please log in to view your cart.')
        return redirect(url_for('login'))

# remove item from cart 
@app.route("/remove_from_cart", methods=['POST'])
def remove_from_cart():
    if 'user_id' in session:  # Check if the user is logged in
        user_id = session['user_id']
        product_id = request.form['product_id']  # Get product_id from the form

        # Find the cart item by user_id and product_id
        cart_item = User_product_data.query.filter_by(user_id=user_id, product_id=product_id).first()

        if cart_item:
            db.session.delete(cart_item)
            db.session.commit()
            return jsonify({'success': True, 'message': 'Product removed from cart.'})
        else:
            return jsonify({'success': False, 'message': 'Product not found in cart.'})
    else:
        return jsonify({'success': False, 'message': 'Please log in to manage your cart.'})

# Get product details form the database
def get_product_data():
    try:
        product_data = Product_data.query.all()
        data_list = []
        for data in product_data:
            data_list.append({
                'Prod Id': int(data.prod_id),
                'Product Url': data.product_url,
                'Product Name': data.product_name,
                'Description': data.description,
                'Filtered Description': data.filtered_description,
                'List Price': float(data.list_price),
                'Sale Price': float(data.sale_price),
                'Brand': data.brand,
                'Category': data.category,
                'Filtered Category': data.filtered_category,
                'Available': bool(data.available),
                'Rating': float(data.rating),
                'Rating Count': float(data.rating_count),
                'Tags': data.tags,
            })
        df = pd.DataFrame(data_list)
        return df
    
    except Exception as e:
        return {'error': str(e)}
    
# Get user and product details from the database
def get_user_and_product_data():
    try:
        user_product_data = User_product_data.query.all()
        # Convert the query result into a list of dictionaries
        data_list = []
        for data in user_product_data:
            data_list.append({
                'User Id': int(data.user_id),
                'Prod Id': int(data.product_id),
                'Product Name': data.product_name,
                'List Price': float(data.price),
                'Rating': float(data.rating)
            })
        df = pd.DataFrame(data_list)
        return df
    
    except Exception as e:
        return {'error': str(e)}


# Function to fetch product details
def fetch_product_by_id(product_id, product_data):
    if product_id in list(product_data['Prod Id']):
        return product_data[product_data['Prod Id'] == product_id].iloc[0]

# Route to render the product details page
@app.route('/product/<int:product_id>')
def product_detail(product_id):
    # Fetch product details from your data source using the product_id
    user_data = get_user_and_product_data()
    product_data = get_product_data()
    #print(product_data[['Prod Id', 'Product Name', 'Rating', 'Rating Count']])
    product = fetch_product_by_id(product_id, product_data)

    # Content based recommendations
    content_based_rec, cb_scores, cb_metrics = content_based_recommendations(product_data, product['Product Name'], top_n=20)
    # Item based collaborative recommendations
    specific_item_based_rec, sib_scores, sib_metrics = specific_item_based_recommendation(product_id, user_data, product_data, n_top=20)

    print(f'METRICS: Content Based Recommendations for Product {product['Prod Id']}: {cb_metrics}')
    print(f'METRICS: Specific Item Based Recommendation Scores for Product {product_id}: {sib_metrics}')

    return render_template('product.html', 
                           product=product, 
                           content_based_recommendations=content_based_rec, 
                           item_based_recommendations=specific_item_based_rec)

# routes
@app.route("/home")
def home():
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    prices = train_data['List Price'].head(5).tolist()

    # Pass zip function to Jinja2
    return render_template(
        'index.html',
        signup_success="True",
        trending_products=trending_products.head(5),
        truncate=truncate,
        random_product_image_urls=random_product_image_urls,
        random_price=prices,
        zip=zip  # Passing zip function
    )

if __name__=='__main__':
    app.run(debug=True)