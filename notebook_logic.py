# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import sklearn
import warnings
import random
import sklearn
import scipy
import math
import nltk
import string

# Tokenization
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# Stopwords
from nltk.corpus import stopwords
nltk.download('stopwords')

# Stemming & Lemmatization
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

# ML Model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

warnings.simplefilter('ignore')
pd.set_option('display.max_colwidth', -1)

# %% NEW CELL

from google.colab import drive
drive.mount('/content/drive')

# %% NEW CELL

# Book Data
book_data = pd.read_csv("/content/drive/MyDrive/Machine Learning/Books.csv")

# Users Data
users_data= pd.read_csv("/content/drive/MyDrive/Machine Learning/Users.csv")

# Ratings Data
ratings_data = pd.read_csv("/content/drive/MyDrive/Machine Learning/Ratings.csv")

# %% NEW CELL

# Dataset First Look

# Book Data
book_data.head()

# %% NEW CELL

# Users Data
users_data.head()

# %% NEW CELL

# Ratings Data
ratings_data.head()

# %% NEW CELL

# Dataset Rows & Columns count

# Book Data
book_data.shape

# %% NEW CELL

# User Data
users_data.shape

# %% NEW CELL

# ratings_data
ratings_data.shape

# %% NEW CELL

# Dataset Info
# Book Data
book_data.info()

# %% NEW CELL

# User Data
users_data.info()

# %% NEW CELL

# ratings_data
ratings_data.info()

# %% NEW CELL

# Dataset Duplicate Value Count
# Book Data
book_data.duplicated().sum()

# %% NEW CELL

# User Data
users_data.duplicated().sum()

# %% NEW CELL

# ratings_data
ratings_data.duplicated().sum()

# %% NEW CELL

# Missing Values/Null Values Count
# Book Data
book_data.isnull().sum()

# %% NEW CELL

# User Data
users_data.isnull().sum()

# %% NEW CELL

# ratings_data
ratings_data.isnull().sum()

# %% NEW CELL

# Visualizing the missing values
# Book Data
book_missing_value=book_data.isnull().sum()
columns_with_missing_values = book_missing_value[book_missing_value > 0]      #  Filter columns with missing values

# Calculate the percentage of missing values in each column
total_rows = len(book_data)
percentage_missing = (columns_with_missing_values / total_rows) * 100

# Create a bar chart
plt.figure(figsize=(10, 6))
bar_plot = columns_with_missing_values.plot(kind='bar', color='lightcoral')
plt.xlabel('Columns with Missing Value',fontsize=14)
plt.ylabel('Number of Missing Values',fontsize=14)
plt.title('Number of Missing Values in Book Dataset',fontsize=14)
plt.xticks(rotation=0, ha='center',fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# %% NEW CELL

users_missing_value=users_data.isnull().sum()
columns_with_missing_values = users_missing_value[users_missing_value > 0]      #  Filter columns with missing values

# Calculate the percentage of missing values in each column
total_rows = len(book_data)
percentage_missing = (columns_with_missing_values / total_rows) * 100

# Create a bar chart
plt.figure(figsize=(10, 6))
bar_plot = columns_with_missing_values.plot(kind='bar', color='lightcoral')
plt.xlabel('Columns with Missing Value',fontsize=14)
plt.ylabel('Number of Missing Values',fontsize=14)
plt.title('Number of Missing Values in User Dataset',fontsize=14)
plt.xticks(rotation=0, ha='center',fontsize=14)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the percentage of missing values on top of bar
for index, value in enumerate(columns_with_missing_values):
    plt.text(index, value, f'{percentage_missing[index]:.2f}%', ha='center', va='bottom',fontsize=10)

plt.show()

# %% NEW CELL

# Dataset Columns
# Book Data
book_data.columns

# %% NEW CELL

# User Data
users_data.columns

# %% NEW CELL

# ratings_data
ratings_data.columns

# %% NEW CELL

# Dataset Describe
# Book Data
book_data.describe().T

# %% NEW CELL

# User Data
users_data.describe().T

# %% NEW CELL

# ratings_data
ratings_data.describe().T

# %% NEW CELL

# Check Unique Values for each variable.
# Book Data
book_data.nunique()

# %% NEW CELL

# User Data
users_data.nunique()

# %% NEW CELL

# ratings_data
ratings_data.nunique()

# %% NEW CELL

# Write your code to make your dataset analysis ready.
book_data.rename(columns = {'Book-Title':'title', 'Book-Author':'author', 'Year-Of-Publication':'year', 'Publisher':'publisher'}, inplace=True)

# droping the url
book_data.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis= 1, inplace= True)


# %% NEW CELL

book_data.info()

# %% NEW CELL

book_data.isnull().sum()

# %% NEW CELL

# nan values in book_author column
book_data.loc[(book_data['author'].isnull()),: ]

# %% NEW CELL

# nan values in publisher column
book_data.loc[(book_data['publisher'].isnull()),: ]

# %% NEW CELL

# getting unique value from 'year_of_publication' feature
book_data['year'].unique()

# %% NEW CELL

# Extracting rows with year column="DK Publishing Inc"
book_data[book_data['year'] == 'DK Publishing Inc']

# %% NEW CELL

# Extracting rows with year column="Gallimard"
book_data[book_data['year'] == 'Gallimard']

# %% NEW CELL

book_data.loc[187689]

# %% NEW CELL

book_data.loc[221678]

# %% NEW CELL

book_data.loc[209538]

# %% NEW CELL

book_data.loc[220731]

# %% NEW CELL

# function to fix mismatch data in feature 'book_title', 'book_author', ' year_of_publication', 'publisher'
def replace_df_value(df, idx, col_name, val):
  df.loc[idx, col_name] = val
  return df


# %% NEW CELL

replace_df_value(book_data, 209538, 'title', 'DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)')
replace_df_value(book_data, 209538, 'author', 'Michael Teitelbaum')
replace_df_value(book_data, 209538, 'year', 2000)
replace_df_value(book_data, 209538, 'publisher', 'DK Publishing Inc')

replace_df_value(book_data, 221678, 'title', 'DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)')
replace_df_value(book_data, 221678, 'author', 'James Buckley')
replace_df_value(book_data, 221678, 'year', 2000)
replace_df_value(book_data, 221678, 'publisher', 'DK Publishing Inc')

replace_df_value(book_data, 220731,'title', "Peuple du ciel, suivi de 'Les Bergers")
replace_df_value(book_data, 220731, 'author', 'Jean-Marie Gustave Le ClÃ?Â©zio')
replace_df_value(book_data, 220731, 'year', 2003)
replace_df_value(book_data, 220731, 'publisher', 'Gallimard')

# %% NEW CELL

book_data.loc[209538]

# %% NEW CELL

book_data.loc[221678]

# %% NEW CELL

book_data.loc[220731]

# %% NEW CELL

book_data['year'].unique()

# %% NEW CELL

# Change the datatype of year column from object to int
book_data['year'] = book_data['year'].astype(int)


# %% NEW CELL

book_data.info()

# %% NEW CELL

# Renaming the column
users_data.rename(columns = {'User-ID':'user_id', 'Location':'location', 'Age':'age'}, inplace=True)


# %% NEW CELL

users_data.info()

# %% NEW CELL

# Renamimg the column
ratings_data.rename(columns = {'User-ID':'user_id', 'Book-Rating':'rating'}, inplace=True)

# %% NEW CELL

ratings_data.info()

# %% NEW CELL

ratings_data['rating'].unique()

# %% NEW CELL

ratings_data['user_id'].value_counts()

# %% NEW CELL

x = ratings_data['user_id'].value_counts() > 200

# %% NEW CELL

y = x[x].index  # user_ids
print(y.shape)

# %% NEW CELL

ratings_data = ratings_data[ratings_data['user_id'].isin(y)]

# %% NEW CELL

ratings_data.shape

# %% NEW CELL

number_rating = ratings_data.groupby('ISBN').count()['rating'].reset_index()
number_rating.rename(columns= {'rating':'number_of_ratings'}, inplace=True)
number_rating

# %% NEW CELL

final_rating=ratings_data.merge(number_rating,on='ISBN')
final_rating


# %% NEW CELL

final_rating = final_rating[final_rating['number_of_ratings'] >= 50]

# %% NEW CELL

# Drop Duplicated Row
final_rating.drop_duplicates(['user_id','ISBN'], inplace=True)

# %% NEW CELL

rating_with_books = final_rating.merge(book_data, on='ISBN')


# %% NEW CELL

rating_book_users=rating_with_books.merge(users_data,on='user_id')


# %% NEW CELL

rating_book_users.head()

# %% NEW CELL

# Extract 'country' values from 'location' column
rating_book_users['country'] = rating_book_users['location'].str.split(',').str[-1].str.strip()

# Drop the 'location' column
rating_book_users.drop(columns=['location'], inplace=True)

# %% NEW CELL

rating_book_users['country'].unique()

# %% NEW CELL

# Replace 'usa' with 'us',double quotes (") & 'n/a' with 'nan' in 'country' column
rating_book_users['country'] = rating_book_users['country'].str.replace('usa','us').replace('n/a',np.nan).replace('', np.nan)

# Display the modified 'country' column
rating_book_users['country'].unique()

# %% NEW CELL

rating_book_users.tail()

# %% NEW CELL

rating_book_users.shape

# %% NEW CELL

rating_book_users.info()

# %% NEW CELL

rating_book_users.loc[41803]

# %% NEW CELL

# Chart - 1 visualization code
plt.figure(figsize=(14,6))
ax=sns.countplot(x="rating",palette = 'Paired',data= rating_book_users)
plt.title('Count of Each Ratings',fontsize=15)
plt.xlabel('Rating',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add value annotations to the bars
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.show()

# %% NEW CELL

# Chart - 2 visualization code
plt.figure(figsize=(15,6))
sns.distplot(rating_book_users['age'])
plt.title('Age Distribution\n',fontsize=15)
plt.xlabel('Age',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.show()

# %% NEW CELL

# Chart - 3 visualization code
plt.figure(figsize=(10,6))
sns.boxplot(rating_book_users['age'])
plt.show()

# %% NEW CELL

# Chart - 4 visualization code
filtered_data = rating_book_users[rating_book_users['year'] != 0]

sns.boxplot(x=filtered_data['year'])
plt.xlabel('Year')
plt.title('Distribution of Years in Book Ratings (Excluding Year = 0)')
plt.show()

# %% NEW CELL

# Chart - 5 visualization code
sns.boxplot(x=rating_book_users['number_of_ratings'])
plt.xlabel('Number of Ratings')
plt.title('Distribution of Number of Ratings')
plt.show()

# %% NEW CELL

# Chart - 6 visualization code
sns.boxplot(x=rating_book_users['rating'])
plt.xlabel('Rating')
plt.title('Distribution of Ratings')
plt.show()

# %% NEW CELL

# Chart - 7 visualization code
plt.figure(figsize=(15,6))
ax=sns.countplot(data=rating_book_users, y="author", palette = 'Paired', order=rating_book_users['author'].value_counts().index[0:20])
plt.title("Top 20 author with number of books",fontsize=15)
plt.xlabel("Count of Books",fontsize=15)
plt.ylabel("Author Name",fontsize=15)

# Add values on top of each bar
for p in ax.patches:
    width = p.get_width()
    plt.text(width + 40, p.get_y() + p.get_height() / 2, f'{int(width)}',
             ha='center', va='center', fontsize=10, color='black')
plt.show()

# %% NEW CELL

# Chart - 8 visualization code
plt.figure(figsize=(15, 6))
ax = sns.countplot(data=rating_book_users, y="publisher", palette='Paired',
                   order=rating_book_users['publisher'].value_counts().index[0:20])

# Set the title
plt.title("Top 20 Publishers with the number of books published", fontsize=15)
plt.xlabel("Number of Books", fontsize=15)
plt.ylabel("Publishers Name", fontsize=15)

# Adding values of each bar
for p in ax.patches:
    width = p.get_width()
    plt.text(width + 80, p.get_y() + p.get_height() / 2, f'{int(width)}',
             ha='center', va='center', fontsize=10, color='black')
plt.show()


# %% NEW CELL

# Chart - 9 visualization code
plt.figure(figsize=(12, 6))

# Create a countplot for the top 15 books based on the number of ratings
ax = sns.countplot(y="title", palette='Paired', data=rating_book_users, order=rating_book_users['title'].value_counts().index[0:15])
plt.title("Top 15 Books by Number of Ratings", fontsize=15)
plt.xlabel("Total Number of Ratings Given", fontsize=15)
plt.ylabel("Book Title", fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=8)

# Adding values on top of each bar
for p in ax.patches:
    width = p.get_width()
    plt.text(width + 7, p.get_y() + p.get_height() / 2, f'{int(width)}',
             ha='center', va='center', fontsize=10, color='black')
plt.show()

# %% NEW CELL

# Chart - 10 visualization code
plt.figure(figsize=(15, 10))

# Create a countplot for the number of books published each year
ax=sns.countplot(data=rating_book_users, x="year", palette='Paired', order=sorted(rating_book_users['year'].unique()))

# Set the title and labels
plt.title("Number of Books Published Each Year")
plt.xlabel("Year",fontsize=15)
plt.ylabel("Number of Books",fontsize=15)
plt.xticks(rotation=90)

# Add values on top of each bar
for p in ax.patches:
  ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()+30),
                ha='center', va='bottom', fontsize=10, color='black')
plt.show()

# %% NEW CELL

# Chart - 11 visualization code

plt.figure(figsize=(15, 6))

# Filter out rows where 'year' is not equal to 0
filtered_final_data = rating_book_users[rating_book_users['year'] != 0]

sns.lineplot(x='year', y='number_of_ratings', data=filtered_final_data)

plt.title("Number of Ratings Over the Years", fontsize=15)
plt.xlabel("Year", fontsize=15)
plt.ylabel("Number of Ratings", fontsize=15)
plt.xticks(range(min(filtered_final_data['year']), max(filtered_final_data['year'])+1, 5))
plt.show()

# %% NEW CELL

# Chart - 12 visualization code
# Group the data by 'country' and count the unique 'author' values in each group
country_author_counts = rating_book_users.groupby('country')['author'].nunique()

# Create a bar plot
plt.figure(figsize=(14, 7))
ax=country_author_counts.plot(kind='bar',color='skyblue')
plt.title('Number of Unique Authors by Country',fontsize=15)
plt.xlabel('Country',fontsize=15)
plt.ylabel('Number of Unique Authors',fontsize=15)
plt.xticks(rotation=90,fontsize=11)
plt.tight_layout()

# Add value annotations to the bars
for i, v in enumerate(country_author_counts):
  ax.text(i, v + 1, str(v), ha='center', va='bottom', fontsize=11)
plt.show()

# %% NEW CELL

# Chart - 13 visualization code

# Group the data by 'country' and count the unique 'publisher' values in each group
country_publisher_counts = rating_book_users.groupby('country')['publisher'].nunique()

# Create a bar plot
plt.figure(figsize=(14, 7))
ax = country_publisher_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Unique Publishers by Country', fontsize=15)
plt.xlabel('Country', fontsize=15)
plt.ylabel('Number of Unique Publishers', fontsize=15)
plt.xticks(rotation=90,fontsize=11)

# Add value annotations to the bars
for i, v in enumerate(country_publisher_counts):
    ax.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.show()


# %% NEW CELL

# Correlation Heatmap visualization code

# Calculate the correlation matrix
correlation_matrix = rating_book_users.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of rating_book_users", fontsize=15)
plt.show()

# %% NEW CELL

# Pair Plot visualization code
sns.pairplot(rating_book_users)

# %% NEW CELL

# Perform Statistical Test to obtain P-Value

# Separate the data for users from the USA and Canada
age_usa = rating_book_users[rating_book_users['country'] == 'us']['age'].dropna()
age_canada = rating_book_users[rating_book_users['country'] == 'canada']['age'].dropna()

# Perform a t-test
t_statistic, p_value = stats.ttest_ind(age_usa, age_canada, alternative='two-sided')

# Significance level
alpha = 0.05

# Print results
print("t-statistic:", t_statistic)
print("p-value:", p_value)

if p_value < alpha:
  print("Reject the null hypothesis: The average age of users from the USA is not equal to the average age of users from Canada.")
else:
  print("Fail to reject the null hypothesis: The average age of users from the USA is equal to the average age of users from Canada.")

# %% NEW CELL

# Perform Statistical Test to obtain P-Value

# Separate the data for users who rated books with a rating of 5 and users who rated books with a rating less than 5
age_rating_5 = rating_book_users[rating_book_users['rating'] == 5]['age'].dropna()
age_rating_less_than_5 = rating_book_users[rating_book_users['rating'] < 5]['age'].dropna()

# Perform a one-tailed t-test (greater)
t_stat, p_value = stats.ttest_ind(age_rating_5, age_rating_less_than_5, alternative='two-sided')

# Significance level
alpha = 0.05

# Print results
print("t-statistic:", t_stat)
print("p-value:", p_value)

if p_value < alpha:
  print("Reject the null hypothesis: The average age of users who rated books with a rating of 5 is not equal\
        \nto average age of users who rated books with a rating less than 5.")

else:
  print("Fail to reject the null hypothesis: The average age of users who rated books with a rating of 5 is equal\
        \nto the average age of users who rated books with a rating less than 5.")


# %% NEW CELL

# Perform Statistical Test to obtain P-Value

# Separate the data for books with a rating of 5 and books with a rating less than 5
num_ratings_rating_5 = rating_book_users[rating_book_users['rating'] == 5]['number_of_ratings']
num_ratings_rating_less_than_5 = rating_book_users[rating_book_users['rating'] < 5]['number_of_ratings']

# Perform a one-tailed t-test (greater)
t_stat, p_value = stats.ttest_ind(num_ratings_rating_5, num_ratings_rating_less_than_5, alternative='greater')

# Significance level
alpha = 0.05

# Print results
print("t-statistic:", t_stat)
print("p-value:", p_value)

if p_value < alpha:
    print("Reject the null hypothesis: The average number of ratings for books with a rating of 5\
          \nis higher than the average number of ratings for books with a rating less than 5.")
else:
    print("Fail to reject the null hypothesis: The average number of ratings for books with a rating of 5\
            \nis equal to the average number of ratings for books with a rating less than 5.")


# %% NEW CELL

# Making copy of original dataframe
df=final_rating.copy()

# %% NEW CELL

df.columns

# %% NEW CELL

df.head()

# %% NEW CELL

df.info()

# %% NEW CELL

# Handling Missing Values & Missing Value Imputation
df.isnull().sum()

# %% NEW CELL

book_data.isnull().sum()

# %% NEW CELL

mode_publisher = book_data['publisher'].mode()[0]
mode_author = book_data['author'].mode()[0]

# Fill missing values in 'publisher' and 'author' columns with their respective modes
book_data['publisher'].fillna(mode_publisher, inplace=True)
book_data['author'].fillna(mode_author, inplace=True)

# %% NEW CELL

book_data.isnull().sum()

# %% NEW CELL

# Handling Outliers & Outlier treatments
sns.boxplot(df)

# %% NEW CELL

# Manipulate Features to minimize feature correlation and create new features
avg_rating=df.groupby('ISBN').mean()['rating'].reset_index()
avg_rating.rename(columns= {'rating':'avg_ratings'}, inplace=True)
avg_rating.head()

# %% NEW CELL

avg_rating_df=df.merge(avg_rating,on='ISBN')
avg_rating_df.head()

# %% NEW CELL

df=book_data.copy()

# %% NEW CELL

df.head()

# %% NEW CELL

# Expand Contraction

# %% NEW CELL

# Lower Casing
def string_lower(word):
  return word.lower()
df['title']=df['title'].apply(string_lower)
# df['author']=df['author'].apply(string_lower)
# df['publisher']=df['publisher'].apply(string_lower)

# %% NEW CELL

df.head()

# %% NEW CELL

# Remove Punctuations
[punc for punc in string.punctuation]

# %% NEW CELL

def remove_punc(text):
  nopunc =[char for char in text if char not in string.punctuation]
  nopunc=''.join(nopunc)
  return nopunc
df['title']=df['title'].apply(remove_punc)
# df['author']=df['author'].apply(remove_punc)
# df['publisher']=df['publisher'].apply(remove_punc)

# %% NEW CELL

# Remove URLs & Remove words and digits contain digits

# Function to remove digits from text & sentence
def remove_digits(text):
  return ''.join([char for char in text if not char.isdigit()])

# Apply the remove_digits function to the 'title' column
df['title']=df['title'].apply(remove_digits)

# %% NEW CELL

# Remove Stopword

def remove_stopwords(sentence, language='english'):
  # Get the list of stopwords for the specified language
  stop_words = set(stopwords.words(language))
  words = sentence.split()

  # Remove stopwords from the list of words
  filtered_words = [word for word in words if word not in stop_words]

  # Join the filtered words to form a sentence without stopwords
  filtered_sentence = ' '.join(filtered_words)
  return filtered_sentence

# %% NEW CELL

df['title']=df['title'].apply(remove_stopwords)

# %% NEW CELL

# Remove White spaces
df['title']=df['title'].replace(" ","")

# %% NEW CELL

# Rephrase Text
# Create a new columns & Concatenate all the columns into it
df.head()

# %% NEW CELL

# Tokenize the 'tags' column using nltk
df['title'] = df['title'].apply(word_tokenize)

# Display the result
df

# %% NEW CELL

# Creating a new dataframe
book_data_new=df[['ISBN','title']]
book_data_new

# %% NEW CELL

# Normalizing Text (i.e., Stemming, Lemmatization etc.)

# Create lemmatizer objects
lemmatizer = WordNetLemmatizer()



# Define a function to lemmatize and join tokens
def lemmatize_and_join(tokens):
  # Lemmatize each token and join them back into a single string
  lemmatized_text = " ".join([lemmatizer.lemmatize(token) for token in tokens])

  return lemmatized_text

book_data_new['title']=book_data_new['title'].apply(lemmatize_and_join)
book_data_new.head()

# %% NEW CELL

# Vectorizing Text
stopwords_list = stopwords.words('french') + stopwords.words('portuguese') + stopwords.words('spanish') + stopwords.words('german')+ stopwords.words('finnish')+ stopwords.words('swedish')

#Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus, ignoring stopwords
vectorizer = TfidfVectorizer(analyzer='word',
                     ngram_range=(1, 2),
                     min_df=0.04,
                     max_df=0.7,
                     max_features=5000,
                     stop_words=stopwords_list)
tfidf_matrix = vectorizer.fit_transform(book_data_new['title'])
tfidf_feature_names = vectorizer.get_feature_names_out()
tfidf_matrix

# %% NEW CELL

print(vectorizer.get_feature_names_out())

# %% NEW CELL

tfidf_matrix.shape

# %% NEW CELL

len(vectorizer.get_feature_names_out())

# %% NEW CELL

# Split your data to train and test. Choose Splitting ratio wisely.
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(avg_rating_df,test_size=0.2,random_state=42)

# %% NEW CELL

print(f'Training set lengths: {len(train_data)}')
print(f'Testing set lengths: {len(test_data)}')
print(f'Test set is {(len(test_data)/(len(train_data)+len(test_data))*100):.0f}% of the full dataset.')

# %% NEW CELL

#Indexing by user_id to speed up the searches during evaluation
rating_full_df = avg_rating_df.set_index('user_id')
rating_train_df =train_data.set_index('user_id')
rating_test_df = test_data.set_index('user_id')

# %% NEW CELL

train_data.head()

# %% NEW CELL

#Creating a sparse pivot table with ISBN in rows and user_id in columns
users_items_pivot_matrix_df = train_data.pivot_table(columns='ISBN', index='user_id', values="rating")

# %% NEW CELL

users_items_pivot_matrix_df.shape

# %% NEW CELL

users_items_pivot_matrix_df.head()

# %% NEW CELL

users_items_pivot_matrix_df.fillna(0, inplace=True)

# %% NEW CELL

users_items_pivot_matrix_df.head()

# %% NEW CELL

users_items_pivot_matrix=users_items_pivot_matrix_df.values
users_items_pivot_matrix[:10]

# %% NEW CELL

user_id = list(users_items_pivot_matrix_df.index)
user_id[:10]

# %% NEW CELL

# The number of factors to factor the user-item matrix.
NUMBER_OF_FACTORS_MF = 15

#Performs matrix factorization of the original user item matrix
U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)

# %% NEW CELL

users_items_pivot_matrix.shape

# %% NEW CELL

U.shape

# %% NEW CELL

sigma = np.diag(sigma)
sigma.shape

# %% NEW CELL

Vt.shape

# %% NEW CELL

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
all_user_predicted_ratings

# %% NEW CELL

all_user_predicted_ratings.shape

# %% NEW CELL

#Converting the reconstructed matrix back to a Pandas dataframe
cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns = users_items_pivot_matrix_df.columns, index=user_id).transpose()
cf_preds_df.head()

# %% NEW CELL

len(cf_preds_df.columns)

# %% NEW CELL

class CFRecommender:

    MODEL_NAME = 'Collaborative Filtering'

    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False).reset_index().rename(columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating content that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['ISBN'].isin(items_to_ignore)].sort_values('recStrength', ascending = False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left',
                                                          left_on = 'ISBN',
                                                          right_on = 'ISBN')[['recStrength', 'ISBN','title']]


        return recommendations_df

cf_recommender_model = CFRecommender(cf_preds_df,book_data_new)

# %% NEW CELL

def get_items_interacted_collaborative(user_id, ratings_data):
    interacted_items = ratings_data.loc[user_id]['ISBN']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

# %% NEW CELL

#Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100

class ModelEvaluator_collaborative:

    # Function for getting the set of items which a user has not interacted with
    def get_not_interacted_items_sample(self, user_id, sample_size, seed=42):
        interacted_items = get_items_interacted_collaborative(user_id, rating_full_df)
        all_items = set(book_data_new['ISBN'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    # Function to verify whether a particular item_id was present in the set of top N recommended items
    def _verify_hit_top_n(self, item_id, recommended_items, topn):
            try:
                index = next(i for i, c in enumerate(recommended_items) if c == item_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index

    # Function to evaluate the performance of model for each user
    def evaluate_model_for_user(self, model, user_id):
      try:

        # Getting the items in test set
        interacted_values_testset = rating_test_df.loc[user_id]

        if type(interacted_values_testset['ISBN']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['ISBN'])
        else:
            person_interacted_items_testset = set(interacted_values_testset['ISBN'])

        interacted_items_count_testset = len(person_interacted_items_testset)

        # Getting a ranked recommendation list from the model for a given user
        person_recs_df = model.recommend_items(user_id, items_to_ignore=get_items_interacted_collaborative(user_id, rating_train_df),topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0

        # For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:

            # Getting a random sample of 100 items the user has not interacted with
            non_interacted_items_sample = self.get_not_interacted_items_sample(user_id, sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS)

            # Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            # Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = person_recs_df[person_recs_df['ISBN'].isin(items_to_filter_recs)]
            valid_recs = valid_recs_df['ISBN'].values

            # Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        # Recall is the rate of the interacted items that are ranked among the Top-N recommended items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        user_metrics = {'hits@5_count':hits_at_5_count,
                          'hits@10_count':hits_at_10_count,
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return user_metrics
      except KeyError:
        # Handle the KeyError gracefully, e.g., by returning default metrics or logging the error
        print(f"User with user_id {user_id} not found in the test set.")
        return {'hits@5_count': 0, 'hits@10_count': 0, 'interacted_count': 0, 'recall@5': 0, 'recall@10': 0}


    # Function to evaluate the performance of model at overall level
    def evaluate_model(self, model):

        people_metrics = []

        for idx, user_id in enumerate(list(rating_test_df.index.unique().values)):
            person_metrics = self.evaluate_model_for_user(model, user_id)
            person_metrics['_user_id'] = user_id
            people_metrics.append(person_metrics)

        print('{0} users processed' .format(idx))

        detailed_results_df = pd.DataFrame(people_metrics).sort_values('interacted_count', ascending=False)

        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())

        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}
        return global_metrics, detailed_results_df

model_evaluator = ModelEvaluator_collaborative()

# %% NEW CELL

print('Evaluating Collaborative Filtering (SVD Matrix Factorization) model...')
cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(cf_recommender_model)

# Move the user_id column to the first position
user_id_column = cf_detailed_results_df['_user_id']  # Extract the user_id column
cf_detailed_results_df = cf_detailed_results_df.drop(columns=['_user_id'])
cf_detailed_results_df.insert(0, '_user_id', user_id_column)

print('\nGlobal metrics:\n{}'.format(cf_global_metrics))
cf_detailed_results_df.head(10)

# %% NEW CELL

stopwords_list

# %% NEW CELL

tfidf_feature_names

# %% NEW CELL

item_ids = book_data_new['ISBN'].tolist()

def get_item_profile(item_id):
  try:
    idx = item_ids.index(item_id)
    item_profile = tfidf_matrix[idx:idx+1]
    return item_profile

  except ValueError:
    # Handle the case where the item_id is not found
    print(f"Item with ISBN '{item_id}' not found in the list.")
    return None

def get_item_profiles(ids):
  item_profiles_list = []

  for x in ids:
    item_profile = get_item_profile(x)

    if item_profile is not None:
      # Ensure item_profile is a 2-D matrix with the same number of columns as tfidf_matrix
      if item_profile.shape[1] == tfidf_matrix.shape[1]:
        item_profiles_list.append(item_profile)

  if item_profiles_list:
      item_profiles = np.vstack(item_profiles_list)
      return item_profiles
  else:
      return None


def build_users_profile(user_id, avg_indexed_df):
  interactions_person_df = avg_indexed_df.loc[user_id]
  user_item_profiles = get_item_profiles(interactions_person_df['ISBN'])
  user_item_strengths = np.array(interactions_person_df['avg_ratings']).reshape(-1, 1)
  return user_item_strengths


def build_users_profiles():
  avg_indexed_df = avg_rating_df[avg_rating_df['ISBN'].isin(book_data_new['ISBN'])].set_index('user_id')
  user_profiles = {}
  for user_id in avg_indexed_df.index.unique():
      user_profiles[user_id] = build_users_profile(user_id, avg_indexed_df)
  return user_profiles


# %% NEW CELL

user_profiles = build_users_profiles()
len(user_profiles)

# %% NEW CELL

user_profiles

# %% NEW CELL

user_profile = user_profiles[155916]
print(user_profile.shape)

pd.DataFrame(sorted(zip(tfidf_feature_names,
                        user_profiles[178950].flatten().tolist()), key=lambda x: -x[1])[:20],
             columns=['token', 'relevance'])

# %% NEW CELL

class ContentBasedRecommender:

    MODEL_NAME = 'Content-Based'

    def __init__(self, items_df=None):
        self.item_ids = item_ids
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def _get_similar_items_to_user_profile(self, user_id, topn=1000):

        # Compute the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(user_profiles[user_id], tfidf_matrix)

        # Get the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]

        # Sort the similar items by similarity
        similar_items = sorted([(item_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(user_id)

        #Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))

        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['ISBN', 'recStrength']).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left',
                                                          left_on = 'ISBN',
                                                          right_on = 'ISBN')[['recStrength', 'ISBN','title']]


        return recommendations_df

content_based_recommender_model = ContentBasedRecommender(book_data_new)

# %% NEW CELL

def get_items_interacted_content(user_id, ratings_data):
  interacted_items = ratings_data.loc[user_id]['ISBN']
  return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

# %% NEW CELL

#Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100

class ModelEvaluator_content:

    # Function for getting the set of items which a user has not interacted with
    def get_not_interacted_items_sample(self, user_id, sample_size, seed=42):
      interacted_items = get_items_interacted_content(user_id, rating_full_df)
      all_items = set(book_data_new['ISBN'])
      non_interacted_items = all_items - interacted_items

      random.seed(seed)
      non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
      return set(non_interacted_items_sample)

    # Function to verify whether a particular item_id was present in the set of top N recommended items
    def _verify_hit_top_n(self, item_id, recommended_items, topn):
      try:
        index = next(i for i, c in enumerate(recommended_items) if c == item_id)
      except:
        index = -1
      hit = int(index in range(0, topn))
      return hit, index


    # Function to evaluate the performance of model for each user
    def evaluate_model_for_user(self, model, user_id):
      try:

        # Getting the items in test set
        interacted_values_testset = rating_test_df.loc[user_id]

        if type(interacted_values_testset['ISBN']) == pd.Series:
          person_interacted_items_testset = set(interacted_values_testset['ISBN'])
        else:
          person_interacted_items_testset = set(interacted_values_testset['ISBN'])

        interacted_items_count_testset = len(person_interacted_items_testset)

        # Getting a ranked recommendation list from the model for a given user
        person_recs_df = model.recommend_items(user_id, items_to_ignore=get_items_interacted_content(user_id, rating_train_df),topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0

        # For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:

            # Getting a random sample of 100 items the user has not interacted with
            non_interacted_items_sample = self.get_not_interacted_items_sample(user_id, sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS)

            # Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            # Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = person_recs_df[person_recs_df['ISBN'].isin(items_to_filter_recs)]
            valid_recs = valid_recs_df['ISBN'].values

            # Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        # Recall is the rate of the interacted items that are ranked among the Top-N recommended items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        user_metrics = {'hits@5_count':hits_at_5_count,
                          'hits@10_count':hits_at_10_count,
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return user_metrics
      except KeyError:
        # Handle the KeyError gracefully, e.g., by returning default metrics or logging the error
        print(f"User with user_id {user_id} not found in the test set.")
        return {'hits@5_count': 0, 'hits@10_count': 0, 'interacted_count': 0, 'recall@5': 0, 'recall@10': 0}


    # Function to evaluate the performance of model at overall level
    def evaluate_model(self, model):

        people_metrics = []

        for idx, user_id in enumerate(list(rating_test_df.index.unique().values)):
            person_metrics = self.evaluate_model_for_user(model,user_id)
            person_metrics['_person_id'] = user_id
            people_metrics.append(person_metrics)

        print('{0} users processed' .format(idx))

        detailed_results_df = pd.DataFrame(people_metrics).sort_values('interacted_count', ascending=False)

        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())

        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}
        return global_metrics, detailed_results_df

model_evaluator = ModelEvaluator_content()

# %% NEW CELL

print('Evaluating Content-Based Filtering model...')
cb_global_metrics, cb_detailed_results_df = model_evaluator.evaluate_model(content_based_recommender_model)

# Move the person_id column to the first position
user_id_column = cb_detailed_results_df['_person_id']
cb_detailed_results_df = cb_detailed_results_df.drop(columns=['_person_id'])
cb_detailed_results_df.insert(0, '_person_id', user_id_column)

print('\nGlobal metrics:\n{}' .format(cb_global_metrics))
cb_detailed_results_df.head(10)

# %% NEW CELL

rating_book_users.head()

# %% NEW CELL

rating_book_users.columns

# %% NEW CELL

# Split your data to train and test. Choose Splitting ratio wisely.
train_data_model, test_data_model = train_test_split(rating_book_users, test_size=0.2,random_state=42)

# %% NEW CELL

print(f'Training set lengths: {len(train_data_model)}')
print(f'Testing set lengths: {len(test_data_model)}')
print(f'Test set is {(len(test_data_model)/(len(train_data_model)+len(test_data_model))*100):.0f}% of the full dataset.')

# %% NEW CELL

pivot_matrix = train_data_model.pivot_table(columns='user_id', index='title', values="rating")

pivot_matrix

# %% NEW CELL

pivot_matrix.fillna(0, inplace=True)

# %% NEW CELL

pivot_matrix.head()

# %% NEW CELL

users_items_model_based_pivot_matrix=pivot_matrix.values
users_items_model_based_pivot_matrix[:10]

# %% NEW CELL

title = list(pivot_matrix.index)
title[:10]

# %% NEW CELL

# Converting the pivot matrix into sparse matrix
from scipy.sparse import csr_matrix
book_sparse = csr_matrix(pivot_matrix)


# %% NEW CELL

# ML Model - 3 Implementation
from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(n_neighbors=10, metric='cosine',algorithm='brute')

# Fit the Algorithm
model.fit(book_sparse)

# %% NEW CELL

def auto_recommend_random_books(test_data_model, num_books=5, num_recommendations=5):
  recommended_books_for_each = []

  try:
    # Randomly select 5 books
    random_book_indices = np.random.choice(len(pivot_matrix.index), num_books, replace=False)
    random_book_names = [pivot_matrix.index[i] for i in random_book_indices]

    for book_name in random_book_names:
      # Check if the book_name exists in the index
      if book_name not in pivot_matrix.index:
        raise KeyError(f"'{book_name}' not found in the index.")

      # Fetch index of book_name
      index = np.where(pivot_matrix.index == book_name)[0][0]

      # Find the nearest neighbors
      distances, neighbor_indices = model.kneighbors(book_sparse[index].reshape(1, -1))

      # Exclude the first neighbor, which is the book itself
      similar_items = neighbor_indices[0][1:]
      similar_distances = distances[0][1:]

      # Get the recommended books and distances
      recommended_books_with_distances = [(pivot_matrix.index[i], distance) for i, distance in zip(similar_items[:num_recommendations], similar_distances)]

      recommended_books_for_each.append((book_name, recommended_books_with_distances))

    return recommended_books_for_each

  except KeyError as e:
    print(e)
    return []

recommended_books_for_each = auto_recommend_random_books(test_data_model, num_books=5, num_recommendations=5)

for book_name, recommended_books_with_distances in recommended_books_for_each:
  if recommended_books_with_distances:
    print(f"Recommended books for '{book_name}':")
    for i, (recommended_book, distance) in enumerate(recommended_books_with_distances):
      print(f"{i + 1}. Book: {recommended_book}, Distance: {distance}")
    print()
  else:
    print(f"No recommendations available for '{book_name}'.")

# %% NEW CELL

# Save the File
import pickle

with open('rating_book_users.pkl', 'wb') as file:
    pickle.dump(rating_book_users, file)
file.close()

# %% NEW CELL

# Load the File and predict unseen data.
with open('rating_book_users.pkl', 'rb') as file:
    loaded_model = pickle.load(file)