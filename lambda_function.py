'''
# Summary:
# Accepts the raw data in csv
# data cleansing
# sentiment analysis using AWS backrock (Amazon Titan)
# Sends API calls in batches and paralell execution
# saves the final DF as parquet
# Change Log:
7/10/2024   Madhur Agarwal  Initial
8/10/2024   Madhur Agarwal  Addedd batches, parallelism, S3, archival
'''
import pandas as pd
import re
import boto3
import json 
import io 
import nltk
import time
import random
from tqdm import tqdm
from nltk.corpus import stopwords
from concurrent.futures import ThreadPoolExecutor

## 1. DEFINE FUNCTIONS

def read_csv_from_s3(bucket_name, key):
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    return pd.read_csv(io.BytesIO(response['Body'].read()))

def data_clean(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#', '', text)  # Remove the '#' symbol
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters and punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stop words
    return text

def analyze_sentiment(text):
    prompt = f"Classify the following tweet as positive, negative, or neutral:\n\n{text}\n\nAnswer as 'positive', 'negative', or 'neutral'."
    request = {
        "inputText": prompt,
        "textGenerationConfig": {"temperature": 0, "topP": 1}
    }
    request = json.dumps(request)
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = bedrock_client.invoke_model(
                modelId=model_id,
                body=request,
                accept='application/json',
                contentType='application/json'
            )
            response_data = json.loads(response.get('body').read())
            
            # Extract the sentiment from the response
            sentiment = response_data.get('results')[0].get('outputText').strip().lower()

            # Extract only "positive", "negative", or "neutral" from the string
            sentiment_clean = re.search(r'(positive|negative|neutral)', sentiment)
            
            if sentiment_clean:
                return sentiment_clean.group(0)  # Return the cleaned sentiment (either "positive", "negative", or "neutral")
            else:
                return "error"  # Return error if sentiment is not found in response

        except Exception as e:
            if "ThrottlingException" in str(e):
                wait_time = 2 ** attempt + random.uniform(0, 1)  # Exponential backoff with some randomness
                print(f"Throttling detected. Waiting for {wait_time:.2f} seconds before retrying...")
                time.sleep(wait_time)  # Wait before retrying
            else:
                print(f"Error analyzing sentiment: {e}")
                return "error"
    return "error"  # If all retries fail, return error

def analyze_sentiment_batch(texts):
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Use map to apply analyze_sentiment function to the batch of texts
        return list(executor.map(analyze_sentiment, texts))

def save_to_s3(df, bucket_name, file_name):
    """Save DataFrame to S3 in Parquet format."""
    # Convert DataFrame to Parquet format in memory
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine='pyarrow')
    buffer.seek(0)  # Seek to the beginning of the BytesIO buffer

    # Upload the buffer to S3
    # s3_client = boto3.client('s3')
    s3_client.upload_fileobj(buffer, bucket_name, file_name)

def archive_file(src_bucket, key, archive_bucket):
    copy_source = {'Bucket': src_bucket, 'Key': key}

    # Copy the file to the archive bucket
    s3_client.copy_object(CopySource=copy_source, Bucket=archive_bucket, Key=key)
    print(f"File {key} successfully archived to {archive_bucket}.")
    
    # Delete the file from the source bucket after successful copy
    s3_client.delete_object(Bucket=src_bucket, Key=key)
    print(f"File {key} deleted from {src_bucket}.")

## 2. DATA INGESTION
s3_client = boto3.client('s3')
s3_resource = boto3.resource('s3')
src_bucket_name = 'datawranglerraww'
src_bucket = s3_resource.Bucket(src_bucket_name)

# Process each CSV file in the bucket
for obj in src_bucket.objects.all():
    if obj.key.endswith('.csv'):  # Only process CSV files
        print(f"Processing file: {obj.key}")
        df_raw = read_csv_from_s3(src_bucket_name, obj.key)
        df_raw.rename(columns={'Unnamed: 0': 'rowid'}, inplace=True)

## 3. INITIALIZE BEDROCK CLIENT
bedrock_client = boto3.client("bedrock-runtime", region_name="ap-south-1")
model_id = "amazon.titan-text-express-v1"

## 4. DATA CLEANSING

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

selected_columns = df_raw[['rowid', 'base_tweet', 'yyyymmdd', 'language']]

# filter on language
df_filtered = selected_columns[selected_columns['language'].str.lower().isin(['en', 'eng', 'english'])]
    
# get NOT NULL base_tweet
df_filtered = df_filtered.dropna(subset=['base_tweet'])
    
# Extract 'yyyymmdd' from the 'date' column
#df_filtered['date'] = pd.to_datetime(df_filtered['date']).dt.strftime('%Y%m%d')
    
# Convert 'rowid' to numeric
df_filtered['rowid'] = pd.to_numeric(df_filtered['rowid'], errors='coerce')
df_filtered = df_filtered.dropna(subset=['rowid'])

# Normalize 'language' to lowercase
df_filtered.loc[:, 'language'] = df_filtered['language'].str.lower()

# Remove duplicates
df_filtered = df_filtered.drop_duplicates()

# clean tweets
df_filtered['cleaned_tweet'] = df_filtered['base_tweet'].apply(data_clean)

# Filter out tweets with less than 100 characters
df_filtered = df_filtered[df_filtered['cleaned_tweet'].str.len() > 100]

# only include fields of interest
df_filtered = df_filtered[['yyyymmdd', 'cleaned_tweet']]

## 5. SENTIMENT ANALYSIS ON CLEANSED DATA
batch_size = 10
sentiments = []
for i in tqdm(range(0, len(df_filtered), batch_size), desc="Performing sentiment analysis in batches"):
    batch = df_filtered['cleaned_tweet'].iloc[i:i + batch_size].tolist()
    batch_sentiments = analyze_sentiment_batch(batch)
    
    # Ensure the length of returned sentiments matches the batch size
    if len(batch_sentiments) != len(batch):
        raise ValueError(f"Batch processing returned {len(batch_sentiments)} results, expected {len(batch)}")
    
    # Append the sentiments to the list
    sentiments.extend(batch_sentiments) 
    
# Verify if the length of sentiments matches the total rows
if len(sentiments) != len(df_filtered):
    raise ValueError(f"Length of sentiments ({len(sentiments)}) does not match the DataFrame length ({len(df_filtered)})")

# Assign the sentiment values to the DataFrame
df_filtered['sentiment'] = sentiments

print(df_filtered)

# 6. DATA LOAD
# Save the cleansed data along with sentiments in parquet format
tgt_bucket_name = 'datawranglerclean'
output_file_name = f"cleaned_data_{obj.key.split('/')[-1].replace('.csv', '.parquet')}"
save_to_s3(df_filtered, tgt_bucket_name, output_file_name)
print(f"Saved cleaned data to S3 as: {output_file_name}")

# 7. ARCHIVE RAW DATA 
archive_bucket_name = 'datawranglerarchive'
archive_file(src_bucket_name, obj.key, archive_bucket_name)