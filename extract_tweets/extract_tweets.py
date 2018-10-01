import tweepy
import sys


# set encoding
# reload(sys)
# sys.setdefaultencoding('utf8')

key = 'C3rDXiF7xn0Ng7zvkMQbMhMr5'
secret = 'tlUcNSwx1YwyirF1p41cU478nHI7iPrflXkMmv8akRgEYbcxUZ'
access_token = '1445640739-Dw49RlSvZymgDss3t3HoFwiKrtgAC9EvQ7jNEZY'
access_token_secret = 'C6hAhadwQDRR1KDQEZzWd18tjR3MGCDWY01yGcsD6XJm5'

# authenticate to Twitter API
auth = tweepy.OAuthHandler(key, secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# extract tweets of a handle


def extract_tweets_lazy(user_handle, number_of_tweets):
    try:
        tweets = api.user_timeline(screen_name=user_handle, count =number_of_tweets)
        tweets_to_csv = [tweet.text.replace(',', '').replace('\t', '').replace(
            '\n', '') for tweet in tweets if (not tweet.retweeted) and ('RT @' not in tweet.text)]
        # tweets_to_csv = [tweet.text for tweet in tweets]  # create csv file
    except:
        print(user_handle)
        tweets_to_csv = None
    return tweets_to_csv


if __name__ == '__main__':
    handles_file = '../handles.csv'
    output_file = '../mbti_1.csv'
    #output_file = 'test.csv'
    number_of_tweets = 200
    index = 0
    all_tweets = []
    handles = []
    output=''
    with open (handles_file) as f:
        rows = f.readlines()
        for handle in rows:
            handle, personality = handle.split(',')
            if handle:
                tweets = extract_tweets_lazy(handle, number_of_tweets)
                if tweets:
                    output += personality + ',' + '\n'
                    for i in tweets:
                        output += ' ' + ',' + i +'\n'
    with open(output_file, 'w') as o:
        o.write(output)
