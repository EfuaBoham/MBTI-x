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
        tweets = api.user_timeline(
            screen_name=user_handle, count=number_of_tweets)
        tweets_to_csv = [tweet.text.replace(',', '').replace('\t', '').replace(
            '\n', '') for tweet in tweets if (not tweet.retweeted) and ('RT @' not in tweet.text)]
        # tweets_to_csv = [tweet.text for tweet in tweets]  # create csv file
    except:
        print(user_handle)
        tweets_to_csv = []
    return tweets_to_csv


if __name__ == '__main__':
    handles_file = '../handles.csv'
    output_file = '../mbti_1.csv'
    number_of_tweets = 200
    index = 0
    all_tweets = []
    handles = []
    print_output = ''

    with open(handles_file) as f:
        handles = f.readlines()
        for row in handles:
            handle, personality = row.split(',')
            individual_tweets = extract_tweets_lazy(handle, number_of_tweets)
            if individual_tweets:
                print_output += handle + ',' + personality + '\n'
                num_of_user_tweets = len(individual_tweets)
                for i in range(num_of_user_tweets):
                    print_output += ' ' + ',' + individual_tweets[i] + '\n'
    output = open(output_file, 'w')
    output.write(print_output)
