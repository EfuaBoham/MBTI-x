from extract_tweets.extract_and_clean_tweets import extract_and_clean_user_tweets
from evaluate.train import predict_personality

t= extract_and_clean_user_tweets('donnashatti987')
p = predict_personality(t,'wiki-news-300d-1M.vec')

print(p)