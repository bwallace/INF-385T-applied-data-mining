

from collections import defaultdict
import csv 

import pdb
import gensim
from gensim import matutils, corpora
from gensim.models.ldamodel import LdaModel
import pandas as pd
import nltk
import statsmodels 
import matplotlib as plt  
import seaborn as sns 
# for language classification
import langid

tags = ["pap smear",
        "pap test",
        "HPV",
        "human papillomavirus",
        "HPV vaccination",
        "Gardasil",
        "cervical cancer",
        "#GoingToTheDoctor",
        "#WomensHealth",
        "colonoscopy",
        "cancer prevention",
        "cancer screening",
        "mammogram",
        "vaxx",
        "#fightcancer",
        "#stopcancerb4itstarts",
        "#screened",
        "#vaccinated",
        "#crc"]


def which_tags(tweet, merge_list=None):
    if merge_list is None: 
        # @TODO extend!
        merge_list = [("pap smear", "pap test"), 
                      ("HPV", "human papillomavirus"), 
                      ("cancer prevention", "cancer screening")]

    tweet_lowercased = tweet.lower()
    tag_set = []
    for t in tags: 
        if t.lower() in tweet_lowercased:
            for to_merge in merge_list:
                if t in to_merge:
                    t = to_merge[0]
                    break
            tag_set.append(t)
    tag_set = list(set(tag_set)) 
    if len(tag_set) == 0:
        tag_set.append("other")

    return tag_set

def read_data(path="CancerReport-clean-whitelisted-en.txt"):
    data = pd.read_csv(path, delimiter="\t")
    ''' 
    the data did not come with any header info (column names).
    so for now setting at least the column names for at least 
    the main columns of interest. 
    @note emailed the annenberg folks 10/21/15 with 
            request for headers.
    '''
    #data.columns.values[1] = "tweet"
    #data.columns.values[2] = "date"
    return data

def fit_lda(X, vocab, num_topics=10, passes=20, alpha=0.001):
    ''' fit LDA from a scipy CSR matrix (X). '''
    print("fitting lda...")
    return LdaModel(matutils.Sparse2Corpus(X), num_topics=num_topics,
                    passes=passes, alpha=alpha, 
                    id2word=dict([(i, s) for i, s in enumerate(vocab)]))

def gen_lda_model(toked_tweets, num_topics=10):
    dictionary = corpora.Dictionary(toked_tweets)
    gensim_corpus = [dictionary.doc2bow(tweet) for tweet in toked_tweets]
    lda = LdaModel(gensim_corpus, num_topics=num_topics,
                    passes=10, alpha=0.001, id2word=dictionary)

    return lda, gensim_corpus, dictionary 
    

def _seems_to_be_about_soccer(tweet):  
    ''' does this tweet seem to be about soccer???'''
    if isinstance(tweet, str):
        tweet = gensim.utils.tokenize(tweet, lower=True)

    terms = ["worldcup", "ger", "usavcrc", "fra", "italia", 
                "mexvcrc", "#mexvcrc", "nedvscrc", "#nedvscrc", 
                "nedcrc", "#nedcrc", "itavscrc", "#itavscrc", 
                "uruvscrc", "#uruvscrc", "worldcup2014", "#worldcup2014",
                "uruguay", "usavcrc", "costa", "rica", "uravscrc", "ussoccer",
                "ussoccer_wnt", "fifa", "itavscrc", "victorytour", "nedvscrc",
                "crclub", "#nedvscrc", "#worldcup2014", "brazil2014",
                "netherlandsVscostarica", "netherlands"]

    return any([t.lower() in terms for t in tweet])

def gen_data_for_slda(original_path="CancerReport.txt", 
                      clean_path="CancerReport-clean-all-data.txt", 
                      use_whitelist=False,
                      en_only=True, THRESHOLD=.6):
    '''
    Redundant with clean_data below, but keeps all headers, basically.
    Also performs more aggressive filtering of non-en messages.
    '''
    headers = ['tweet_id', 'tweet_text', 'tweet_created_at', 'in_reply_to_status_id_str', 'in_reply_to_screen_name', 'retweet_count', 'favorite_count', 'machine_translated_language', 'geo_lat', 'geo_long', 'country_code', 'location_full_name', 'source', 'truncated', 'screen_name', 'user_created_at', 'person_name', 'statuses_count', 'friends_count', 'followers_count', 'user_profile_location', 'user_profile_language', 'media_url', 'expanded_urls', 'tweet_urls', 'hashtag_text', 'usermentions_screen_name', 'retweet', 'retweet_id_str', 'retweet_text', 'retweet_created_at', 'retweet_screen_name', 'retweet_user_created_at', 'retweet_person_name', 'retweet_statuses_count', 'retweet_friends_count', 'retweet_followers_count', 'retweet_urls', 'retweet_hashtag_text', 'retweet_usermentions_screen_name']
    out_str = [headers]
    tweet_idx = headers.index("tweet_text")
    expected_num_cols = len(headers) # 40.
    skipped_count = 0
    ignored = []

    whitelist = []
    if use_whitelist:
        whitelist = ["papsmear", "cervicalcancer", "pap smear", "hpv",
                    "gardasil", "cervical cancer"]

    def _contains_term_from(tweet, whitelist):
        return any([term in tweet.lower() for term in whitelist])

    #tags = [snowball.which_tags(t) for t in raw_tweets]
    #raw_tweets = [t for i,t in enumerate(raw_tweets) if 
    #                    not "#crc" in tags[i]]

    #tokenized_tweets = [word_tokenize(tw) for tw in raw_tweets]
    with open(original_path, 'rU') as orig_data:
        csv_reader = csv.reader(orig_data, delimiter="\t")
        for line in csv_reader:
            if len(line) != expected_num_cols:
                skipped_count += 1
            else: 
                tweet = line[tweet_idx]
                tags = which_tags(tweet)
                lang_pred = langid.classify(tweet)
                # note that I'm including "de" here because for whatever 
                # reason langid kept making this mistake on english tweets. 
                # I think we should see relatively few actual German
                # tweets anyway.
                #### added 'fr' 12/2

                if (en_only and lang_pred[0] not in ("en", "de", "sq", "fr") and 
                        lang_pred[1] > THRESHOLD) or (_seems_to_be_about_soccer(tweet)) or (
                        "#crc" in tags) or (not _contains_term_from(tweet, whitelist)):
                    ignored.append(line[1])
                else:
                    out_str.append(line)
    
    if en_only:
        clean_path = clean_path.replace(".txt", "-en.txt")

    with open(clean_path, 'w') as out_f:
        csv_writer = csv.writer(out_f, delimiter="\t")
        csv_writer.writerows(out_str)

    return out_str, ignored

def _count_up_retweets(grouped_retweets):
    '''
    attempt to count the number of retweets 
    corresponding to the original tweets pointed 
    to by those comprising 'grouped_retweets'
    (the grouping here is assumed to be by 
    retweet_id_str). this is probably imperfect! 
    
    my understanding is that the retweet counts 
    store the # of retweets *at the time when the 
    retweet under consideration was collected*. 
    therefore, many of these are 0, because they were 
    the first retweet. as a somewhat heuristic proxy, 
    I'm here taking the max over the max(retweet counts) 
    and the number of reweets in our corpus. 
    '''
    orig_tweet_ids, orig_tweeter_names, orig_follower_counts, orig_tweet_texts, counts = [], [], [], [], []
    retweet_ids = []
    for orig_tweet_id, cur_retweets in grouped_retweets:
        cur_count = cur_retweets["retweet_count"].max() 
        cur_count = max(cur_count, cur_retweets.shape[0])
        counts.append(cur_count)
        #
        orig_follower_counts.append(cur_retweets["followers_count"].values[0])
        orig_tweet_texts.append(cur_retweets["retweet_text"].values[0])
        orig_tweeter_names.append(cur_retweets["retweet_screen_name"].values[0])
        orig_tweet_ids.append(cur_retweets["retweet_id_str"].values[0])

        retweet_ids.append(cur_retweets['retweet_id_str'].values[0])
    return orig_tweet_ids, orig_tweeter_names, orig_follower_counts, orig_tweet_texts, retweet_ids, counts 

def retweet_analysis():
    tweet_data = pd.read_csv("CancerReport-clean-all-data-en.txt", delimiter="\t", low_memory=False)
    
    # read out and process retweets 
    retweets = tweet_data[tweet_data["retweet"] == True]
    grouped_retweets = retweets.groupby("retweet_id_str")
    orig_tweet_texts, retweet_counts = _count_up_retweets(grouped_retweets)

    ''' @TODO tmp tmp tmp re-tweet issue! @TODO ''' 
    dup_retweets=[x for x in retweet_texts if retweet_texts.count(x)>1]
    retweets[retweets["retweet_text"]==dup_retweets[0]]["tweet_text"].tolist()
    # WTF going on?
    pdb.set_trace()
    ''' end tmp ''' 

    primary_tweets = tweet_data[tweet_data["retweet"] == False]
    # now merge tweet sets (retweeted and not)
    orig_tweet_texts.extend(primary_tweets["tweet_text"].values)
    retweet_counts.extend([0]*primary_tweets.shape[0])
    pdb.set_trace()
    # topic modeling 
    toked_tweets, kept_indices = snowball.build_gensim_corpus(orig_tweet_texts, split_up_by_tag=False)
    lda, gensim_corpus, dict_ = snowball.gen_lda_model(toked_tweets)
    inferred_topic_matrix = lda.inference(gensim_corpus)[0]
    # remove the tweets that were cleaned/not included in gensim corpus
    retweet_counts = [retweet_counts[idx] for idx in kept_indices]
    orig_tweet_texts = [orig_tweet_texts[idx] for idx in kept_indices]

def clean_data(original_path="CancerReport.txt", clean_path="CancerReport-clean.txt", 
                    en_only=True, THRESHOLD=.6):
    ''' 
    Read and clean the data originally provided data, 
    which was messy in that it often contained an inconsistent number 
    of columns, due to the tweets often containing tabs (which were 
    also being used as delimiters!). here we spit out a new file, 
    where we just skip those lines. 

    If the en_only flag is true here, we skip lines not classified
    by langid as English. 
    '''
    expected_num_cols = 40
    skipped_count = 0
    not_english = []
    out_str = [["id", "tweet", "date", "tweeter_name", "tweeter_info"]]
    cols = [0, 1, 2, 14, 16]
    with open(original_path, 'rU') as orig_data:
        csv_reader = csv.reader(orig_data, delimiter="\t")
        for line in csv_reader:
            if len(line) != expected_num_cols:
                skipped_count += 1
            else: 
                cols_of_interest = [line[j] for j in cols]
                lang_pred = langid.classify(cols_of_interest[1])
                # note that I'm including "de" here because for whatever 
                # reason langid kept making this mistake on english tweets. 
                # I think we should see relatively few actual German
                # tweets anyway.
                #### added 'fr' 12/2
                if en_only and lang_pred[0] not in ("en", "de", "sq", "fr") and lang_pred[1] > THRESHOLD:
                    not_english.append(cols_of_interest)
                else:
                    #if not contains_tag(cols_of_interest[1]):
                    #    pdb.set_trace()


                    out_str.append(cols_of_interest)
                    #pdb.set_trace()
            
    if en_only:
        clean_path = clean_path.replace(".txt", "-en.txt")

    #pdb.set_trace()
    with open(clean_path, 'w') as out_f:
        csv_writer = csv.writer(out_f, delimiter="\t")
        csv_writer.writerows(out_str)


def write_out_raw_tweets_with_tags(tags_to_raw_tweets, output_path="tweets-by-keyword.csv"):
    out_str = [["hashtag", "tweet"]]
    for tag, tweets_for_tag in tags_to_raw_tweets.items():
        for t in tweets_for_tag:
            out_str.append([tag, t])


    with open(output_path, 'w') as out_f:  
        w = csv.writer(out_f, delimiter="\t")
        w.writerows(out_str)

def build_gensim_corpus(tweets, at_least=5, split_up_by_tag=False):
    # record frequencies
    STOP_WORDS = nltk.corpus.stopwords.words('english')

    # first tokenize
    toked_tweets = [list(gensim.utils.tokenize(t, lower=True)) for t in tweets]
    
    # counts
    frequency = defaultdict(int)
    for t in toked_tweets:
        for token in t:
            frequency[token] += 1

    # only used if we split by tags though.
    tags_to_tweets = defaultdict(list) 
    # also store raw tweets
    tags_to_raw_tweets = defaultdict(list)
    cleaned_toked = []
    kept_indices = [] # keep track of the tweets we hold on to
    for tweet_idx, tweet in enumerate(toked_tweets):
        cur_t = []

        for token in tweet:
            if (frequency[token] >= at_least and 
                not token in STOP_WORDS and
                len(token) > 1):
                    cur_t.append(token)


        if len(cur_t) > 0:
            kept_indices.append(tweet_idx)
            if not split_up_by_tag:
                cleaned_toked.append(cur_t)
            else: 
                orig_tweet = tweets[tweet_idx]
                tag_set = which_tags(orig_tweet)
                for t in tag_set:
                    tags_to_tweets[t].append(cur_t)
                    tags_to_raw_tweets[t].append(orig_tweet)
        

    if split_up_by_tag:
        return tags_to_tweets, tags_to_raw_tweets

    return cleaned_toked, kept_indices









