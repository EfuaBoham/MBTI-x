#####################################################################################
# Before running, ensure all all the following R pacakges and all their dependencies are installed
# 1. ggplot2
# 2. plyr
# 3. dplyr
# 4. tm
# 5. SnowballC
# 6. wordcloud


########### SET WORKING DIRECORY(folder-contining data). Change the path to match your system settings ####
setwd("~/Documents/Ashesi/Senior-Year/Spring/Machine Learning/Group Project/final-project/")
###########################################################


# ********************************************
 # Reading the data for basinc visualization process
data = read.csv("word_counts.vsv", header=FALSE)

# group words into the distinct personality types
library(dplyr)
grouped_words = data %>% group_by(V1) %>% summarise(mean(V2))


total_number_of_Words = sum(data$V2)
print(total_number_of_Words)


#load plyr
library(plyr)
distribution_of_types = as.data.frame(table(data$V1))

#load ggplot for basic bargraphs
library(ggplot2)
countPlots =  ggplot(distribution_of_types, aes(x=Var1, y=Freq)) + geom_bar(stat="Identity", fill = 'skyblue1', width=.7) 
+ scale_x_discrete(labels = levels(distribution_of_types$Var1))
countPlots = countPlots + ggtitle("Distribution of personality types in data") 
countPlots +  labs(x = "Personality Types", y = "Frequency")


plot = ggplot(grouped_words, aes(x=types, y=mean_word_counts)) + geom_bar(stat="Identity", fill="skyblue3", width=.7) 
   + scale_x_discrete(labels = levels(grouped_words$types))

plot = plot + ggtitle("Distribution of word counts") 
plot +  labs(x = "Personality Types", y = "Mean Word Counts")



##################################################################################


# VISUALIZING THE WORDS CLOUD
posts = read.csv("valid_words.csv", header=FALSE, stringsAsFactors = FALSE)

# Loading Required Libraries
library(tm)
library(SnowballC)
library(wordcloud)
# ***********************************************

################################################
# Cleaning and stemming the posts - cleanning not necessary since the posts are already cleaned

# ************************************
# Working all the post
post_corpus = Corpus(VectorSource(posts$V2))
post_corpus = tm_map(post_corpus, removePunctuation)
post_corpus <- tm_map(post_corpus, removeWords, stopwords('english'))
post_corpus <- tm_map(post_corpus, stemDocument)

#******************************************************

# Working with individual personality types
# *******************************************
# Personality type with the highest posts
infj = subset(posts, V1 == 'infp')  # recorded the highest number of posts
infj_copus = Corpus(VectorSource(infj$V2))
infj_copus = tm_map(infj_copus, removePunctuation)
infj_copus = tm_map(infj_copus, removeWords, stopwords('english'))
infj_copus = tm_map(infj_copus, stemDocument)

# *****************************************
# Personality type with the fewest posts
estj = subset(posts, V1 == 'estj')  # recorded fewer appearances
estJ_copus = Corpus(VectorSource(estj$V2))
estJ_copus = tm_map(estJ_copus, removePunctuation)
estJ_copus = tm_map(estJ_copus, removeWords, stopwords('english'))
estJ_copus = tm_map(estJ_copus, stemDocument)

# *************************************

############################################
# You need to export pictures manually

# Plot the word cloud
pal <- brewer.pal(9, "Spectral") # different collor palet
wordcloud(post_corpus, max.words = 100, random.order = FALSE, colors = pal)
wordcloud(infj_copus, max.words = 100, random.order = FALSE, colors = pal)
wordcloud(estJ_copus, max.words = 100, random.order = FALSE, colors = pal)
