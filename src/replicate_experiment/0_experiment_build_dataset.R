# NOTE: 1. EBook survey data cleaning

# Demand Estimation with Text and Image Data
# By: Giovanni Compiani, Ilya Morozov, and Stephan Seiler
# RP: Celina Park

# Goal: The goal of this script is to conduct some initial cleaning and processing of the ebook survey data.

# Set CRAN mirror
options(repos = c(CRAN = "https://cran.r-project.org"))

# Install packages if not already installed
install.packages("data.table")
install.packages("tidyverse")
install.packages("readxl")


# Some preliminary commands
rm(list = ls()) # clearing the workspace
library(data.table)
library(tidyverse)
library(readxl)
getwd()

# A.) Some preliminary investigation of the survey data
# survey_data <- fread("../../input/ebook_survey_data.csv", na.strings = "") # reading the csv as a data.table
# # Note: There are currently 11330 observations of 98 variables. Also, everything is read as character. This is most probably due to the first three rows (of the csv) being
# # strings. We may have to do some editing of the csv before reading it (to facilitate columns being read in the correct data type).
# names(survey_data) # checking the variable names (1st row of the csv)
# temp <- survey_data[1] # checking the 2nd row of the csv
# # Note: For several columns, the 2nd row of the csv exactly equals the 1st row of the csv. However, for others they are not the same. My impression is that we can just drop
# # the second row of the csv.
# temp <- survey_data[2] # checking the 3rd row of the csv
# # Note: It also appears to me that the 3rd row of the csv is not really needed. Thus, we could drop it as well
# 
# # Thus, what we do is remove the 2nd and 3rd row of the csv before reading it. The csv with these rows removed is called ebook_survey_data_edited.csv. We do this to ensure
# # that the variables are read in the correct format (numeric, character, etc.).
# survey_data <- survey_data[-c(1, 2), ] 
# fwrite(survey_data, "data/experiment/intermediate/survey_responses/ebook_survey_data_edited.csv") # 11328 obs

# B.) Doing some cleaning of the data
# Reading the data 
survey_data <- fread("data/experiment/input/survey_responses/ebook_survey_data_edited.csv", na.strings = "")
survey_data <- survey_data[554:nrow(survey_data)] # removing the pilot data and the test data (this is from manual inspection)

# Saving the filtered data
# fwrite(survey_data, "data/experiment/intermediate/survey_responses/ebook_survey_data_filtered.csv") # saving as a csv

# Doing some renaming of the columns
names(survey_data) <- tolower(names(survey_data)) # first converting all the variable names to lowercase
names(survey_data) <- str_replace_all(names(survey_data), " ", "_") # replacing all spaces with "_"
setnames(survey_data, c("comprehension_#1", "comprehension_#2"), c("comprehension_1", "comprehension_2"), skip_absent = TRUE) # renaming due to the special characters

# Converting some variables to date-time
survey_data[, `:=`(startdate = as.POSIXct(startdate, format = "%m/%d/%y %H:%M"), 
                   enddate = as.POSIXct(enddate, format = "%m/%d/%y %H:%M"), 
                   recordeddate = as.POSIXct(recordeddate, format = "%m/%d/%y %H:%M"))]

# Replacing curly apostrophes with straight apostrophes
survey_data[, `:=`(selected_title_1 = gsub("’", "'", selected_title_1),
                   selected_title_2 = gsub("’", "'", selected_title_2),
                   order_array_titles_1 = gsub("’", "'", order_array_titles_1),
                   order_array_titles_2 = gsub("’", "'", order_array_titles_2),
                   final_title = gsub("’", "'", final_title))]

# Removing invalid observations 
# Start from 10775 obs
survey_data <- survey_data[comprehension_1 == 2] 
# Now 9860 obs => 915 obs removed

survey_data <- survey_data[comprehension_2 == "The e-book that you would be most likely to buy in an online bookstore if you were presented with this selection of e-books outside this study"]
# Now 9344 obs => 516 obs removed
# 
survey_data <- survey_data[consent == "I agree to participate in the research"] 
# 0 obs removed 

survey_data <- survey_data[progress == 100] # only keeping surveys with complete progress 
# Now 9265 obs => 79 obs removed

survey_data <- survey_data[finished == TRUE] # only keeping surveys that finished 
# 0 obs removed

survey_data <- survey_data[q_totalduration >= 60] # drop users who took less than 60 seconds for the entire survey
# 0 obs removed


# Obtaining an id variable
survey_data[, id_var := 1:nrow(survey_data)]
setcolorder(survey_data, "id_var")
# Note: We already have responseid, but I want to obtain another id variable through which I can order

# Changing the position variables so that it will be from 1-10 instead of 0-9
survey_data[, `:=`(selected_position_1 = selected_position_1 + 1,
                   selected_position_2 = selected_position_2 + 1)]

dir.create("data/experiment/intermediate", showWarnings = FALSE)

# Create directory if it does not exist
dir.create("data/experiment/intermediate/survey_responses", showWarnings = FALSE)

# Saving the cleaned data
fwrite(survey_data, "data/experiment/intermediate/survey_responses/ebook_survey_data_cleaned.csv") # saving as a csv


# Check unique date in the data
survey_data1 <- survey_data[, date := as.Date(startdate)]
unique(survey_data1$date)
# "2024-06-11" "2024-06-14" "2024-06-16" "2024-06-17" "2024-06-20" "2024-06-21" "2024-06-24" "2024-06-25" "2024-06-26" "2024-06-27"

# C.) Obtaining a product_id - ASIN - Title - genre - pages - year dictionary
# Obtaining a product_id, title dictionary
# Relevant variables: order_array_id_1, order_array_titles_1 (using the first observation of the cleaned dataset)
survey_data <- fread("data/experiment/intermediate/survey_responses/ebook_survey_data_cleaned.csv") # saving as a csv

product_ids <- unlist(str_split(survey_data$order_array_id_1[1], ";")) # obtaining the product ids
product_titles <- unlist(str_split(survey_data$order_array_titles_1[1], ";")) # obtaining the product titles
dictionary <- data.table(product_id = as.numeric(product_ids), title = product_titles)[order(product_id)]

# Reading the complete book list
# book_list <- data.table(read_xlsx("../../../pilot/input/books/input/book_list_five_reviews.xlsx"))[, .(`ASIN`, `Title`, `Genre`, `Number of pages`, `Publication Year`)][1:50] # reading the book list
# setnames(book_list, c("Title", "Genre", "Number of pages", "Publication Year"), c("title", "genre", "pages", "year")) # setting some column names to lowercase
# dictionary <- merge(x = dictionary, y = book_list, by = "title", all.x = TRUE) # merging
#
# # Abbreviating some of the genre observations
# dictionary[genre == "Science Fiction & Fantasy", genre := "science_fiction"]
# dictionary[genre == "Self-Help", genre := "self_help"]
# dictionary[genre == "Mystery, Thriller & Suspense", genre := "mystery"]
#
# # Ordering and saving the dictionary
# setcolorder(dictionary, c("product_id", "title", "genre", "ASIN", "year", "pages")) # setting the column order
# dictionary <- dictionary[order(product_id)] # ordering by product_id
# fwrite(dictionary, "data/experiment/intermediate/survey_responses/ebook_product_dictionary.csv") # saving as a csv
#
# NOTE: 2. Choice data generation


# Demand Estimation with Text and Image Data
# By: Giovanni Compiani, Ilya Morozov, and Stephan Seiler
# RP: Celina Park

# Goal: The goal of this script is to conduct a multinomial logit estimation on the survey data, with product fixed effects and the position variable.

# Some preliminary commands
rm(list = ls()) # clearing the workspace

# Install packages if not already installed
install.packages("data.table")
install.packages("tidyverse")
install.packages("logitr")
install.packages("xtable")
install.packages("fastDummies")

library(data.table)
library(tidyverse)
library(logitr)
library(xtable)
library(fastDummies)
getwd() # checking the directory


# A.) First obtaining the needed dataset
# Loading the needed data
survey_data <- fread("data/experiment/intermediate/survey_responses/ebook_survey_data_cleaned.csv") # loading the cleaned survey data
# dictionary <- fread("data/experiment/intermediate/survey_responses/ebook_product_dictionary.csv")# loading the product dictionary

# Only keeping relevant variables and looking at the first and second choices of each survey
# Relevant variables: id_var, selected_id_1, selected_price_1, selected_position_1, order_array_id_1, order_array_prices_1,
# selected_id_2, selected_position_2, order_array_id_2, order_array_prices_2, 
survey_data <- survey_data[, .(id_var, order_array_id_1, order_array_prices_1, selected_id_1, selected_position_1,
                               order_array_id_2, order_array_prices_2, selected_id_2, selected_position_2)]

# 1.) First looking at the first choice situations of each survey
first_choices <- survey_data[, .(id_var, order_array_id_1, order_array_prices_1)] # the first choice situations
first_choices[, order_array_id_1 := str_split(as.character(order_array_id_1), ";")] # separating the semi-colon separated numbers into lists
first_choices[, order_array_prices_1 := str_split(as.character(order_array_prices_1), ";")]

# Changing to long format
first_choices_long <- first_choices[, .(product_id = unlist(order_array_id_1), price = as.numeric(unlist(order_array_prices_1))),
                                    by = "id_var"] # conducting unlisting
first_choices_long[, position := 1:nrow(.SD), by = "id_var"] # obtaining the position variables

# Obtaining the choices
first_choices_long <- merge(x = first_choices_long, y = survey_data[, .(id_var, selected_id_1, selected_position_1)], by = "id_var", all.x = TRUE) # merging to obtain the choices
first_choices_long[, choice := 1*(product_id == selected_id_1), by = "id_var"] # obtaining the choice variable
temp <- first_choices_long[choice == 1] # for checking the position variable in first_choices_long
identical(temp$position, temp$selected_position_1) # TRUE (everything looks good)

# Doing some cleaning
first_choices_long[, c("selected_id_1", "selected_position_1") := NULL] # removing some variables that we don't need
first_choices_long[, product_id := as.numeric(product_id)] # converting to numeric
first_choices_long[, choice_number := 1] # to indicate that this is the first choice situation for the given id_var
setcolorder(first_choices_long, c("id_var", "choice_number", "product_id", "choice")) # changing the column order
first_choices_long <- first_choices_long[order(id_var, product_id)] # ordering

setnames(first_choices_long, "id_var", "respondent_id") # renaming for clarity
first_choices_long[, choice_id := respondent_id] # obtaining another id for choice situations
setcolorder(first_choices_long, c("choice_id")) # setting the choice_id to be the first variable

# Doing some final checking
temp <- first_choices_long[choice == 1]
all(temp$product_id == survey_data$selected_id_1) # all are TRUE
all(temp$position == survey_data$selected_position_1) # all are TRUE

# 2.) Now looking at the second choice situations of each survey
second_choices <- survey_data[, .(id_var, order_array_id_2, order_array_prices_2)] # the first choice situations
second_choices[, order_array_id_2 := str_split(as.character(order_array_id_2), ";")] # separating the semi-colon separated numbers into lists
second_choices[, order_array_prices_2 := str_split(as.character(order_array_prices_2), ";")]

# Changing to long format
second_choices_long <- second_choices[, .(product_id = unlist(order_array_id_2), price = as.numeric(unlist(order_array_prices_2))),
                                      by = "id_var"] # conducting unlisting
second_choices_long[, position := 1:nrow(.SD), by = "id_var"] # obtaining the position variables

# Obtaining the choices
second_choices_long <- merge(x = second_choices_long, y = survey_data[, .(id_var, selected_id_2, selected_position_2)], by = "id_var", all.x = TRUE) # merging to obtain the choices
second_choices_long[, choice := 1*(product_id == selected_id_2), by = "id_var"] # obtaining the choice variable
temp <- second_choices_long[choice == 1] # for checking the position variable in second_choices_long
identical(temp$position, temp$selected_position_2) # TRUE (everything looks good)

# Doing some cleaning
second_choices_long[, c("selected_id_2", "selected_position_2") := NULL] # removing a variable that we don't need
second_choices_long[, product_id := as.numeric(product_id)] # converting to numeric
second_choices_long[, choice_number := 2] # to indicate that this is the second choice situation for the given id_var
setcolorder(second_choices_long, c("id_var", "choice_number", "product_id", "choice")) # changing the column order
second_choices_long <- second_choices_long[order(id_var, product_id)] # ordering

setnames(second_choices_long, "id_var", "respondent_id") # renaming for clarity
second_choices_long[, choice_id := respondent_id] # obtaining another id for choice situations
setcolorder(second_choices_long, c("choice_id")) # setting the choice_id to be the first variable
second_choices_long[, choice_id := max(first_choices_long$choice_id) + choice_id] # since we want to do appending

# Doing some final checking
temp <- second_choices_long[choice == 1]
all(temp$product_id == survey_data$selected_id_2) # all are TRUE
all(temp$position == survey_data$selected_position_2) # all are TRUE 

# 3.) Appending to obtain the final dataset
mult_logit_data <- rbindlist(list(first_choices_long, second_choices_long))
mult_logit_data <- dummy_cols(mult_logit_data, "product_id")
mult_logit_data[, product_id_4 := NULL] # removing one product_id as the base category

# create directory if it does not exist
dir.create("data/experiment/intermediate/survey_responses", showWarnings = FALSE)
fwrite(mult_logit_data, "data/experiment/intermediate/survey_responses/ebook_mult_logit_data.csv") # saving as a csv
# Important variables to note in this dataset:
# choice_id - an ID for a given choice situation or task
# respondent_id - an ID for a given respondent in the ebook survey 
# choice_number - equals 1 or 2, and indicates which choice task is it for the given respondent
# choice - equals 0 or 1, and indicates the product chosen in the given choice task
# Note: For the multinomial logit estimation, we treat the first and second choices for a given respondent as separate choice situations.


# B.) Conducting the multinomial logit estimation for pooled data
covariates <- c("price", "position") # obtaining the covariates vector
product_id_columns <- grep("^product_id_", colnames(mult_logit_data), value = TRUE) # obtaining the product_id columns
covariates <- c(covariates, product_id_columns) # appending to the vector

set.seed(123) # setting the seed (although not really necessary for a multinomial logit estimation)
mult_logit_est <- logitr(data = mult_logit_data, outcome = "choice", obsID = "choice_id", pars = covariates) # conducting the multinomial logit estimation
summary(mult_logit_est) # looking at the results

# Obtaining a summary of the results in latex
results_dt <- as.data.table(tidy(mult_logit_est)) # obtaining a data.table of the results
print(xtable(results_dt, digits = 3), include.rownames = FALSE) # obtaining the latex code


# C.) conducting multinomial logit estimation separately for the first and second choices


# Multinomial Logit Estimation for First Choice Only
first_choice_data <- mult_logit_data[choice_number == 1]
first_choice_logit <- logitr(data = first_choice_data, outcome = "choice", obsID = "choice_id", pars = covariates)
summary(first_choice_logit)

# Multinomial Logit Estimation for Second Choice Only
second_choice_data <- mult_logit_data[choice_number == 2]
second_choice_logit <- logitr(data = second_choice_data, outcome = "choice", obsID = "choice_id", pars = covariates)
summary(second_choice_logit)

# Linear Probability Model (OLS) for Pooled Data
pooled_ols <- lm(choice ~ 0 +price + position + ., data = mult_logit_data[, .SD, .SDcols = c("choice", covariates)])
summary(pooled_ols)

# OLS for First Choice Only
first_choice_ols <- lm(choice ~ 0 +price + position + ., data = first_choice_data[, .SD, .SDcols = c("choice", covariates)])
summary(first_choice_ols)

# OLS for Second Choice Only
second_choice_ols <- lm(choice ~ 0 + price + position + ., data = second_choice_data[, .SD, .SDcols = c("choice", covariates)])
summary(second_choice_ols)

# Save the results as LaTeX tables
# Multinomial Logit for First Choice
first_choice_results_dt <- as.data.table(tidy(first_choice_logit))
first_choice_results_dt[, predictor := rownames(first_choice_results_dt)]
print(xtable(first_choice_results_dt, digits = 3), include.rownames = FALSE)

# Multinomial Logit for Second Choice
second_choice_results_dt <- as.data.table(tidy(second_choice_logit))
second_choice_results_dt[, predictor := rownames(second_choice_results_dt)]
print(xtable(second_choice_results_dt, digits = 3), include.rownames = FALSE)

# OLS for Pooled Data
pooled_ols_results_dt <- as.data.table(broom::tidy(pooled_ols))
print(xtable(pooled_ols_results_dt, digits = 3), include.rownames = FALSE)

# OLS for First Choice Only
first_choice_ols_results_dt <- as.data.table(broom::tidy(first_choice_ols))
print(xtable(first_choice_ols_results_dt, digits = 3), include.rownames = FALSE)

# OLS for Second Choice Only
second_choice_ols_results_dt <- as.data.table(broom::tidy(second_choice_ols))
print(xtable(second_choice_ols_results_dt, digits = 3), include.rownames = FALSE)
