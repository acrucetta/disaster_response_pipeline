#!/usr/bin/env python3

import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how='inner', left_on="id", right_on="id")
    return df

def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # up to the second to last character of each string with slicing
    category_colnames = row.str.slice(stop=-2)

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Iterate through the category columns in df to keep only the last character of each string (the 1 or 0).
    # For example, `related-0` becomes `0`, `related-1` becomes `1`. Convert the string to a numeric value.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1)

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])

    # concatenating original df and cleaned categories column
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()

    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///disaster_response.db')
    df.to_sql(database_filename, engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()