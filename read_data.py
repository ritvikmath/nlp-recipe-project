import json
import pandas as pd
from pandas.io.json import json_normalize
import re
import matplotlib.pyplot as plt



#create dataframe
with open("../layer1.json") as f:
  data = json.load(f)

#create ingredient, instruction, and recipie dataframes
ingredients = json_normalize(data,record_path='ingredients',meta="id")
instructions = json_normalize(data,record_path='instructions',meta="id")
recipes = pd.DataFrame(data)[["url","title","id"]]
del data

#add websites to data frame
search = []
for values in recipes['url']:
    search.append(re.search('http(.+?).com/',str(values)).group())
recipes["Website"] = search
del search,values

#add number of and length of ingredients and instructions to dataframe
recipes['N_Ingredients'] = ingredients.groupby('id').count().reset_index()["text"]
recipes['N_Instructions'] = instructions.groupby('id').count().reset_index()["text"]
recipes['Len_Ingredients'] = ingredients["text"].str.len().groupby(ingredients["id"]).sum().reset_index()["text"]
recipes['Len_Instructions'] = instructions["text"].str.len().groupby(instructions["id"]).sum().reset_index()["text"]



#subsets to view
head_recipes = recipes.head(100)
head_ingredients = ingredients.head(100)
head_instructions = instructions.head(100)





#unique values of variables
websites = recipes.groupby('Website').count()["id"]
titles = recipes.groupby('title').count()["id"]

#distribution of number of ingredients
plt.hist(recipes['N_Ingredients'], bins=40)
plt.title("Histogram of Number of Ingredients")
plt.xlabel("Number of Ingredients")
plt.ylabel("Number of Recipes")
plt.show()

plt.hist(recipes['Len_Ingredients'], bins=40)
plt.title("Histogram of Length of Ingredients")
plt.xlabel("Length of Ingredients")
plt.ylabel("Number of Recipes")
plt.show()

plt.hist(recipes['Len_Ingredients'] / recipes['N_Ingredients'], bins=40)
plt.title("Histogram of Average Ingredient Length")
plt.xlabel("Ingredient Length per Ingredient")
plt.ylabel("Number of Recipes")
plt.show()

#distribution of number of instructions
plt.hist(recipes['N_Instructions'], bins=40)
plt.title("Histogram of Number of Instructions")
plt.xlabel("Number of Instructions")
plt.ylabel("Number of Recipes")
plt.show()

plt.hist(recipes['Len_Instructions'], bins=40)
plt.title("Histogram of Length of Instructions")
plt.xlabel("Length of Instructions")
plt.ylabel("Number of Recipes")
plt.show()

plt.hist(recipes['Len_Instructions'] / recipes['N_Instructions'], bins=40)
plt.title("Histogram of Average Instruction Length")
plt.xlabel("Instruction Length per Instruction")
plt.ylabel("Number of Recipes")
plt.show()