# %%
##SuperStore ETL & EDA Project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Style
sns.set_style("whitegrid")

# Load Dataset
df = pd.read_csv(r"C:\Users\Nayana\Downloads\Sample - Superstore.csv", encoding="latin1")

print(df.head())
print(df.info())
# %%
##Data Cleaning (ETL)
# Check missing values
print(df.isnull().sum())

# Convert dates
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# Extract Day and Month
df['Day'] = df['Order Date'].dt.day_name()
df['Month'] = df['Order Date'].dt.month_name()

# Remove duplicates
df.drop_duplicates(inplace=True)

print("Shape:", df.shape)
# %%
##1. Exploratory Data Analysis
##Most Frequent Product Sold
top_product = df['Product Name'].value_counts().head(10)

print(top_product)

plt.figure(figsize=(10,5))
top_product.plot(kind='bar')
plt.title("Most Frequent Products Sold")
plt.show()
# %%
##2.Most Ordered Region
region_orders = df['Region'].value_counts()

sns.countplot(data=df, x='Region')
plt.title("Orders by Region")
plt.show()
# %%
##3.Most Ordered State
top_states = df['State'].value_counts().head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=top_states.values,
            y=top_states.index)

plt.title("Top Ordered States")
plt.show()
# %%
##4.Most Ordered City
top_city = df['City'].value_counts().head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=top_city.values,
            y=top_city.index)

plt.title("Top Ordered Cities")
plt.show()
# %%
##5.Highest Discount Given
# noinspection PyStatementEffect
discount_customer = df.groupby('Customer Name')['Discount'].sum()

print(discount_customer.sort_values(ascending=False).head(10))
# %%
##6.Most Sold Category
sns.countplot(data=df, x='Category')
plt.title("Most Sold Category")
plt.show()
# %%
##7.Most Sold Sub-Category
plt.figure(figsize=(12,6))

df['Sub-Category'].value_counts().plot(kind='bar')

plt.title("Most Sold Sub-Category")
plt.show()
# %%
##8.Most Loyal Customer
loyal_customer = df['Customer Name'].value_counts()

print(loyal_customer.head(10))
# %%
##9.Best Business Day
day_sales = df.groupby('Day')['Sales'].sum()

day_sales.sort_values().plot(kind='bar')

plt.title("Sales by Day")
plt.show()
# %%
##10.Best Business Month
month_sales = df.groupby('Month')['Sales'].sum()

month_sales.sort_values().plot(kind='bar')

plt.title("Sales by Month")
plt.show()
# %%
##2. Sales & Profit Analysis
##1.Sales-Profit Ratio per Customer
customer_ratio = df.groupby('Customer Name')[['Sales','Profit']].sum()

customer_ratio['Ratio'] = (
    customer_ratio['Profit']
    / customer_ratio['Sales']
)

print(customer_ratio.head())
# %%
##2.Region Sales & Profit
region_sp = df.groupby('Region')[['Sales','Profit']].sum()

print(region_sp)

region_sp.plot(kind='bar')
plt.title("Region Sales & Profit")
plt.show()
# %%
##3.State Sales & Profit
state_sp = df.groupby('State')[['Sales','Profit']].sum()

print(state_sp.sort_values('Sales', ascending=False).head(10))
# %%
##4.Most Profitable Segment
segment_profit = df.groupby('Segment')['Profit'].sum()

segment_profit.plot(kind='bar')
plt.title("Profit by Segment")
plt.show()
# %%
##5.Most Profitable Product
product_profit = df.groupby('Product Name')['Profit'].sum()

print(product_profit.sort_values(
      ascending=False).head(10))
# %%
##6.Best Seller Category
category_sales = df.groupby('Category')['Sales'].sum()

category_sales.plot(kind='pie',
                    autopct='%1.1f%%')

plt.title("Category Sales Share")
plt.show()
# %%
##3. Bivariate Analysis
##1.Profitable Category by Region
pivot1 = pd.pivot_table(
    df,
    values='Profit',
    index='Region',
    columns='Category',
    aggfunc='sum'
)

sns.heatmap(pivot1,
            annot=True,
            cmap='YlGnBu')

plt.title("Profit by Category & Region")
plt.show()
# %%
##2.Best Seller Day by Category
#Create Day column from Order Date First
df['Order Date']=pd.to_datetime(df['Order Date'])
df['Day']=df['Order Date'].dt.day_name()
pivot2 = pd.crosstab(
    df['Day'],
    df['Category'])

pivot2.plot(kind='bar',figsize=(10,5))
plt.title("Category Sales by Day")
plt.xlabel('Day of Week')
plt.ylabel('Number of Orders')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# %%
##3.Profitable Ship Mode by Region
pivot3 = pd.pivot_table(
    df,
    values='Profit',
    index='Region',
    columns='Ship Mode',
    aggfunc='sum'
)

sns.heatmap(pivot3,
            annot=True)

plt.title("Profit by Ship Mode")
plt.show()
# %%
##4.Best Seller Month by Category
pivot4 = pd.crosstab(
    df['Month'],
    df['Category']
)

pivot4.plot(figsize=(12,6))
plt.title("Month vs Category")
plt.show()
# %%
##5.Best Seller Sub-Category by Segment
pivot5 = pd.crosstab(
    df['Sub-Category'],
    df['Segment']
)

pivot5.plot(kind='bar',
            figsize=(14,6))

plt.title("Sub-Category by Segment")
plt.show()
# %%
##4. Multivariate Analysis
##1.Correlation Analysis
corr = df[['Sales',
           'Quantity',
           'Discount',
           'Profit']].corr()

plt.figure(figsize=(8,5))

sns.heatmap(corr,
            annot=True,
            cmap='coolwarm')

plt.title("Correlation Matrix")
plt.show()
# %%
##Project Conclusion

##Print key insights:
print("Top Product Sold:")
print(df['Product Name'].value_counts().head(1))

print("\nMost Loyal Customer:")
print(df['Customer Name'].value_counts().head(1))

print("\nHighest Profit Region:")
print(df.groupby('Region')['Profit'].sum().idxmax())

print("\nHighest Sales Category:")
print(df.groupby('Category')['Sales'].sum().idxmax())