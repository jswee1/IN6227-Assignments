import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 
                'capital-loss', 'hours-per-week', 'native-country', 'income']

# Load dataset (replace 'your_file_path' with the correct file path)
#adult_data = pd.read_csv('adult.data', delimiter=",", header=None, names=column_names, na_values=' ?')
adult_data = pd.read_csv('adult.test', delimiter=",", header=None, names=column_names, na_values=' ?')

# Strip leading/trailing spaces and remove the period at the end of the 'income' column
adult_data['income'] = adult_data['income'].str.strip().str.replace('.', '', regex=False)

# Filter and count the number of individuals earning >50K and <=50K
people_above_50k = adult_data[adult_data['income'] == '>50K'].shape[0]
people_below_50k = adult_data[adult_data['income'] == '<=50K'].shape[0]

# Display the counts
print(f"People earning >50K: {people_above_50k}")
print(f"People earning <=50K: {people_below_50k}")
print("\n\n")


# ====================== Column Name Retrieval ======================

with open('adult.names', 'r') as file:
    names_content = file.readlines()

# Initialize a list for column names
column_names = []

# Loop through each line to find column names
for line in names_content:
    line = line.strip()  # Remove leading/trailing whitespace
    # Check if the line contains an attribute definition (not starting with '@' or '|')
    if not line.startswith('@') and line and not line.startswith('|'):
        # The attribute name is before the ':' character (if present)
        if ':' in line:
            column_names.append(line.split(':')[0].strip())
        elif line.lower() == "income":  # Check specifically for the income label
            continue  # Skip adding income in this step

# Add 'income' as the target variable
column_names.append('income')

# Print the cleaned column names
print(column_names)
print("\n\n")

# ====================== Generating HeatMap ======================

adult_data1 = pd.read_csv('adult.data', delimiter=",", header=None, names=column_names, na_values=' ?')

plt.figure(figsize=(10, 6))
sns.heatmap(adult_data1[['workclass', 'occupation', 'native-country']].isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in Workclass, Occupation, and Native-Country')
plt.show()


print(adult_data1['workclass'].value_counts())
print("\n\n")
print(adult_data1['occupation'].value_counts())
print("\n\n")
print(adult_data1['native-country'].value_counts())

# ====================== Linearity between attributes ======================


adult_data2 = pd.read_csv('adult.data', delimiter=",", header=None, names=column_names, na_values=' ?')

# Consider only relevant features for the analysis
selected_features = ['age', 'education-num', 'hours-per-week', 'capital-gain', 'capital-loss', 'income']

# Filter the dataset to include the selected features
adult_data_filtered = adult_data2[selected_features]

# Convert income to a binary variable: <=50K as 0, >50K as 1
adult_data_filtered.loc[:, 'income'] = adult_data_filtered['income'].apply(lambda x: 1 if x == ' >50K' else 0)

# Calculate the correlation matrix for selected features
correlation_matrix = adult_data_filtered.corr()

# Plot the correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Selected Features')
plt.show()

# Scatter plot to visualize the relationship between each feature and income
plt.figure(figsize=(14, 10))

# Scatter plot for Age vs Income
plt.subplot(2, 3, 1)
sns.scatterplot(x='age', y='income', data=adult_data_filtered)
plt.title('Age vs Income')

# Scatter plot for Education Number vs Income
plt.subplot(2, 3, 2)
sns.scatterplot(x='education-num', y='income', data=adult_data_filtered)
plt.title('Education Number vs Income')

# Scatter plot for Hours per Week vs Income
plt.subplot(2, 3, 3)
sns.scatterplot(x='hours-per-week', y='income', data=adult_data_filtered)
plt.title('Hours per Week vs Income')

# Scatter plot for Capital Gain vs Income
plt.subplot(2, 3, 4)
sns.scatterplot(x='capital-gain', y='income', data=adult_data_filtered)
plt.title('Capital Gain vs Income')

# Scatter plot for Capital Loss vs Income
plt.subplot(2, 3, 5)
sns.scatterplot(x='capital-loss', y='income', data=adult_data_filtered)
plt.title('Capital Loss vs Income')

plt.tight_layout()
plt.show()