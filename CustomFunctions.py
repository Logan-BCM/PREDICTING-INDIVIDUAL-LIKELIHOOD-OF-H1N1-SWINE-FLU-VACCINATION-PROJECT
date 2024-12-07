import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to calculate null percentages
def missing_percentage(df):
    a = df.shape
    nan_percent = (df.isna().sum() / a[0])*100 # total percent of missing values per column
    return nan_percent


"DATA ANALYSIS FUNCTIONS FOR UNIVARIATE RELATIONSHIPS"
# Function to display value counts for categorical columns
def analyze_categorical(data, columns):
    for col in columns:
        print(f"\nValue counts for {col}:\n")
        print(data[col].value_counts())

# Function to plot bar charts for categorical features
def plot_categorical(data, columns):
    for col in columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=data, x=col, palette='pastel', order=data[col].value_counts().index)
        plt.title(f"Distribution of {col}")
        plt.xticks(rotation=45)
        plt.show()

# Function to display summary statistics for numerical columns
def analyze_numerical(data, columns):
    print(data[columns].describe())

# Function to plot histograms for numerical features
def plot_histograms(data, columns):
    for col in columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(data=data, x=col, kde=True, color='skyblue', bins=20)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.show()

# Function to plot box plots for numerical features
def plot_boxplots(data, columns):
    fig, axes = plt.subplots(nrows=8,figsize=(15, 15))
    axes = axes.flatten()
    for i, col in enumerate(columns):
        if i < len(axes):  # Avoid exceeding the number of available subplots
            sns.boxplot(data=data, x=col, palette='pastel', ax=axes[i])
            axes[i].set_title(f"Boxplot of {col}")
            axes[i].set_xlabel(col)
        else:
            break 
    plt.tight_layout()
    plt.show()

# Function to analyze binary features
def analyze_binary(data, columns):
    for col in columns:
        print(f"\nProportions for {col}:\n")
        print(data[col].value_counts(normalize=True) * 100)

# Function to plot binary proportions
def plot_binary(data, columns):
    for col in columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=data, x=col, palette='pastel')
        plt.title(f"Proportion of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        ax = plt.gca()
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["NO", "YES"])
        plt.show()

# Function to analyze opinion features
def analyze_opinion(data, columns):
    for col in columns:
        print(f"\nValue counts for {col}:\n")
        print(data[col].value_counts())

# Function to plot opinion features as bar plots
def plot_opinion(data, columns, opinion_labels):
    for col in columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(
            data=data, 
            x=col, 
            palette='coolwarm', 
            order=sorted(data[col].unique())
        )
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(
            ticks=range(len(opinion_labels[col])),
            labels=[opinion_labels[col].get(val, val) for val in sorted(data[col].unique())],
            rotation=45)
        plt.show()


"DATA ANALYSIS FUNCTIONS FOR BIVARIATE RELATIONSHIPS"
def analyze_target_vs_categorical(data, target, categorical_columns):
    for col in categorical_columns:
        print(f"\nRelationship between {target} and {col}:\n")
        print(pd.crosstab(data[col], data[target], normalize='index'))

# Function to visualize target vs categorical features
def plot_target_vs_categorical(data, target, categorical_columns):
    for col in categorical_columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=data, x=col, hue=target, palette='coolwarm', order=data[col].value_counts().index)
        plt.title(f"{target} vs {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.legend(title=target, labels = ["Not Vaccinated", "Vaccinated"])
        plt.xticks(rotation=45)
        plt.show()

# Function to analyze target vs numerical features
def analyze_target_vs_numerical(data, target, numerical_columns):
    for col in numerical_columns:
        print(f"\nSummary statistics for {col} by {target}:\n")
        print(data.groupby(target)[col].describe())

# Function to calculate and visualize a correlation matrix
def plot_correlation_matrix(data, list_columns):
    corr_matrix = data[list_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title("Correlation Matrix")
    plt.show()

#Analyzes the interaction between two features and their relationship with the target variable.
def analyze_feature_interaction(data, feature1, feature2, target):
    interaction = pd.crosstab(index=data[feature1], columns=data[feature2], values=data[target], aggfunc='mean')
    print(f"\nMean {target} for interaction between {feature1} and {feature2}:\n")
    print(interaction)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.heatmap(interaction, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title(f"Interaction Between {feature1} and {feature2} on {target}")
    plt.xlabel(feature2)
    plt.ylabel(feature1)
    plt.show()


"DATA ANALYSIS FUNCTIONS FOR MUlTIVARIATE RELATIONSHIPS"
def plot_behavioral_vs_vaccine(data, target, behavioral_features):
    for col in behavioral_features:
        plt.figure(figsize=(12, 6))
        sns.countplot(data=data, x=col, hue=target, palette='pastel')
        plt.title(f"{col} vs Vaccine Status")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.legend(title=target, labels=["Not Vaccinated","Vaccinated"])
        plt.xticks(rotation=45)
        ax = plt.gca()
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["NO", "YES"])
        plt.show()

# Function to plot box plots for opinion features against vaccine uptake
def plot_opinion_vs_vaccine(data, target, opinion_feature, opinion_labels):
    for col in opinion_feature:
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=col, y=target, data=data, palette='Set2', marker = "o")
        plt.title(f"Trend of {col} by {target}")
        plt.xlabel(col)
        plt.ylabel(target)
        plt.xticks(
            ticks=sorted(data[col].dropna().unique()),
            labels=[opinion_labels[col].get(val, val) for val in sorted(data[col].unique())],
            rotation=45)
        plt.show()

# Function to create a stacked bar plot to compare vaccine uptake across different demographic categories
def plot_stacked_bar(data, target, demographic_feature):
    for col in demographic_feature:
        vaccine_counts = data.groupby([col, target]).size().unstack().fillna(0)
        vaccine_counts.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')
        plt.title(f"Vaccine Uptake by {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.legend(title=target, labels=["Not Vaccinated", "Vaccinated"])
        plt.xticks(rotation=45)
        plt.show()


# Function to plot bar plots for health features and vaccine uptake
def plot_health_feature_vs_vaccine(data, target, health_feature):
    for col in health_feature:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=data, x=col, y=target, ci=None, palette='Set1')
        plt.title(f"Comparison of {col} with {target}")
        plt.xlabel(col)
        plt.ylabel("Proportion of Vaccinated")
        plt.xticks(rotation=45)
        ax = plt.gca()
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Not Vaccinated", "Vaccinated"])
        plt.show()

