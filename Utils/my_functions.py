
# my_functions.py

import pandas as pd
import numpy as np
import scipy.stats as stats

#import importlib.util

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats.mstats import winsorize
from scipy.stats import kstest, norm
from scipy.stats import boxcox
from prince import FAMD

# Models
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# matplotlit and seaborn for visualizations
import matplotlib.pyplot as plt
import seaborn as sns


def pie_plot(data, column='TARGET', title='Target Distribution', labels=None, save_path=None):
    """
    Plots a pie chart showing the distribution of target classes in the dataset.
    Displays the number of rows and columns in the dataset above the pie chart.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset containing the target column.
    column : str, optional
        The name of the target column in the dataset (default is 'TARGET').
    title : str, optional
        The title of the plot (default is 'Target Distribution').
    labels : list of str, optional
        Labels for the target classes. If None, default labels ['Not Default', 'Default'] are used.
    save_path : str, optional
        The file path to save the figure. If None, the figure is not saved (default is None).

    Returns:
    --------
    None
        Displays the pie chart with customized labels, percentage values, and a transparent background.
        Saves the plot to a file if `save_path` is provided.
    
    Notes:
    ------
    - The pie chart segments represent the proportions of each class in the target column.
    - A legend is added to the chart to clearly label each class, positioned to the left.
    - Dataset information (number of rows and columns) is displayed above the plot, centered.
    """
    if labels is None:
        labels = ['Not Default', 'Default']
    
    # Get dataset information
    n_rows, n_columns = data.shape
    data_info = f"Dataset Info:\nColumns: {n_columns}\nRows: {n_rows}"
    
    # Create a pie chart with a transparent background
    fig, ax = plt.subplots()
    fig.patch.set_alpha(0.0)  # Transparent background for the figure

    wedges, texts, autotexts = ax.pie(
        data[column].value_counts(),
        labels=labels,
        autopct='%0.02f%%',
        startangle=90
    )

    plt.title(title)

    # Customize legend
    plt.legend(wedges, labels, title=column, loc="center left", bbox_to_anchor=(1, 0.3, 1, 1))

    # Add dataset information text above the chart, centered
    plt.gcf().text(0.9, 0.3, data_info, ha='center', va='center', fontsize=10, color='gray', 
                   bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.5))

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', transparent=True)
    
    # Show the plot
    plt.show()


def convert_to_categorical(df, min_unique=2, max_unique=20):
    """
    Converts columns of type int64 with a unique value count within a specified range 
    to categorical type.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to modify.
    min_unique : int, optional
        The minimum number of unique values a column must have to be converted (default is 2).
    max_unique : int, optional
        The maximum number of unique values a column must have to be converted (default is 10).

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with specified columns converted to categorical type.
    """
    # Identify columns of type int64 with unique values between min_unique and max_unique
    cols_to_convert = [
        #col for col in df.select_dtypes(['float64','int64', 'object']).columns 
        col for col in df.select_dtypes(['object']).columns 
        if min_unique <= df[col].nunique() <= max_unique
    ]

    # Convert the selected columns to categorical type
    df[cols_to_convert] = df[cols_to_convert].astype('category')

    # Print out the columns that were converted
    print(f"Converted columns to categorical: {cols_to_convert}")
    
    return df



def encode_and_one_hot(df):
    """
    Encodes categorical columns with 2 or fewer unique values using label encoding,
    and applies one-hot encoding to the remaining categorical columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to encode.

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with label encoding applied to columns with 2 or fewer unique values
        and one-hot encoding applied to all other categorical columns.
    int
        The count of columns that were label encoded.

    Notes:
    ------
    - Columns with 2 or fewer unique values are label encoded, which replaces each unique 
      category with a numeric code.
    - Columns with more than 2 unique values are one-hot encoded, which creates binary columns 
      for each unique category.
    - This function prints the shape of the DataFrame before and after encoding.
    """

    # Display shape of the DataFrame before encoding
    print('DataFrame shape before encoding:', df.shape)
    
    # Initialize label encoder
    le = LabelEncoder()
    le_count = 0

    # Iterate through columns to apply label encoding where appropriate
    for col in df.columns:
        if df[col].dtype == 'object' and len(df[col].unique()) <= 2:
            # Apply label encoding to columns with 2 or fewer unique values
            df[col] = le.fit_transform(df[col])
            le_count += 1  # Track the count of label-encoded columns

    print(f'{le_count} columns were label encoded.')

    # Apply one-hot encoding to the remaining categorical columns
    df = pd.get_dummies(df, drop_first=True)
    
    # Display shape of the transformed DataFrame
    print('DataFrame shape after encoding:', df.shape)
    
    return df


def summarize_dataframe(df):
    """
    Summarizes the given DataFrame by displaying the total number of data types, 
    counts of unique values for float, integer, and object types, and the DataFrame's dimensions.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to be summarized.
        
    Returns:
    --------
    None
    """
    print("\nData Total Number of Each Type:\n", df.dtypes.value_counts())
    
    print("\nFloat Types Count:\n", df.select_dtypes('float64').apply(pd.Series.nunique, axis=0))
    
    print("\nInteger Types Count:\n", df.select_dtypes('int64').apply(pd.Series.nunique, axis=0))
    
    print("\nObject Types Count:\n", df.select_dtypes('object').apply(pd.Series.nunique, axis=0))
    
    print("\nData Dimension:", df.shape)


def detect_outliers_iqr_all(df, factor=3):
    """
    Detects outliers in all continuous features of the DataFrame
    using the IQR method and visualizes them with box plots.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    factor : float, optional
        The multiplier for the IQR range (default is 3 for extreme outliers).

    Returns:
    --------
    list
        A list of column names that contain outliers.
    """
    # Identify continuous features
    continuous_columns = df.select_dtypes(include=['float64', 'int64']).columns
    outlier_columns = []  # List to store names of columns with outliers

    # Set up for plotting
    num_cols = len(continuous_columns)
    fig, axes = plt.subplots(nrows=(num_cols + 2) // 3, ncols=3, figsize=(15, num_cols * 1.5))
    fig.suptitle("Outlier Detection Using Box Plots", fontsize=16, fontweight='bold')
    axes = axes.flatten()

    # Loop through each continuous feature
    for idx, column in enumerate(continuous_columns):
        # Calculate Q1, Q3, and IQR
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        # Define the outlier bounds
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        # Identify outliers
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        num_outliers = outliers.sum()

        # Print information about the outliers
        print(f"Outliers detected in '{column}': {num_outliers} rows.")
        print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}\n")

        # If there are outliers, add the column to the list
        if num_outliers > 0:
            outlier_columns.append(column)

        # Plot the box plot with outliers
        ax = axes[idx]
        ax.boxplot(df[column].dropna(), vert=False, patch_artist=True, boxprops=dict(facecolor='skyblue'))
        ax.set_title(f'{column} (Outliers: {num_outliers})')
        ax.set_xlabel(column)

        # Highlight outliers in red
        outlier_values = df.loc[outliers, column]
        ax.scatter(outlier_values, np.ones_like(outlier_values), color='red', label='Outliers', alpha=0.7)

    # Remove any empty subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return outlier_columns 


def winsorize_selected_features(df, features, lower_percentile=0.05, upper_percentile=0.95, factor=3):
    """
    Detects and caps outliers in specified continuous features using winsorization based on percentiles.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    features : list of str
        List of column names to apply winsorization on.
    lower_percentile : float, optional
        Lower percentile for winsorization (default is 0.05, or 5%).
    upper_percentile : float, optional
        Upper percentile for winsorization (default is 0.95, or 95%).
    factor : float, optional
        Multiplier for IQR range to identify extreme outliers (default is 3).
        
    Returns:
    --------
    pandas.DataFrame
        The DataFrame with winsorized values for specified features.
    """
    # Copy the DataFrame to avoid modifying the original during processing
    df_transformed = df.copy()

    # Loop through each specified feature to apply winsorization
    for column in features:
        if column in df.columns:
            # Calculate Q1, Q3, and IQR
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1

            # Define the outlier bounds
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR

            # Identify outliers for information purposes
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
            num_outliers = outliers.sum()
            print(f"Outliers detected in '{column}': {num_outliers} rows.")
            print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}\n")

            # Apply winsorization and update the column in the copied DataFrame
            df_transformed[column] = winsorize(df[column], limits=(lower_percentile, 1 - upper_percentile))

    print("Winsorization applied. Extreme values have been capped at lower bound 5% or upper bound 95%.")
    return df_transformed


def plot_histogram(data, column, bins=30, title=None, xlabel=None, ylabel='Frequency', color='skyblue', edge_color='black'):
    """
    Plots a histogram with a transparent background, custom color, and labels.

    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset containing the column to plot.
    
    Returns:
    --------
    None
        Displays the histogram with customized styling and labels.
    """
    # Set up figure and transparent background
    fig, ax = plt.subplots()
    fig.patch.set_alpha(0.0)  # Transparent background for the figure

    # Plot histogram
    ax.hist(data[column], bins=bins, color=color, edgecolor=edge_color)

    # Set title and labels
    ax.set_title(f'Histogram of {title}' if title  else f'Histogram of {column}', fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel if xlabel else column, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # Style grid and layout
    ax.grid(visible=True, color='grey', linestyle='--', linewidth=0.5)
    plt.tight_layout()  # Adjust layout for better fit

    # Show plot
    plt.show()


def plot_selected_histograms(data, features, bins=30, color='skyblue', edge_color='black'):
    """
    Plots histograms for specified continuous (numeric) features in the dataset with a transparent background,
    custom color, and labels. Adds a normal density bell curve overlay for each histogram.

    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset containing the columns to plot.
    features : list of str
        List of feature names to plot histograms for.
    bins : int, optional
        Number of bins for the histograms (default is 30).
    color : str, optional
        Color for the bars (default is 'skyblue').
    edge_color : str, optional
        Color for the edges of the bars (default is 'black').

    Returns:
    --------
    None
        Displays histograms for each specified continuous feature in a grid layout with density overlays.
    """
    # Filter to only include specified features that exist in the DataFrame
    continuous_columns = [col for col in features if col in data.columns and 
                          pd.api.types.is_numeric_dtype(data[col])]
    
    if not continuous_columns:
        print("No valid continuous features found in the provided list.")
        return

    # Determine grid size for subplots
    num_columns = len(continuous_columns)
    grid_size = int(num_columns**0.5) + 1  # For a balanced grid layout

    # Set up figure with transparent background
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 10))
    fig.patch.set_alpha(0.0)  # Transparent background for the figure
    axes = axes.flatten()  # Flatten the grid for easy indexing

    # Loop through each numeric column and create a histogram
    for i, column in enumerate(continuous_columns):
        ax = axes[i]
        
        # Create histogram
        ax.hist(data[column].dropna(), bins=bins, color=color, edgecolor=edge_color, alpha=0.6, density=True)
        
        # Calculate mean and standard deviation
        mean = data[column].mean()
        std_dev = data[column].std()
        
        # Generate x values for the bell curve
        x = np.linspace(data[column].min(), data[column].max(), 100)
        # Calculate the normal distribution
        p = np.exp(-0.5 * ((x - mean) / std_dev) ** 2) / (std_dev * np.sqrt(2 * np.pi))
        
        # Plot the normal density curve
        ax.plot(x, p, color='red', lw=2, label='Normal Density', alpha=0.8)
        
        # Add titles and labels
        ax.set_title(f'Histogram of {column}', fontsize=12, fontweight='bold')
        ax.set_xlabel(column, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.grid(visible=True, color='grey', linestyle='--', linewidth=0.5)
        ax.legend()
    
    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout for better fit
    plt.tight_layout()
    plt.show()


def transform_to_log(df, features):
    """
    Transforms specified continuous features in a DataFrame to their logarithmic values,
    adds these transformed features back to the DataFrame, and removes the original features.

    Parameters:
    -----------
    df : pandas.DataFrame
        The original DataFrame containing the features to be transformed.
    features : list of str
        A list of column names in the DataFrame to be transformed to log scale.

    Returns:
    --------
    pandas.DataFrame
        The modified DataFrame with original features removed and log-transformed features added.

    Raises:
    -------
    ValueError
        If any of the specified features are not found in the DataFrame or contain non-positive values.
    """
    # Check for missing features
    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Features not found in DataFrame: {missing_features}")

    # Create a copy of the DataFrame to avoid modifying the original one
    df_transformed = df.copy()

    for feature in features:
        # Check for non-positive values
        if (df_transformed[feature] <= 0).any():
            raise ValueError(f"Feature '{feature}' contains non-positive values, cannot apply log transformation.")

        # Create a new feature with the log transformation
        log_feature_name = f"log_{feature}"
        df_transformed[log_feature_name] = np.log(df_transformed[feature])

        # Optionally, you can remove the original feature
        df_transformed.drop(columns=[feature], inplace=True)

    return df_transformed


def plot_qq_hist(data, columns):
    """
    Plots histograms with normal density curves and Q-Q plots for each selected column.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The DataFrame containing the data.
    columns : list
        A list of column names to plot.

    Returns:
    --------
    None
        Displays histograms and Q-Q plots for each specified column.
    """
    for column in columns:
        plt.figure(figsize=(8, 4))

        # Histogram with density plot
        plt.subplot(1, 2, 1)
        plt.hist(data[column].dropna(), bins=30, density=True, alpha=0.6, color='skyblue')
        min_val, max_val = data[column].min(), data[column].max()
        x = np.linspace(min_val, max_val, 100)
        plt.plot(x, stats.norm.pdf(x, data[column].mean(), data[column].std()), 'r', lw=2)
        plt.title(f'Histogram and Normal Curve of {column}', fontsize=10)

        # Q-Q plot
        plt.subplot(1, 2, 2)
        stats.probplot(data[column].dropna(), dist="norm", plot=plt)
        plt.title(f'Q-Q Plot of {column}', fontsize=10)

        plt.tight_layout()
        plt.show()
        


def kolmogorov_smirnov_test(data, columns, alpha=0.05):
    """
    Performs the Kolmogorov-Smirnov test for normality on a list of selected columns.
    Prints a table of p-values and normality results for each column.

    Parameters:
    -----------
    data : pandas.DataFrame
        The DataFrame containing the data.
    columns : list
        A list of column names to test for normality.
    alpha : float, optional
        The significance level to determine normality (default is 0.05).

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with columns indicating the p-values and normality results for each column.
    """
    results = []

    # Perform K-S test for each column
    for column in columns:
        # Drop NA values in case of missing data
        data_column = data[column].dropna()
        
        # Perform K-S test against the normal distribution
        stat, p_value = kstest(data_column, 'norm', args=(data_column.mean(), data_column.std()))
        
        # Determine normality based on the p-value
        normality = 'Normal' if p_value > alpha else 'Non-Normal'
        
        # Append results to the list
        results.append({'Column': column, 'K-S p-value': p_value, 'Normality': normality})
    
    # Convert results to a DataFrame for a nice table format
    results_df = pd.DataFrame(results)
    
    # Display the table
    print("\nKolmogorov-Smirnov Test for Normality:")
    print(results_df)

    return results_df


def box_cox_transform(data, columns, add_constant=1e-6):
    """
    Apply Box-Cox transformation to selected columns in the DataFrame, handling non-positive
    values and imputing missing values with the column median.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The DataFrame containing the data.
    columns : list
        List of columns to apply the Box-Cox transformation to.
    add_constant : float, optional
        Small positive constant to add to ensure all values are positive (default is 1e-6).
    
    Returns:
    --------
    pandas.DataFrame
        A DataFrame with original features retained and Box-Cox transformed features added.
    """
    # Copy data to avoid modifying the original DataFrame directly
    transformed_data = data.copy()

    for column in columns:
        # Impute missing values with the median of the column
        median_val = transformed_data[column].median()
        transformed_data[column].fillna(median_val, inplace=True)

        # Ensure positivity by adding a constant if necessary
        if (transformed_data[column] + add_constant).min() <= 0:
            print(f"Skipping {column}: Contains non-positive values even after adding constant.")
            continue
        try:
            # Apply Box-Cox transformation
            transformed, _ = boxcox(transformed_data[column] + add_constant)
            # Replace the original column with transformed values
            transformed_data[column] = transformed
            print(f"Box-Cox transformation applied to {column}.")
        except ValueError as e:
            print(f"Could not apply Box-Cox to {column}: {e}")
    
    return transformed_data


def apply_robust_scaling(data: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Applies robust scaling to specified features of the given DataFrame and 
    replaces the original features with their scaled versions.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input DataFrame containing the data.
    features : list
        List of column names to apply robust scaling.

    Returns:
    --------
    pandas.DataFrame
        A new DataFrame with the scaled features replacing the original ones.
    """
    # Initialize the RobustScaler
    scaler = RobustScaler()
    
    # Fit and transform the specified features
    scaled_features = scaler.fit_transform(data[features])
    
    # Create a DataFrame from the scaled features with the same index
    scaled_df = pd.DataFrame(scaled_features, columns=features, index=data.index)
    
    # Replace the original features in the original DataFrame with the scaled features
    data[features] = scaled_df
    
    # Return the modified DataFrame with scaled features
    return data


def get_top_correlations(df, target, top_n=15):
    """
    Calculates and returns the top positive and negative correlations of DataFrame features with the target.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the features.
    target : pandas.Series
        The target variable for correlation calculations.
    top_n : int, optional (default=15)
        The number of top positive and negative correlations to display.

    Returns:
    --------
    list
        Combined list of column names with the top positive and negative correlations.
    """
    # Calculate correlations
    correlations = df.apply(lambda x: x.corr(target))

    # Get positive and negative correlations
    positive_correlations_sorted = correlations[correlations > 0].sort_values(ascending=False)
    negative_correlations_sorted = correlations[correlations < 0].sort_values()

    # Get top n column names
    top_n_column_names = positive_correlations_sorted.head(top_n).index.tolist()
    bottom_n_column_names = negative_correlations_sorted.head(top_n).index.tolist()

    # Combine both lists
    combined_column_names = top_n_column_names + bottom_n_column_names

    # Display the most positive correlations
    print(f'Most Positive Correlations (Top {top_n}):\n\n', positive_correlations_sorted.head(top_n))

    # Display the most negative correlations
    print(f'\n\nMost Negative Correlations (Top {top_n}):\n\n', negative_correlations_sorted.head(top_n))

    return combined_column_names


def preprocess_data(df):
    """
    Detects outliers, applies winsorization, and robust scaling for specified columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame to be processed.

    Returns:
    --------
    pandas.DataFrame
        The final processed DataFrame with outliers handled and scaled.
    """
    # Step 1: Detect outliers in the DataFrame using IQR method
    outlier_columns = detect_outliers_iqr_all(df)
    
    # Step 2: Ensure only numeric columns are processed
    numeric_outlier_columns = [col for col in outlier_columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if not numeric_outlier_columns:
        print("No numeric columns with outliers were found.")
        return df  # Return the original DataFrame if no valid columns

    #print("Start to winsorize the detected numeric outlier columns")
    #print("-------------------------------------------------------")
    # Step 3: Winsorize the detected numeric outlier columns
    #df2 = winsorize_selected_features(df, numeric_outlier_columns)
    #print("\nEnd to winsorize the detected numeric outlier columns")
    #print("-------------------------------------------------------")

    #print("\n\nRecheck the outlier columns\n")
    #print("-------------------------------------------------------")
    # Step 4: Apply robust scaling to the numeric outlier columns
    df2 = apply_robust_scaling(df, numeric_outlier_columns)
    
    # Recheck detect outliers again after scaling
    #detect_outliers_iqr_all(df3)
    
    return df2


def missing_values_table(df):
    """
    Calculates and displays a table of missing values in the DataFrame.

    This function identifies missing values in each column of the DataFrame,
    calculates the percentage of missing values, and presents the results
    in a structured format.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame for which to calculate missing values.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the number and percentage of missing values 
        for each column with missing data.

    Notes:
    ------
    - The output DataFrame is sorted by the percentage of missing values in 
      descending order.
    - Only columns with missing values are included in the output.
    """

    # Total missing values
    mis_val = df.isnull().sum()
    
    # Percentage of missing values
    mis_val_percent = 100 * mis_val / len(df)
    
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    # Rename the columns
    mis_val_table.columns = ['Missing Values', '% of Total Values']
    
    # Sort the table by percentage of missing values in descending order
    mis_val_table = mis_val_table[mis_val_table['% of Total Values'] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    
    # Print summary information
    print(f"Number of columns of this dataframe is {df.shape[1]}.\n"
          f"Columns with missing values are {mis_val_table.shape[0]}.")
    
    # Return the DataFrame with missing information
    return mis_val_table


def preprocess_and_reduce_features(data, n_components=20, target_variance=0.7):
    """
    Preprocesses a dataset by handling missing values, encoding categorical variables, and reduces dimensionality 
    using Factor Analysis of Mixed Data (FAMD) until the desired explained variance is achieved.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input DataFrame containing mixed types of data, including both continuous and categorical features.
    n_components : int, optional (default=20)
        Initial number of FAMD components to start with for dimensionality reduction. The function dynamically 
        increases this number until the target variance is met or a third of the feature count is reached.
    target_variance : float, optional (default=0.7)
        Target cumulative explained variance (between 0 and 1) required to determine the final number of FAMD components.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with the transformed data, reduced to a subset of components as determined by FAMD.
    """
    # Step 1: Separate continuous and categorical features
    continuous_features = data.select_dtypes(include=['float64', 'int64'])
    categorical_features = data.select_dtypes(include=['object', 'category', 'bool', 'boolean'])
    
    # Step 2: Convert categorical features to string type
    categorical_features = categorical_features.astype(str)
    
    # Step 3: Impute missing values
    # Impute continuous features with mean
    continuous_imputer = SimpleImputer(strategy='mean')
    continuous_imputed = pd.DataFrame(continuous_imputer.fit_transform(continuous_features),
                                      columns=continuous_features.columns, index=data.index)
    
    # Impute categorical features with the most frequent value
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    categorical_imputed = pd.DataFrame(categorical_imputer.fit_transform(categorical_features),
                                       columns=categorical_features.columns, index=data.index)
    
    # Step 4: Combine the imputed continuous and categorical data
    data_imputed = pd.concat([continuous_imputed, categorical_imputed], axis=1)
    
    # Step 5: Initialize cumulative variance
    cumulative_variance = 0
   
    # Increment n_components until cumulative variance exceeds target_variance
    while cumulative_variance < target_variance:
        famd = FAMD(n_components=n_components, n_iter=3, copy=True, check_input=True, random_state=45, engine="sklearn")
        famd = famd.fit(data_imputed)
        
        # Calculate cumulative variance
        explained_variance = famd.percentage_of_variance_
        cumulative_variance = np.sum(explained_variance[:n_components]) / 100

         # Print explained variance by FAMD components
        #print(famd.eigenvalues_summary)
        #print("Cumulative Variance:", cumulative_variance)
        
        if cumulative_variance < target_variance:
            n_components += 10  # Increase components by 10 if target variance not reached
    
    #print(f"Number of components to reach {target_variance*100}% variance: {n_components}")
    data_reduced = pd.DataFrame(famd.fit_transform(data_imputed), index=data.index)
    
    # Print explained variance by FAMD components
    print(famd.eigenvalues_summary)
    print("Cumulative Variance:", cumulative_variance)
    print("\n\nReduced Data Shape:", data_reduced.shape)
    print("\nReduced Data (first 5 rows):\n", data_reduced.head())
    
    return data_reduced


def merge_data_on_index(data1, data2, data1_name="First data input", data2_name="Second data input", how='inner'):
    """
    Merges two DataFrames on their indexes.

    Parameters:
    -----------
    data1 : pandas.DataFrame
        The first DataFrame to merge.
    data2 : pandas.DataFrame
        The second DataFrame to merge.
    how : str, optional (default='inner')
        Type of merge to be performed (e.g., 'inner', 'outer', 'left', 'right').

    Returns:
    --------
    combined_data : pandas.DataFrame
        The merged DataFrame.
    """
    # Display shapes of input data
    print(f'{data1_name} shape: {data1.shape}')
    print(f'{data2_name} shape: {data2.shape}')
    
    # Merge on index
    combined_data = pd.merge(data1, data2, left_index=True, right_index=True, how=how)
    
    # Display shape of merged data
    print("Combined data shape:", combined_data.shape)
    
    return combined_data


def preprocess_data_for_split(data, target, impute_strategy='median', scale_range=(0, 1)):
    """
    Preprocesses the data by ensuring column names are strings, imputing missing values, and scaling features.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input DataFrame containing features.
    target : pandas.Series or numpy array
        The target variable.
    impute_strategy : str, optional (default='median')
        The strategy for imputing missing values (e.g., 'mean', 'median', 'most_frequent').
    scale_range : tuple, optional (default=(0, 1))
        The range for MinMax scaling.

    Returns:
    --------
    X : numpy array
        The preprocessed feature data, ready for model training.
    y : numpy array
        The target variable data.
    """
    # Ensure all column names in data are strings
    data.columns = data.columns.astype(str)

    # Initialize the imputer and scaler
    imputer = SimpleImputer(strategy=impute_strategy)
    scaler = MinMaxScaler(feature_range=scale_range)

    # Impute and scale the data
    data_imputed = imputer.fit_transform(data)
    data_scaled = scaler.fit_transform(data_imputed)

    # Prepare the final X and y
    X = data_scaled
    y = target.values if hasattr(target, 'values') else target  # Ensure y is an array

    print(f'Impute strategy for missing values uses {impute_strategy}.')
    print(f'Scale ranges of Data is between {scale_range[0]} and {scale_range[1]}.')
    print(f'Transfoming methods are applied to X data only.')
    print('X data shape:', X.shape)
    print('y data shape:', y.shape)
    
    return X, y


def evaluate_models3_with_resampling(X, y, test_size=0.3, random_state=45):
    """
    Splits data, initializes classifiers with different configurations, and evaluates each model.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Feature data.
    y : pandas.Series or numpy.ndarray
        Target data.
    test_size : float, optional (default=0.3)
        Proportion of data to include in the test split.
    random_state : int, optional (default=45)
        Random seed for reproducibility.

    Returns:
    --------
    None
    """

    # Ensure y is a pandas Series with 1-dimensional before creating a Series
    y = pd.Series(y.squeeze())
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Calculate class weights and scale_pos_weight based on imbalance ratio
    class_weights = [1, (y_train.value_counts()[0] / y_train.value_counts()[1])]
    scale_pos_weight = class_weights[1]

    # Initialize SMOTE for later use
    smote = SMOTE(random_state=random_state)

    # Define models with their specific configurations
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, 
            random_state=random_state),
        
        "LightGBM with SMOTE": LGBMClassifier(
            objective='binary', 
            learning_rate=0.05,
            n_estimators=500, 
            max_depth=7, 
            random_state=random_state),
        
        "LightGBM with Class Weighting": LGBMClassifier(
            objective='binary', learning_rate=0.05,
            n_estimators=500, max_depth=7, 
            scale_pos_weight=scale_pos_weight, 
            random_state=random_state),
        
        "CatBoost with Class Weights": CatBoostClassifier(
            iterations=500, 
            learning_rate=0.05, 
            depth=7, 
            class_weights=class_weights, 
            random_seed=random_state, 
            verbose=0)
    }

    # Initialize a dataframe to store the results
    results = []

    # Evaluate each model
    for model_name, model in models.items():
        X_train_res, y_train_res = X_train, y_train
        
        # Apply SMOTE only for the specified LightGBM model
        if model_name == "LightGBM with SMOTE":
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        # Fit the model
        model.fit(X_train_res, y_train_res)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        # Store the results
        results.append({
            "Model": model_name,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "AUC-ROC": auc_roc,
            "Alert Rate": (y_pred == 1).mean()
        })

    # Convert results to a pandas dataframe
    results_df = pd.DataFrame(results)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    # Print results
    print(results_df)




