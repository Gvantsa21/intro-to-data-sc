# from faker import Faker
# import pandas as pd
# import numpy as np
# import random

# fake = Faker()

# def generate_employee_data(n_rows):
#     """Generate realistic employee data"""
#     data = {
#         "Employee_ID": range(1, n_rows + 1),
#         "Name": [fake.name() for _ in range(n_rows)],
#         "Age": [random.randint(22, 65) for _ in range(n_rows)],
#         "Salary": [round(random.uniform(35000, 150000), 2) for _ in range(n_rows)],
#         "City": [fake.city() for _ in range(n_rows)],
#         "Joining_Date": [fake.date_between(start_date='-10y', end_date='today') for _ in range(n_rows)],
#         "Email": [fake.email() for _ in range(n_rows)],
#         "Phone": [fake.phone_number() for _ in range(n_rows)],
#         "Department": [random.choice(['Sales', 'IT', 'Marketing', 'HR', 'Finance']) for _ in range(n_rows)],
#         "Job_Title": [fake.job() for _ in range(n_rows)]
#     }
#     return pd.DataFrame(data)

# # Generate 100 employees
# df = generate_employee_data(100)

# # Save to CSV
# df.to_csv('employees_data.csv', index=False)

# # Task 1
# data = pd.read_csv('employees_data.csv')
# df = pd.DataFrame(data)

# filtered_df = df[df["Age"].between(25, 40) & (df["Salary"] > 60000) & (df["Department"].isin(["Sales", "Marketing"]))]
# print(filtered_df)

# # Task 2
# print("=" * 80)
# print("TASK 2")
# print("=" * 80)

# data = pd.read_csv('employees_data.csv')
# df = pd.DataFrame(data)

# result1 = df.loc[2:4, ['Name', 'Age', 'Salary']]
# print("Task1: Using loc:")
# print(result1)


# result2 = df.iloc[0:3, 0:2]
# print("Task 2 - Using iloc:")
# print(result2)


# result3 = df.loc[df['Age'] > 30, ['Name', 'Department']]
# print("Task 3 - Conditional with loc:")
# print(result3)

# # task3
# data = pd.read_csv('employees_data.csv')
# df = pd.DataFrame(data)

# # 1. Identify missing values
# print("Missing values per column:")
# print(df.isnull().sum())
# print()

# # 2. Fill Salary with mean
# df['Salary'].fillna(df['Salary'].mean(), inplace=True)

# # 3. Fill Department with 'Unknown'
# df['Department'].fillna('Unknown', inplace=True)

# # 4. Drop rows where Age is missing
# df_dropped = df.dropna(subset=['Age'])

# print("After filling and dropping:")
# print(df_dropped)
# print()

# # 5. Complete cases only
# complete_cases = df.dropna()
# print("Complete cases only:")
# print(complete_cases)



# Task 4: Student Data TransformationProblem: Generate student data and:

# Create a GPA category (Excellent: 3.5-4.0, Good: 3.0-3.49, Average: 2.5-2.99, Below Average: <2.5)
#  Calculate age from date of birth Normalize GPA scores using transform() Find students who are eligible 
# for Dean's List (GPA >= 3.7 and Credits >= 60) Group by major and calculate statistics

from faker import Faker
import pandas as pd
import numpy as np
import random
from datetime import datetime

fake = Faker()
Faker.seed(42)
random.seed(42)

def generate_student_data(n_students):
    majors = ['Computer Science', 'Business', 'Engineering', 'Psychology',
              'Biology', 'Mathematics', 'English', 'History']

    data = {
        "Student_ID": [f"STU{str(i).zfill(6)}" for i in range(1, n_students + 1)],
        "Name": [fake.name() for _ in range(n_students)],
        "Email": [fake.email() for _ in range(n_students)],
        "Date_of_Birth": [fake.date_of_birth(minimum_age=18, maximum_age=25) for _ in range(n_students)],
        "Enrollment_Date": [fake.date_between(start_date='-4y', end_date='today') for _ in range(n_students)],
        "Major": [random.choice(majors) for _ in range(n_students)],
        "GPA": [round(random.uniform(2.0, 4.0), 2) for _ in range(n_students)],
        "Credits_Completed": [random.randint(0, 150) for _ in range(n_students)],
        "Scholarship": [random.choice([True, False]) for _ in range(n_students)]
    }

    return pd.DataFrame(data)

df = generate_student_data(100)

print("=" * 80)
print("TASK 4: STUDENT DATA TRANSFORMATION")
print("=" * 80)

# 1. Calculate Age from Date_of_Birth
today = pd.to_datetime('today')
df['Date_of_Birth'] = pd.to_datetime(df['Date_of_Birth'])
df['Age'] = (today - df['Date_of_Birth']).dt.days // 365

# 2. Create GPA Category
def gpa_category(gpa):
    if gpa >= 3.5:
        return 'Excellent'
    elif gpa >= 3.0:
        return 'Good'
    elif gpa >= 2.5:
        return 'Average'
    else:
        return 'Below Average'

df['GPA_Category'] = df['GPA'].apply(gpa_category)

# 3. Normalize GPA using transform (min-max normalization)
gpa_min = df['GPA'].min()
gpa_max = df['GPA'].max()
df['GPA_Normalized'] = df['GPA'].transform(lambda x: (x - gpa_min) / (gpa_max - gpa_min))

# 4. Find students eligible for Dean's List (GPA >= 3.7 and Credits >= 60)
deans_list = df[(df['GPA'] >= 3.7) & (df['Credits_Completed'] >= 60)]
print("Students eligible for Dean's List:")
print(deans_list[['Student_ID', 'Name', 'GPA', 'Credits_Completed', 'Major']])

# 5. Group by Major and calculate statistics
major_stats = df.groupby('Major').agg(
    Num_Students=('Student_ID', 'count'),
    Avg_GPA=('GPA', 'mean'),
    Max_GPA=('GPA', 'max'),
    Min_GPA=('GPA', 'min'),
    Avg_Credits=('Credits_Completed', 'mean')
).reset_index()

print("\nStatistics by Major:")
print(major_stats)

# Optional: Show first few rows of transformed student dataframe
print("\nTransformed Student Data (first 5 rows):")
print(df.head())



# import pandas as pd
# import numpy as np
# from faker import Faker
# import random

# fake = Faker()
# Faker.seed(42)
# random.seed(42)

# # --- Generate student data ---
# def generate_student_data(n_students):
#     majors = ['Computer Science', 'Business', 'Engineering', 'Psychology',
#               'Biology', 'Mathematics', 'English', 'History']

#     data = {
#         "Student_ID": [f"STU{str(i).zfill(6)}" for i in range(1, n_students + 1)],
#         "Name": [fake.name() for _ in range(n_students)],
#         "Email": [fake.email() for _ in range(n_students)],
#         "Date_of_Birth": [fake.date_of_birth(minimum_age=18, maximum_age=25) for _ in range(n_students)],
#         "Enrollment_Date": [fake.date_between(start_date='-4y', end_date='today') for _ in range(n_students)],
#         "Major": [random.choice(majors) for _ in range(n_students)],
#         "GPA": [round(random.uniform(2.0, 4.0), 2) for _ in range(n_students)],
#         "Credits_Completed": [random.randint(0, 150) for _ in range(n_students)],
#         "Scholarship": [random.choice([True, False]) for _ in range(n_students)]
#     }

#     return pd.DataFrame(data)

# df = generate_student_data(100)

# # --- Task 4 Transformations ---

# # 1. Convert DOB to datetime and compute Age
# df['Date_of_Birth'] = pd.to_datetime(df['Date_of_Birth'])
# reference_date = pd.Timestamp(2025, 10, 23)

# def compute_age(dob, ref=reference_date):
#     years = ref.year - dob.year - ((ref.month, ref.day) < (dob.month, dob.day))
#     return years

# df['Age'] = df['Date_of_Birth'].apply(lambda d: compute_age(d))

# # 2. Create GPA categories
# bins = [-np.inf, 2.5, 3.0, 3.5, np.inf]
# labels = ['Below Average', 'Average', 'Good', 'Excellent']
# df['GPA_Category'] = pd.cut(df['GPA'], bins=bins, labels=labels, right=False)

# # 3. Normalize GPA (min-max)
# df['GPA_Normalized'] = (df['GPA'] - df['GPA'].min()) / (df['GPA'].max() - df['GPA'].min())

# # 4. Identify Dean's List students (GPA >= 3.7 & Credits >= 60)
# df['Deans_List'] = (df['GPA'] >= 3.7) & (df['Credits_Completed'] >= 60)
# deans_df = df[df['Deans_List']].copy()

# # 5. Group by Major and calculate statistics
# grouped_stats = df.groupby('Major').agg(
#     count_students = ('Student_ID', 'count'),
#     mean_GPA = ('GPA', 'mean'),
#     median_GPA = ('GPA', 'median'),
#     std_GPA = ('GPA', 'std'),
#     mean_Credits = ('Credits_Completed', 'mean'),
#     deans_count = ('Deans_List', 'sum')
# ).reset_index()

# # 6. Optional: Drop rows with any missing values
# clean_df = df.dropna()

# # --- Quick Checks / Prints ---
# print("Sample transformed rows:")
# print(df.head())

# print("\nDean's List (first 10 rows):")
# print(deans_df.head(10))

# print("\nGrouped statistics by Major:")
# print(grouped_stats)
