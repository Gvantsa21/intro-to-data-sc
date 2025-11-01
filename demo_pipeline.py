import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from faker import Faker
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Initialize Faker
fake = Faker()
Faker.seed(42)
np.random.seed(42)

print("=" * 80)
print("MATPLOTLIB PRACTICAL WORKSHOP")
print("Demonstrating: Faker → NumPy → Pandas → Matplotlib Pipeline")
print("=" * 80)

# ============================================================================
# EXAMPLE 1: E-COMMERCE SALES ANALYSIS
# ============================================================================
print("\n📊 EXAMPLE 1: E-Commerce Sales Analysis")
print("-" * 80)

def generate_ecommerce_data(n_records=500):
    """Generate realistic e-commerce sales data"""

    # Generate data using Faker
    data = {
        'order_id': [fake.uuid4() for _ in range(n_records)],
        'customer_name': [fake.name() for _ in range(n_records)],
        'product': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Headphones', 'Watch'], n_records),
        'quantity': np.random.randint(1, 5, n_records),
        'price': np.random.uniform(50, 2000, n_records),
        'date': [fake.date_between(start_date='-1y', end_date='today') for _ in range(n_records)],
        'country': [fake.country() for _ in range(n_records)],
        'rating': np.random.choice([1, 2, 3, 4, 5], n_records, p=[0.05, 0.1, 0.15, 0.35, 0.35])
    }

    # Create DataFrame (Pandas)
    df = pd.DataFrame(data)

    # Calculate total sales using NumPy operations
    df['total_sales'] = df['quantity'] * df['price']

    # Add month column
    df['month'] = pd.to_datetime(df['date']).dt.month_name()

    return df

# Generate data
ecommerce_df = generate_ecommerce_data(500)
print("\n📋 Data Sample:")
print(ecommerce_df.head())
print(f"\n📈 Dataset Info: {len(ecommerce_df)} records")

# Visualization 1.1: Sales by Product (Bar Chart) - SEPARATE PLOT
plt.figure(figsize=(12, 6))
product_sales = ecommerce_df.groupby('product')['total_sales'].sum().sort_values(ascending=False)
colors = plt.cm.viridis(np.linspace(0, 1, len(product_sales)))
plt.bar(product_sales.index, product_sales.values, color=colors)
plt.title('Total Sales by Product', fontsize=14, fontweight='bold')
plt.xlabel('Product')
plt.ylabel('Total Sales ($)')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(product_sales.values):
    plt.text(i, v, f'${v:,.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
print("✅ Plot 1.1: Sales by Product displayed")

# Visualization 1.2: Sales Distribution (Histogram) - SEPARATE PLOT
plt.figure(figsize=(12, 6))
plt.hist(ecommerce_df['total_sales'], bins=30, color='coral', edgecolor='black', alpha=0.7)
plt.title('Distribution of Order Values', fontsize=14, fontweight='bold')
plt.xlabel('Order Value ($)')
plt.ylabel('Frequency')
plt.axvline(ecommerce_df['total_sales'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: ${ecommerce_df["total_sales"].mean():.2f}')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
print("✅ Plot 1.2: Sales Distribution displayed")

# ============================================================================
# EXAMPLE 2: STUDENT PERFORMANCE ANALYSIS
# ============================================================================
print("\n📊 EXAMPLE 2: Student Performance Analysis")
print("-" * 80)

def generate_student_data(n_students=200):
    """Generate realistic student performance data"""

    subjects = ['Mathematics', 'Physics', 'Chemistry', 'English', 'History']

    data = {
        'student_id': range(1, n_students + 1),
        'name': [fake.name() for _ in range(n_students)],
        'age': np.random.randint(18, 25, n_students),
        'gender': np.random.choice(['Male', 'Female'], n_students),
        'study_hours': np.random.uniform(1, 10, n_students),
        'attendance': np.random.uniform(60, 100, n_students),
    }

    df = pd.DataFrame(data)

    # Generate scores correlated with study hours (NumPy operations)
    for subject in subjects:
        base_score = np.random.uniform(40, 60, n_students)
        study_bonus = df['study_hours'] * 3
        attendance_bonus = df['attendance'] * 0.2
        noise = np.random.normal(0, 5, n_students)

        df[subject] = np.clip(base_score + study_bonus + attendance_bonus + noise, 0, 100)

    # Calculate average score
    df['average_score'] = df[subjects].mean(axis=1)

    return df, subjects

students_df, subjects = generate_student_data(200)
print("\n📋 Data Sample:")
print(students_df.head())

# Visualization 2.1: Scatter Plot - Study Hours vs Performance - SEPARATE PLOT
plt.figure(figsize=(12, 6))
scatter = plt.scatter(students_df['study_hours'],
                     students_df['average_score'],
                     c=students_df['attendance'],
                     cmap='RdYlGn',
                     s=100,
                     alpha=0.6,
                     edgecolors='black',
                     linewidth=0.5)
plt.colorbar(scatter, label='Attendance %')
plt.title('Study Hours vs Average Score', fontsize=14, fontweight='bold')
plt.xlabel('Study Hours per Day')
plt.ylabel('Average Score')
plt.grid(True, alpha=0.3)

# Add trend line using NumPy
z = np.polyfit(students_df['study_hours'], students_df['average_score'], 1)
p = np.poly1d(z)
plt.plot(students_df['study_hours'].sort_values(),
         p(students_df['study_hours'].sort_values()),
         "r--", linewidth=2, label='Trend')
plt.legend()
plt.tight_layout()
plt.show()
print("✅ Plot 2.1: Study Hours vs Score displayed")

# Visualization 2.2: Subject Performance (Box Plot) - SEPARATE PLOT
plt.figure(figsize=(12, 6))
subject_data = [students_df[subject] for subject in subjects]
bp = plt.boxplot(subject_data, labels=subjects, patch_artist=True)

# Color boxes
colors_box = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)

plt.title('Score Distribution by Subject', fontsize=14, fontweight='bold')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Passing Grade')
plt.legend()
plt.tight_layout()
plt.show()
print("✅ Plot 2.2: Subject Performance displayed")

# ============================================================================
# EXAMPLE 3: WEATHER DATA ANALYSIS
# ============================================================================
print("\n📊 EXAMPLE 3: Weather Data Analysis")
print("-" * 80)

def generate_weather_data(n_days=365):
    """Generate realistic weather data for a year"""

    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]

    # Simulate seasonal temperature patterns using NumPy
    day_of_year = np.arange(n_days)
    base_temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365)  # Seasonal variation
    temp_noise = np.random.normal(0, 3, n_days)
    temperature = base_temp + temp_noise

    # Simulate humidity (inversely correlated with temperature)
    humidity = 70 - (temperature - 15) * 1.5 + np.random.normal(0, 5, n_days)
    humidity = np.clip(humidity, 20, 95)

    # Rainfall probability
    rainfall = np.random.exponential(scale=5, size=n_days)

    data = {
        'date': dates,
        'temperature': temperature,
        'humidity': humidity,
        'rainfall': rainfall,
        'month': [d.strftime('%B') for d in dates]
    }

    return pd.DataFrame(data)

weather_df = generate_weather_data(365)
print("\n📋 Data Sample:")
print(weather_df.head())

# Visualization 3.1: Temperature Over Time (Line Plot) - SEPARATE PLOT
plt.figure(figsize=(14, 6))
plt.plot(weather_df['date'], weather_df['temperature'], linewidth=1, color='darkred', alpha=0.7)
plt.fill_between(weather_df['date'], weather_df['temperature'], alpha=0.3, color='red')
plt.title('Temperature Variation Throughout the Year', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("✅ Plot 3.1: Temperature Over Time displayed")

# Visualization 3.2: Temperature vs Humidity (Scatter) - SEPARATE PLOT
plt.figure(figsize=(12, 6))
scatter = plt.scatter(weather_df['temperature'],
                     weather_df['humidity'],
                     c=weather_df['rainfall'],
                     cmap='Blues',
                     s=20,
                     alpha=0.5)
plt.colorbar(scatter, label='Rainfall (mm)')
plt.title('Temperature vs Humidity Relationship', fontsize=14, fontweight='bold')
plt.xlabel('Temperature (°C)')
plt.ylabel('Humidity (%)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print("✅ Plot 3.2: Temperature vs Humidity displayed")

# Visualization 3.3: Monthly Average Temperature (Bar Chart) - SEPARATE PLOT
plt.figure(figsize=(12, 6))
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
monthly_temp = weather_df.groupby('month')['temperature'].mean().reindex(month_order)
colors_temp = plt.cm.RdYlBu_r(np.linspace(0, 1, len(monthly_temp)))
plt.bar(monthly_temp.index, monthly_temp.values, color=colors_temp)
plt.title('Average Temperature by Month', fontsize=14, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Average Temperature (°C)')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
for i, v in enumerate(monthly_temp.values):
    plt.text(i, v, f'{v:.1f}°C', ha='center', va='bottom')
plt.tight_layout()
plt.show()
print("✅ Plot 3.3: Monthly Temperature displayed")

# ============================================================================
# EXAMPLE 4: SOCIAL MEDIA ENGAGEMENT ANALYSIS
# ============================================================================
print("\n📊 EXAMPLE 4: Social Media Engagement Analysis")
print("-" * 80)

def generate_social_media_data(n_posts=300):
    """Generate social media engagement data"""

    data = {
        'post_id': range(1, n_posts + 1),
        'username': [fake.user_name() for _ in range(n_posts)],
        'post_type': np.random.choice(['Image', 'Video', 'Text', 'Poll'], n_posts),
        'likes': np.random.poisson(lam=100, size=n_posts) + np.random.randint(0, 500, n_posts),
        'comments': np.random.poisson(lam=20, size=n_posts),
        'shares': np.random.poisson(lam=15, size=n_posts),
        'followers': np.random.exponential(scale=1000, size=n_posts) + 100,
        'time_of_day': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n_posts),
        'hashtags': np.random.randint(0, 15, n_posts)
    }

    df = pd.DataFrame(data)

    # Calculate engagement metrics
    df['total_engagement'] = df['likes'] + df['comments'] * 2 + df['shares'] * 3
    df['engagement_rate'] = (df['total_engagement'] / df['followers']) * 100

    return df

social_df = generate_social_media_data(300)
print("\n📋 Data Sample:")
print(social_df.head())

# Visualization 4.1: Engagement by Post Type (Stacked Bar) - SEPARATE PLOT
plt.figure(figsize=(12, 6))
post_engagement = social_df.groupby('post_type')[['likes', 'comments', 'shares']].mean()
post_engagement.plot(kind='bar', stacked=True, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], ax=plt.gca())
plt.title('Average Engagement by Post Type', fontsize=14, fontweight='bold')
plt.xlabel('Post Type')
plt.ylabel('Average Count')
plt.legend(title='Engagement Type')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
print("✅ Plot 4.1: Engagement by Post Type displayed")

# Visualization 4.2: Engagement by Time of Day (Box Plot) - SEPARATE PLOT
plt.figure(figsize=(12, 6))
time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
time_data = [social_df[social_df['time_of_day'] == time]['engagement_rate'] for time in time_order]
bp = plt.boxplot(time_data, labels=time_order, patch_artist=True)
colors_time = ['#FFE66D', '#FF6B6B', '#4ECDC4', '#95E1D3']
for patch, color in zip(bp['boxes'], colors_time):
    patch.set_facecolor(color)
plt.title('Engagement Rate Distribution by Time of Day', fontsize=14, fontweight='bold')
plt.ylabel('Engagement Rate (%)')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
print("✅ Plot 4.2: Engagement by Time displayed")

# Visualization 4.3: Followers vs Likes (Scatter) - SEPARATE PLOT
plt.figure(figsize=(12, 6))
scatter = plt.scatter(social_df['followers'],
                     social_df['likes'],
                     c=social_df['post_type'].astype('category').cat.codes,
                     cmap='Set1',
                     alpha=0.6,
                     s=50)
plt.title('Followers vs Likes', fontsize=14, fontweight='bold')
plt.xlabel('Number of Followers')
plt.ylabel('Likes')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print("✅ Plot 4.3: Followers vs Likes displayed")

# Visualization 4.4: Engagement Rate Distribution (Histogram) - SEPARATE PLOT
plt.figure(figsize=(12, 6))
plt.hist(social_df['engagement_rate'], bins=30, color='mediumseagreen', edgecolor='black', alpha=0.7)
plt.title('Distribution of Engagement Rates', fontsize=14, fontweight='bold')
plt.xlabel('Engagement Rate (%)')
plt.ylabel('Frequency')
plt.axvline(social_df['engagement_rate'].median(), color='red', linestyle='--', linewidth=2,
           label=f'Median: {social_df["engagement_rate"].median():.2f}%')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
print("✅ Plot 4.4: Engagement Rate Distribution displayed")

# ============================================================================
# EXAMPLE 5: COMPREHENSIVE DASHBOARD (SEPARATED PLOTS)
# ============================================================================
print("\n📊 EXAMPLE 5: Sales Dashboard (All Chart Types - Separated)")
print("-" * 80)

def generate_comprehensive_sales_data():
    """Generate comprehensive sales data for dashboard"""

    n_transactions = 1000

    data = {
        'date': pd.date_range(start='2024-01-01', periods=n_transactions, freq='H'),
        'sales_rep': [fake.name() for _ in range(n_transactions)],
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_transactions),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books', 'Toys'], n_transactions),
        'amount': np.random.gamma(shape=2, scale=50, size=n_transactions),
        'units_sold': np.random.poisson(lam=3, size=n_transactions) + 1,
        'customer_satisfaction': np.random.choice([1, 2, 3, 4, 5], n_transactions, p=[0.05, 0.1, 0.15, 0.4, 0.3])
    }

    df = pd.DataFrame(data)
    df['revenue'] = df['amount'] * df['units_sold']
    df['month'] = df['date'].dt.month_name()
    df['day_of_week'] = df['date'].dt.day_name()

    return df

dashboard_df = generate_comprehensive_sales_data()

# Plot 5.1: Daily Revenue Trend (Line Chart) - SEPARATE PLOT
plt.figure(figsize=(14, 6))
daily_revenue = dashboard_df.groupby(dashboard_df['date'].dt.date)['revenue'].sum()
plt.plot(daily_revenue.index, daily_revenue.values, linewidth=2, color='darkblue')
plt.fill_between(daily_revenue.index, daily_revenue.values, alpha=0.3)
plt.title('Daily Revenue Trend', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Revenue ($)')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("✅ Plot 5.1: Daily Revenue Trend displayed")

# Plot 5.2: Revenue by Region (Horizontal Bar Chart) - SEPARATE PLOT
plt.figure(figsize=(12, 6))
region_revenue = dashboard_df.groupby('region')['revenue'].sum().sort_values(ascending=False)
bars = plt.barh(region_revenue.index, region_revenue.values, color='teal')
plt.title('Revenue by Region', fontsize=14, fontweight='bold')
plt.xlabel('Revenue ($)')
plt.grid(axis='x', alpha=0.3)
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, f'${width:,.0f}',
             ha='left', va='center', fontsize=10)
plt.tight_layout()
plt.show()
print("✅ Plot 5.2: Revenue by Region displayed")

# Plot 5.3: Sales by Category (Pie Chart) - SEPARATE PLOT
plt.figure(figsize=(10, 10))
category_sales = dashboard_df.groupby('product_category')['units_sold'].sum()
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(category_sales)))
wedges, texts, autotexts = plt.pie(category_sales.values, labels=category_sales.index,
                                     autopct='%1.1f%%', colors=colors_pie, startangle=90)
plt.title('Sales Distribution by Category', fontsize=14, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
plt.tight_layout()
plt.show()
print("✅ Plot 5.3: Sales Distribution displayed")

# Plot 5.4: Transaction Amount Distribution (Histogram) - SEPARATE PLOT
plt.figure(figsize=(12, 6))
plt.hist(dashboard_df['amount'], bins=40, color='coral', edgecolor='black', alpha=0.7)
plt.title('Transaction Amount Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Amount ($)')
plt.ylabel('Frequency')
plt.axvline(dashboard_df['amount'].mean(), color='red', linestyle='--',
            linewidth=2, label=f'Mean: ${dashboard_df["amount"].mean():.2f}')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
print("✅ Plot 5.4: Transaction Amount Distribution displayed")

# Plot 5.5: Units Sold vs Revenue (Scatter) - SEPARATE PLOT
plt.figure(figsize=(12, 6))
scatter = plt.scatter(dashboard_df['units_sold'], dashboard_df['revenue'],
                     c=dashboard_df['customer_satisfaction'], cmap='RdYlGn',
                     s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, label='Satisfaction')
plt.title('Units Sold vs Revenue', fontsize=14, fontweight='bold')
plt.xlabel('Units Sold')
plt.ylabel('Revenue ($)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print("✅ Plot 5.5: Units Sold vs Revenue displayed")

# Plot 5.6: Revenue by Day of Week (Box Plot) - SEPARATE PLOT
plt.figure(figsize=(12, 6))
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_data = [dashboard_df[dashboard_df['day_of_week'] == day]['revenue'] for day in days_order if day in dashboard_df['day_of_week'].unique()]
day_labels = [day[:3] for day in days_order if day in dashboard_df['day_of_week'].unique()]
bp = plt.boxplot(day_data, labels=day_labels, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
plt.title('Revenue Distribution by Day', fontsize=14, fontweight='bold')
plt.ylabel('Revenue ($)')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
print("✅ Plot 5.6: Revenue by Day displayed")

# Plot 5.7: Satisfaction Heatmap (Region × Category) - SEPARATE PLOT
plt.figure(figsize=(12, 8))
satisfaction_matrix = dashboard_df.pivot_table(
    values='customer_satisfaction',
    index='region',
    columns='product_category',
    aggfunc='mean'
)
im = plt.imshow(satisfaction_matrix.values, cmap='RdYlGn', aspect='auto')
plt.xticks(np.arange(len(satisfaction_matrix.columns)), satisfaction_matrix.columns, rotation=45, ha='right')
plt.yticks(np.arange(len(satisfaction_matrix.index)), satisfaction_matrix.index)
plt.title('Average Satisfaction: Region × Category', fontsize=14, fontweight='bold')
plt.colorbar(im, label='Satisfaction')

# Add annotations to heatmap
for i in range(len(satisfaction_matrix.index)):
    for j in range(len(satisfaction_matrix.columns)):
        text = plt.text(j, i, f'{satisfaction_matrix.values[i, j]:.1f}',
                       ha="center", va="center", color="black", fontsize=10)
plt.tight_layout()
plt.show()
print("✅ Plot 5.7: Satisfaction Heatmap displayed")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("📈 WORKSHOP COMPLETE!")
print("=" * 80)
print("\n✅ Successfully created separated plots for all 5 examples:")
print("   1. E-Commerce Sales Analysis (2 plots)")
print("   2. Student Performance Analysis (2 plots)")
print("   3. Weather Data Analysis (3 plots)")
print("   4. Social Media Engagement Analysis (4 plots)")
print("   5. Comprehensive Sales Dashboard (7 plots)")
print(f"\n📊 Total Individual Plots: 18")
print("=" * 80)