import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# --- תיקון נתיבים דינמי ואוטומטי ---
# מוצא את התיקייה שבה קובץ הקוד הנוכחי נמצא (ai-engine)
current_dir = os.path.dirname(os.path.abspath(__file__))
# עולה תיקייה אחת למעלה לתיקייה הראשית של ה-Repository
project_root = os.path.dirname(current_dir)

# הגדרת נתיבים מוחלטים דינמיים
csv_path = os.path.join(project_root, "data", "styles.csv")
source_images_dir = os.path.join(project_root, "data", "images")
output_base_dir = os.path.join(project_root, "data_split")
# -------------------------------------

# טעינת המידע
df = pd.read_csv(csv_path, on_bad_lines='skip')

# מיפוי וסינון ארבעת הסגנונות שלכן
style_mapping = {
    'Casual': 'Casual',
    'Formal': 'Formal',
    'Sports': 'Sporty',
    'Ethnic': 'Boho'
}
df_filtered = df[df['usage'].isin(style_mapping.keys())].copy()
df_filtered['target_style'] = df_filtered['usage'].map(style_mapping)

# פתרון חוסר האיזון - לקיחת מקסימום 2000 תמונות מכל סוג
max_samples_per_class = 2000
df_balanced = df_filtered.groupby('target_style').apply(
    lambda x: x.sample(n=min(len(x), max_samples_per_class), random_state=42)
).reset_index(drop=True)

# חלוקה ל-Train (80%) ו-Validation (20%)
train_df, val_df = train_test_split(df_balanced, test_size=0.2, stratify=df_balanced['target_style'], random_state=42)


def distribute_images(dataset_df, subset_name):
    for _, row in dataset_df.iterrows():
        img_id = row['id']
        style = row['target_style']
        img_name = f"{img_id}.jpg"
        source_img_path = os.path.join(source_images_dir, img_name)

        if os.path.exists(source_img_path):
            target_dir = os.path.join(output_base_dir, subset_name, style)
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy(source_img_path, os.path.join(target_dir, img_name))


print("מתחיל מיון והעתקה מקומית...")
distribute_images(train_df, "train")
distribute_images(val_df, "val")
print("הסתיים! נוצרה תיקיית 'data_split' מאוזנת ומוכנה לאימון המודל.")