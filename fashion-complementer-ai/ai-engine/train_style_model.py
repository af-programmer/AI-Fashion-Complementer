import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# --- פתרון שגיאות SSL ברשתות חסומות/אקדמיות ---
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# ----------------------------------------------

# 1. הגדרות בסיסיות של הפרמטרים
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 10  # מספר מחזורי האימון (ניתן להגדיל בהתאם לצורך ולכמות הדאטה)
DATASET_DIR = '../data_split' # עודכן לנתיב התיקייה המאוזנת החדשה

print("--- שלב 1: טעינת הנתונים וביצוע אוגמנטציה (Data Augmentation) ---")

# מחולל נתונים עבור קבוצת האימון - כולל גיוון זמני בתמונות (סיבוב, מתיחה) למניעת Overfitting
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# מחולל נתונים עבור קבוצת הוולידציה - ללא אוגמנטציה (בדיקה נקייה של המודל)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# מחולל נתונים עבור קבוצת האימון מתוך תת-התיקייה train
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# מחולל נתונים עבור קבוצת הוולידציה מתוך תת-התיקייה val
validation_generator = val_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"מיפוי הקטגוריות שנמצאו: {train_generator.class_indices}")

print("\n--- שלב 2: בניית המודל הארכיטקטוני (Transfer Learning) ---")

# טעינת מודל הבסיס ResNet50 ללא שכבות הפלט העליונות
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# הקפאת שכבות הבסיס כדי שהמשקולות המקוריות שלהן לא ישתנו במהלך האימון הראשוני
for layer in base_model.layers:
    layer.trainable = False

# הוספת השכבות החדשות המותאמות לפרויקט שלנו לקצה המודל
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # שכבה נסתרת להבנת מאפיינים מורכבים
predictions = Dense(4, activation='softmax')(x)  # שכבת פלט עבור 4 סגנונות

# הרכבת המודל הסופי
model = Model(inputs=base_model.input, outputs=predictions)

# קומפילציה (הגדרה) של המודל: פונקציית הפסד, אופטימייזר וציון המדד (Accuracy)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- שלב 3: תחילת תהליך האימון בפועל ---")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

print("\n--- שלב 4: שמירת המודל המאומן לדיסק ---")
# שמירת המודל המוכן לקובץ בפורמט קראס (.h5) כדי שנוכל להשתמש בו לניבויים מהירים
model_save_path = 'fashion_style_model.h5'
model.save(model_save_path)
print(f"המודל אומן בהצלחה ונשמר בנתיב: '{model_save_path}'")