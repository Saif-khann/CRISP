import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, InceptionV3, VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

def create_stage_model(base_model, num_classes, input_shape=(224, 224, 3)):
    """Create a model for stage classification."""
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def build_and_save_models():
    print("Building models...")
    
    # Global stage classification models (ensemble)
    print("Building MobileNetV2...")
    mobilenet_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    mobilenet = create_stage_model(mobilenet_base, num_classes=5)  # 5 main stages
    mobilenet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    mobilenet.save('models/mobilenet.keras')
    
    print("Building InceptionV3...")
    inception_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    inception = create_stage_model(inception_base, num_classes=5)
    inception.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    inception.save('models/inception.keras')
    
    print("Building VGG16...")
    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    vgg = create_stage_model(vgg_base, num_classes=5)
    vgg.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    vgg.save('models/vgg16.keras')

    # NOTE: this stage-specific block is illustrative scaffolding, not what
    # actually produced the models currently shipped in models/. The real
    # ones were trained with stage-appropriate class counts and, for
    # "facade" specifically, an InceptionV3 base (Facade_inception.keras) - # not MobileNetV2 - plus filenames capitalized to match app.py's
    # load_models() (e.g. "Foundation_mobile.keras", not
    # "foundation_mobile.keras"). Adjust base/num_classes/filename per
    # stage to match your actual training data before re-running this.
    stages = ['foundation', 'superstructure', 'facade', 'Interior', 'finishing works']
    substages = [3, 3, 3, 3, 3]  # Number of sub-stages for each stage

    for stage, num_classes in zip(stages, substages):
        print(f"Building {stage} model...")
        base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        model = create_stage_model(base, num_classes=num_classes)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.save(f'models/{stage}_mobile.keras')
    
    print("All models built and saved successfully!")

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)
    
    # Build and save all models
    build_and_save_models()
